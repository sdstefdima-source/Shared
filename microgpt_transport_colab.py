#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# Frozen-Weight Transport Microscope + Instruct Chat Window
# Pandas-free single-cell Colab script
#
# No training.
# No pandas.
# Loads a frozen pretrained instruct causal LM.
# Runs transport diagnostics.
# Saves CSV / JSON / ZIP outputs.
# Opens a small chat UI after the tests.
#
# This script is a frozen-weight transport microscope.
# It does not train the model. It observes one forward process:
# text -> tokens -> additive hidden states -> attention kernels -> output probabilities -> path action.
#
# Operator view:
# The model is treated as a composition of operators. Embedding maps tokens into vectors.
# Linear maps produce queries, keys, values, hidden updates, and logits. Softmax maps additive
# scores into simplex-valued probability rows. Attention rows act as stochastic transport kernels.
#
# Modifier view:
# Checkboxes select which extra measurements modify the base observation pass. A selected metric
# does not change the model weights or generation algorithm. It only changes which intermediate
# quantities are extracted, summarized, and written to CSV/JSON/chat diagnostics.
#
# Pedagogical view:
# Every metric answers one question about the run: where the token path became surprising, how
# sharply probabilities concentrated, how far attention transported mass, how much hidden states
# moved, whether heads agreed, and whether the output distribution behaved like a stable Gibbs row.
# ============================================================

# -----------------------------
# 0. Install dependencies
# -----------------------------

# Script-compatible dependency installation.
# In Colab this can be run with: %run microgpt_transport_colab.py
# or copied into a single code cell.
import sys
import subprocess

def ensure_packages():
    packages = ["transformers", "accelerate", "safetensors", "ipywidgets"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", *packages])

ensure_packages()

# -----------------------------
# 1. Imports and configuration
# -----------------------------

import os
import csv
import json
import math
import html
import zipfile
import warnings
from datetime import datetime

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from IPython.display import display, clear_output
import ipywidgets as widgets

warnings.filterwarnings("ignore")

# Default: small instruct model, better for CPU chat than a base completion model.
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Other option:
# MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

OUTPUT_DIR = "outputs_transport_probe"
ZIP_NAME = "transport_probe_outputs.zip"

MAX_TOKENS = 128
TOP_K = 5
EPS = 1e-12

CHAT_MAX_NEW_TOKENS = 160
CHAT_TEMPERATURE = 0.65
CHAT_TOP_P = 0.90
CHAT_REPETITION_PENALTY = 1.10

# ============================================================
# 1A. Metric registry and default checkbox state
# ============================================================
# The checkbox panels are the main control surface.
# Edit these True/False values only to change which boxes start checked.
# The report panel and chat panel can still be changed at runtime.

ADVANCED_METRIC_FLAGS = {
    "action_spike_index": False,
    "target_rank": False,
    "confidence_error_split": False,
    "attention_transport_distance": False,
    "attention_concentration_radius": False,
    "head_disagreement": False,
    "head_consensus": False,
    "layer_work_ratio": False,
    "hidden_curvature": False,
    "residual_overwrite_proxy": False,
    "mlp_activation_sparsity": False,
    "prompt_retention_generation": False,
    "effective_support": False,
    "top_margin": False,
    "output_js_drift": False,
    "token_attention_influence": False,
    "probability_curvature": False,
    "temperature_sensitivity": False,
    "output_entropy_delta": False,
    "attention_spectral_gap": False,
    "attention_stationarity_error": False,
    "attention_reversibility_error": False,
    "cross_head_source_mi": False,
    "causal_chokepoint_score": False,
    "attention_entropy_variance": False,
    "hidden_anisotropy": False,
    "hidden_effective_rank": False,
    "output_distribution_acceleration": False,
    "repetition_loop_index": False,
    "free_energy_variance": False,
    "integrity_checks": False,
    "layer_contraction_expansion": False,
    "prompt_pair_separation": False,
    "attractor_distance_blend": False,
    "sliding_window_markov_cost": False,
    "cache_value_drift": False,
    "gradient_norm": False,
    "gradient_alignment": False,
    "adam_update_diagnostics": False,
    "parameter_heatmap": False,
}

ADVANCED_METRICS = {
    name for name, enabled in ADVANCED_METRIC_FLAGS.items()
    if enabled
}

# Kept for compatibility with older notebook edits.
def ENABLE_ADVANCED_METRIC(name):
    ADVANCED_METRICS.add(name)

ACTION_SPIKE_Z_THRESHOLD = 2.0
MLP_ACTIVATION_THRESHOLD = 1e-6
EPS_METRIC = 1e-12

# ------------------------------------------------------------
# Probe prompts
# ------------------------------------------------------------
# These prompts are not random examples. They are calibration probes.
#
# normal_english:
#   Baseline ordinary sentence. Tests whether path action, entropy, attention, and hidden motion
#   look stable on plain text.
#
# random_noise:
#   Broken local-symbol structure. Should usually raise entropy, path action, or token-rank failure.
#   Useful for checking whether the microscope distinguishes noise from language.
#
# technical_transport:
#   Dense technical vocabulary about the same framework. Useful for seeing confident-wrong regions:
#   entropy can be low while path action is high if the model strongly expects a different term.
#
# markov_style:
#   Repetitive local transition structure. Useful for seeing short-range attention, repetition tracking,
#   and low-cost local token prediction.
#
# cipher_style:
#   Hybrid technical/cipher-language phrase. Useful for testing transition drift, rare terms, and
#   whether the model treats symbolic-analysis language as ordinary prose or specialized structure.

PROMPTS = {
    "normal_english": (
        "The machine converts scores into probabilities and then predicts the next token."
    ),
    "random_noise": (
        "xqj zzp %% 19aa qvnnn @@ lmzqx 7733"
    ),
    "technical_transport": (
        "Log transport turns multiplicative path weights into additive path action. "
        "Attention turns tensor scores into Gibbs-normalized stochastic transport."
    ),
    "markov_style": (
        "the cat sat the cat saw the cat ran the cat sat the cat saw"
    ),
    "cipher_style": (
        "A sliding window Markov chain scores local symbol transitions and detects drift from the reference attractor."
    ),
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("device:", device)
print("dtype:", dtype)
print("model:", MODEL_ID)
print("pandas: not used")

# -----------------------------
# 2. Small table and save helpers
# -----------------------------
# Operator:
#   Convert row dictionaries into readable tables and durable CSV/JSON artifacts.
#
# Modifier:
#   The fieldnames are discovered dynamically. If a checkbox metric adds a new key,
#   write_csv(...) automatically includes it without changing the output writer.
#
# Pedagogical note:
#   The microscope is meant to evolve. Metrics can be added as new columns without
#   rebuilding the reporting system.

def write_csv(path, rows):
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_table(rows, columns=None, limit=25, title=None):
    if title:
        print("\n" + title)

    if not rows:
        print("No rows.")
        return

    if columns is None:
        columns = list(rows[0].keys())

    rows = rows[:limit]

    widths = {}
    for col in columns:
        max_width = len(str(col))
        for row in rows:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.5g}"
            max_width = max(max_width, len(str(val)))
        widths[col] = min(max_width, 28)

    header = " | ".join(str(col)[:widths[col]].ljust(widths[col]) for col in columns)
    print(header)
    print("-" * len(header))

    for row in rows:
        line = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.5g}"
            text = str(val).replace("\n", "\\n")
            if len(text) > widths[col]:
                text = text[:widths[col] - 1] + "…"
            line.append(text.ljust(widths[col]))
        print(" | ".join(line))


def sort_rows(rows, key, reverse=True):
    return sorted(rows, key=lambda r: float(r.get(key, 0) or 0), reverse=reverse)


def mean_or_none(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


CURRENT_ENABLED_METRICS = set(ADVANCED_METRICS)


def set_metric_context(metric_names):
    global CURRENT_ENABLED_METRICS
    CURRENT_ENABLED_METRICS = set(metric_names)


def metric_enabled(name):
    return name in CURRENT_ENABLED_METRICS


def safe_mean(values):
    clean = []
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except Exception:
            continue
        if math.isfinite(value):
            clean.append(value)
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def safe_max(values):
    clean = []
    for value in values:
        if value is None:
            continue
        try:
            value = float(value)
        except Exception:
            continue
        if math.isfinite(value):
            clean.append(value)
    if not clean:
        return None
    return float(max(clean))

# -----------------------------
# 3. Load tokenizer and model
# -----------------------------
# Operator:
#   Load a frozen causal language model and tokenizer. The model is set to eval mode.
#
# Modifier:
#   dtype and device change speed/memory, not the diagnostic logic. output_attentions=True
#   and output_hidden_states=True are activated during probe calls, not during loading.
#
# Pedagogical note:
#   This is a microscope over an existing model, not a trainer. No gradient step, optimizer,
#   dataset update, or weight write occurs.

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
except Exception:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

model.to(device)
model.eval()

print("model loaded")
print("parameters:", sum(p.numel() for p in model.parameters()))

# -----------------------------
# 4. Core metric functions
# -----------------------------
# Operator:
#   These functions are the base instruments. They measure entropy, KL divergence,
#   probability normalization, token decoding, attention drift, attention locality,
#   self-attention, cosine movement, L2 movement, and RMS-normalized hidden geometry.
#
# Modifier:
#   They do not change model behavior. They only transform captured tensors into scalar
#   readings that can be compared across prompts, layers, heads, and token positions.
#
# Pedagogical note:
#   Base metrics separate the three main spaces: additive hidden-vector space, simplex
#   probability space, and stochastic attention-kernel space.

def entropy_from_probs(p, dim=-1):
    p = p.clamp_min(EPS)
    return -(p * p.log()).sum(dim=dim)


def kl_divergence(p, q, dim=-1):
    p = p.clamp_min(EPS)
    q = q.clamp_min(EPS)
    return (p * (p.log() - q.log())).sum(dim=dim)


def normalize_distribution(p):
    p = p.clamp_min(EPS)
    return p / p.sum(dim=-1, keepdim=True).clamp_min(EPS)


def decode_one_token(token_id):
    return tokenizer.decode([int(token_id)])


def token_text_list(input_ids_1d):
    ids = input_ids_1d.detach().cpu().tolist()
    return [tokenizer.convert_ids_to_tokens(i) for i in ids]


def topk_predictions(logits_1d, k=5):
    probs = F.softmax(logits_1d.float(), dim=-1)
    vals, ids = torch.topk(probs, k=k, dim=-1)

    rows = []
    for prob, idx in zip(vals.detach().cpu().tolist(), ids.detach().cpu().tolist()):
        rows.append({
            "token_text": decode_one_token(idx),
            "token_id": int(idx),
            "prob": float(prob),
        })

    return rows


def attention_common_prefix_kl(attn_head):
    seq = attn_head.shape[0]
    if seq <= 1:
        return 0.0

    vals = []
    for t in range(1, seq):
        p = attn_head[t, :t]
        q = attn_head[t - 1, :t]
        p = normalize_distribution(p)
        q = normalize_distribution(q)
        vals.append(kl_divergence(p, q, dim=-1))

    return float(torch.stack(vals).mean().item()) if vals else 0.0


def attention_recent_mass(attn_head, recent_width=4):
    seq = attn_head.shape[0]
    if seq == 0:
        return 0.0

    vals = []
    for t in range(seq):
        start = max(0, t - recent_width + 1)
        vals.append(attn_head[t, start:t + 1].sum())

    return float(torch.stack(vals).mean().item())


def attention_self_mass(attn_head):
    if attn_head.shape[0] == 0:
        return 0.0
    return float(torch.diagonal(attn_head, dim1=-2, dim2=-1).mean().item())


def cosine_mean_adjacent(hidden_2d):
    if hidden_2d.shape[0] <= 1:
        return 0.0

    a = hidden_2d[:-1]
    b = hidden_2d[1:]
    cos = F.cosine_similarity(a, b, dim=-1)
    return float(cos.mean().item())


def l2_mean_adjacent(hidden_2d):
    if hidden_2d.shape[0] <= 1:
        return 0.0

    diff = hidden_2d[1:] - hidden_2d[:-1]
    return float(diff.norm(dim=-1).mean().item())


def rms_normalized_hidden(hidden_2d):
    h = hidden_2d.float()
    rms = torch.sqrt((h ** 2).mean(dim=-1, keepdim=True).clamp_min(EPS))
    return h / rms

# -----------------------------
# 4A. Advanced metric functions
# -----------------------------
# Operator:
#   These functions extract second-layer structure from the same frozen forward pass.
#   They measure surprise spikes, target rank, confidence/error regimes, transport distance,
#   head disagreement, hidden curvature, probability curvature, spectral attention behavior,
#   stationarity, reversibility, hidden effective rank, and replay/generation stability.
#
# Modifier:
#   They are checkbox-gated. When disabled, the run does not compute their heavier values.
#   When enabled, extra columns are added to token_rows, attention_rows, hidden_rows, and summary.
#
# Pedagogical note:
#   The advanced layer turns simple observation into geometry. A scalar loss says whether a
#   sequence was expensive. These probes explain where and in which carrier that expense appears.

def target_rank_from_logits(logits_1d, target_token_id):
    target_score = logits_1d[int(target_token_id)]
    return int((logits_1d > target_score).sum().item() + 1)


def action_spike_zscores(token_nll_1d):
    if token_nll_1d.numel() == 0:
        return torch.empty_like(token_nll_1d)
    mean = token_nll_1d.mean()
    std = token_nll_1d.std(unbiased=False).clamp_min(EPS_METRIC)
    return (token_nll_1d - mean) / std


def confidence_error_regime(nll_value, entropy_value, nll_cutoff, entropy_cutoff):
    high_action = nll_value >= nll_cutoff
    high_entropy = entropy_value >= entropy_cutoff

    if not high_entropy and not high_action:
        return "confident_correct_or_expected"
    if not high_entropy and high_action:
        return "confident_wrong_or_misaligned"
    if high_entropy and high_action:
        return "diffuse_uncertain_and_costly"
    return "broad_distribution_but_target_fit"


def attention_transport_distance(attn_head):
    seq = attn_head.shape[0]
    if seq == 0:
        return 0.0

    idx = torch.arange(seq, device=attn_head.device, dtype=attn_head.dtype)
    vals = []
    for t in range(seq):
        dist = (t - idx).abs()
        vals.append((attn_head[t] * dist).sum())

    return float(torch.stack(vals).mean().item())


def attention_concentration_radius(attn_head):
    seq = attn_head.shape[0]
    if seq == 0:
        return 0.0

    idx = torch.arange(seq, device=attn_head.device, dtype=attn_head.dtype)
    vals = []
    for t in range(seq):
        row = normalize_distribution(attn_head[t])
        center = (row * idx).sum()
        radius = torch.sqrt((row * ((idx - center) ** 2)).sum().clamp_min(EPS_METRIC))
        vals.append(radius)

    return float(torch.stack(vals).mean().item())


def js_divergence(p, q):
    p = normalize_distribution(p)
    q = normalize_distribution(q)
    m = normalize_distribution(0.5 * (p + q))
    return 0.5 * kl_divergence(p, m, dim=-1) + 0.5 * kl_divergence(q, m, dim=-1)


def layer_head_js_disagreement(layer_attn):
    heads = layer_attn.shape[0]
    if heads <= 1:
        return 0.0

    vals = []
    for h in range(heads):
        for g in range(h + 1, heads):
            vals.append(js_divergence(layer_attn[h], layer_attn[g]).mean())

    return float(torch.stack(vals).mean().item()) if vals else 0.0


def layer_head_consensus(layer_attn):
    if layer_attn.numel() == 0:
        return 0.0

    mean_over_heads = layer_attn.mean(dim=0)
    token_consensus = mean_over_heads.max(dim=-1).values
    return float(token_consensus.mean().item())


def hidden_curvature_mean(hidden_2d):
    if hidden_2d.shape[0] <= 2:
        return 0.0

    v0 = hidden_2d[1:-1] - hidden_2d[:-2]
    v1 = hidden_2d[2:] - hidden_2d[1:-1]
    bend = (v1 - v0).norm(dim=-1)
    base = v0.norm(dim=-1).clamp_min(EPS_METRIC)
    return float((bend / base).mean().item())


def output_js_drift_mean(probs_3d):
    if probs_3d.shape[1] <= 1:
        return 0.0
    p = probs_3d[:, 1:, :]
    q = probs_3d[:, :-1, :]
    return float(js_divergence(p, q).mean().item())


def output_js_acceleration_mean(probs_3d):
    if probs_3d.shape[1] <= 2:
        return 0.0
    drift = js_divergence(probs_3d[:, 1:, :], probs_3d[:, :-1, :]).mean(dim=0)
    accel = drift[1:] - drift[:-1]
    return float(accel.abs().mean().item()) if accel.numel() else 0.0


def probability_curvature_trace_mean(probs_3d):
    if probs_3d.numel() == 0:
        return 0.0
    # Trace(Diag(p) - pp^T) = 1 - sum_i p_i^2 for each probability row.
    return float((1.0 - (probs_3d ** 2).sum(dim=-1)).mean().item())


def temperature_entropy_sensitivity(logits_3d, delta=0.05):
    if logits_3d.numel() == 0:
        return 0.0

    tau_low = max(0.05, 1.0 - delta)
    tau_high = 1.0 + delta
    p_low = F.softmax(logits_3d / tau_low, dim=-1).clamp_min(EPS)
    p_high = F.softmax(logits_3d / tau_high, dim=-1).clamp_min(EPS)
    h_low = entropy_from_probs(p_low, dim=-1)
    h_high = entropy_from_probs(p_high, dim=-1)
    return float(((h_high - h_low).abs() / (tau_high - tau_low)).mean().item())


def output_entropy_delta_stats(output_entropy_2d):
    if output_entropy_2d.numel() <= 1 or output_entropy_2d.shape[1] <= 1:
        return {"avg_output_entropy_delta": 0.0, "output_entropy_delta_variance": 0.0}
    delta = output_entropy_2d[:, 1:] - output_entropy_2d[:, :-1]
    return {
        "avg_output_entropy_delta": float(delta.mean().item()),
        "output_entropy_delta_variance": float(delta.var(unbiased=False).item()),
    }


def attention_spectral_gap(attn_head):
    seq = attn_head.shape[0]
    if seq <= 1:
        return 0.0
    p = normalize_distribution(attn_head).detach().float().cpu()
    try:
        eigvals = torch.linalg.eigvals(p).abs().real
        eigvals_sorted = torch.sort(eigvals, descending=True).values
        lambda2 = float(eigvals_sorted[1].item()) if eigvals_sorted.numel() > 1 else 0.0
        return float(max(0.0, 1.0 - lambda2))
    except Exception:
        return None


def attention_stationary_distribution(attn_head, steps=64):
    seq = attn_head.shape[0]
    if seq == 0:
        return None
    p = normalize_distribution(attn_head).detach().float().cpu()
    pi = torch.ones(seq, dtype=torch.float32) / max(seq, 1)
    for _ in range(steps):
        pi = pi @ p
        pi = pi / pi.sum().clamp_min(EPS_METRIC)
    return pi, p


def attention_stationarity_error(attn_head):
    result = attention_stationary_distribution(attn_head)
    if result is None:
        return 0.0
    pi, p = result
    err = (pi @ p - pi).abs().sum()
    return float(err.item())


def attention_reversibility_error(attn_head):
    result = attention_stationary_distribution(attn_head)
    if result is None:
        return 0.0
    pi, p = result
    flow = pi[:, None] * p
    reverse_flow = pi[None, :] * p.T
    return float((flow - reverse_flow).abs().sum().item())


def cross_head_source_mi(layer_attn):
    # p(h,s) is total attention mass head h sends to source position s across all query rows.
    heads, seq, _ = layer_attn.shape
    if heads <= 1 or seq == 0:
        return 0.0
    mass = layer_attn.sum(dim=1).clamp_min(EPS)  # heads x source_positions
    joint = mass / mass.sum().clamp_min(EPS)
    ph = joint.sum(dim=1, keepdim=True)
    ps = joint.sum(dim=0, keepdim=True)
    expected = (ph @ ps).clamp_min(EPS)
    return float((joint * (joint.clamp_min(EPS).log() - expected.log())).sum().item())


def hidden_covariance_eigs(hidden_2d):
    if hidden_2d.shape[0] <= 1:
        return torch.ones(1)
    h = hidden_2d.float()
    centered = h - h.mean(dim=0, keepdim=True)
    gram = centered @ centered.T
    denom = max(hidden_2d.shape[0] - 1, 1)
    eigs = torch.linalg.eigvalsh((gram / denom).detach().cpu()).clamp_min(0.0)
    return eigs


def hidden_anisotropy_value(hidden_2d):
    eigs = hidden_covariance_eigs(hidden_2d)
    total = eigs.sum().clamp_min(EPS_METRIC)
    return float((eigs.max() / total).item())


def hidden_effective_rank_value(hidden_2d):
    eigs = hidden_covariance_eigs(hidden_2d)
    total = eigs.sum().clamp_min(EPS_METRIC)
    weights = (eigs / total).clamp_min(EPS_METRIC)
    entropy = -(weights * weights.log()).sum()
    return float(torch.exp(entropy).item())


def repetition_loop_index_for_text(text, window=24):
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max(window * 2, 32))["input_ids"][0]
    if ids.numel() <= 2:
        return 0.0
    ids = ids[-window:]
    unique = len(set(int(x) for x in ids.tolist()))
    repetition_rate = 1.0 - (unique / max(int(ids.numel()), 1))
    return float(max(0.0, min(1.0, repetition_rate)))


def tensor_stats_for_activation(tensor, module_name):
    t = tensor.detach().float()
    if t.numel() == 0:
        return None
    active = (t.abs() > MLP_ACTIVATION_THRESHOLD).float().mean().item()
    return {
        "module": module_name,
        "active_fraction": float(active),
        "abs_mean": float(t.abs().mean().item()),
        "abs_max": float(t.abs().max().item()),
    }


def flatten_tensors(obj):
    if torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, (tuple, list)):
        out = []
        for item in obj:
            out.extend(flatten_tensors(item))
        return out
    if isinstance(obj, dict):
        out = []
        for item in obj.values():
            out.extend(flatten_tensors(item))
        return out
    return []


def install_mlp_activation_hooks(model_obj, activation_records):
    hooks = []

    def make_hook(name):
        def hook(_module, _inputs, output):
            for tensor in flatten_tensors(output):
                stats = tensor_stats_for_activation(tensor, name)
                if stats is not None:
                    activation_records.append(stats)
        return hook

    for name, module in model_obj.named_modules():
        lname = name.lower()
        # Qwen-style modules usually expose model.layers.N.mlp.act_fn.
        # Other small causal LMs often still contain mlp/ffn/feed_forward names.
        is_candidate = (
            lname.endswith("mlp.act_fn")
            or lname.endswith(".act_fn")
            or "feed_forward" in lname
            or "ffn" in lname
        )
        if is_candidate:
            try:
                hooks.append(module.register_forward_hook(make_hook(name)))
            except Exception:
                pass

    return hooks


def summarize_mlp_activation_records(records):
    if not records:
        return {}
    return {
        "mlp_activation_active_fraction": safe_mean([r.get("active_fraction") for r in records]),
        "mlp_activation_abs_mean": safe_mean([r.get("abs_mean") for r in records]),
        "mlp_activation_abs_max": safe_max([r.get("abs_max") for r in records]),
        "mlp_activation_num_hook_records": int(len(records)),
    }

# -----------------------------
# 5. Main transport probe
# -----------------------------
# Operator pipeline:
#   1. Tokenize input text into a discrete path x_1:n.
#   2. Run one frozen forward pass with attentions and hidden states exposed.
#   3. Convert output logits into log probabilities and probabilities.
#   4. Pull out token path action: -log p(actual next token).
#   5. Pull out attention kernels: alpha_{t,s}^{layer,head}.
#   6. Pull out additive hidden trajectories: h_t^{layer}.
#   7. Conditionally compute selected advanced metrics.
#   8. Return summary rows, token rows, attention rows, and hidden rows.
#
# Modifier logic:
#   metric_enabled(name) checks the current checkbox context. The same function is used for
#   report probes and chat probes, so runtime selections control both without rewriting code.
#
# Pedagogical note:
#   This is the central microscope. It does not interpret text semantically. It measures the
#   model's internal transport response to that text.

def run_transport_probe(prompt, name="prompt", max_tokens=MAX_TOKENS, top_k=TOP_K):
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask", None)

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    mlp_activation_records = []
    mlp_hooks = []
    if metric_enabled("mlp_activation_sparsity"):
        mlp_hooks = install_mlp_activation_hooks(model, mlp_activation_records)

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for hook in mlp_hooks:
            try:
                hook.remove()
            except Exception:
                pass

    logits = outputs.logits.float()
    hidden_states = outputs.hidden_states
    attentions = outputs.attentions

    seq_len = input_ids.shape[1]

    if seq_len >= 2:
        pred_logits = logits[:, :-1, :]
        targets = input_ids[:, 1:]

        log_probs = F.log_softmax(pred_logits, dim=-1)
        probs = log_probs.exp().clamp_min(EPS)

        token_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        output_entropy = entropy_from_probs(probs, dim=-1)
        log_partition = torch.logsumexp(pred_logits, dim=-1)
    else:
        pred_logits = logits[:, :0, :]
        targets = input_ids[:, :0]
        log_probs = torch.empty((1, 0, logits.shape[-1]), device=device)
        probs = torch.empty((1, 0, logits.shape[-1]), device=device)
        token_nll = torch.empty((1, 0), device=device)
        output_entropy = torch.empty((1, 0), device=device)
        log_partition = torch.empty((1, 0), device=device)

    tokens = token_text_list(input_ids[0])

    token_action_z = None
    token_nll_cutoff = None
    output_entropy_cutoff = None

    if token_nll.numel() > 0:
        token_nll_cutoff = float(token_nll[0].mean().item())
        output_entropy_cutoff = float(output_entropy[0].mean().item())

    if metric_enabled("action_spike_index") and token_nll.numel() > 0:
        token_action_z = action_spike_zscores(token_nll[0])

    token_rows = []

    for t in range(targets.shape[1]):
        context_token_id = int(input_ids[0, t].item())
        target_token_id = int(targets[0, t].item())
        top_preds = topk_predictions(pred_logits[0, t], k=top_k)

        row = {
            "prompt_name": name,
            "position": int(t),
            "context_token_text": tokens[t],
            "context_token_id": context_token_id,
            "target_token_text": tokenizer.convert_ids_to_tokens(target_token_id),
            "target_decoded": decode_one_token(target_token_id),
            "target_token_id": target_token_id,
            "nll_path_action": float(token_nll[0, t].item()),
            "output_entropy": float(output_entropy[0, t].item()),
            "log_partition_lse": float(log_partition[0, t].item()),
        }

        if metric_enabled("action_spike_index") and token_action_z is not None:
            z_val = float(token_action_z[t].item())
            row["action_spike_z"] = z_val
            row["action_spike_flag"] = int(z_val >= ACTION_SPIKE_Z_THRESHOLD)

        if metric_enabled("target_rank"):
            row["target_rank"] = target_rank_from_logits(pred_logits[0, t], target_token_id)
            row["target_in_topk"] = int(any(pred["token_id"] == target_token_id for pred in top_preds))

        if metric_enabled("confidence_error_split") and token_nll_cutoff is not None and output_entropy_cutoff is not None:
            row["confidence_error_regime"] = confidence_error_regime(
                row["nll_path_action"],
                row["output_entropy"],
                token_nll_cutoff,
                output_entropy_cutoff,
            )

        if metric_enabled("effective_support"):
            row["effective_support_exp_entropy"] = float(math.exp(row["output_entropy"]))

        if metric_enabled("top_margin"):
            if pred_logits.shape[-1] >= 2:
                top2_probs, _ = torch.topk(F.softmax(pred_logits[0, t], dim=-1), k=2)
                top2_logits, _ = torch.topk(pred_logits[0, t], k=2)
                row["top1_minus_top2_prob_margin"] = float((top2_probs[0] - top2_probs[1]).item())
                row["top1_minus_top2_logit_gap"] = float((top2_logits[0] - top2_logits[1]).item())

        for i, pred in enumerate(top_preds, start=1):
            row[f"top{i}_text"] = pred["token_text"]
            row[f"top{i}_id"] = pred["token_id"]
            row[f"top{i}_prob"] = pred["prob"]

        token_rows.append(row)

    attention_rows = []
    head_js_disagreement_values = []
    head_consensus_values = []
    cross_head_source_mi_values = []
    token_attention_influence = None

    if attentions is not None:
        if (metric_enabled("token_attention_influence") or metric_enabled("causal_chokepoint_score")) and seq_len > 0:
            token_attention_influence = torch.zeros(seq_len, dtype=torch.float32)

        for layer_idx, attn in enumerate(attentions):
            a = attn[0].float().clamp_min(EPS)
            row_entropy = entropy_from_probs(a, dim=-1)
            row_sums = a.sum(dim=-1)

            if metric_enabled("head_disagreement"):
                head_js_disagreement_values.append(layer_head_js_disagreement(a))

            if metric_enabled("head_consensus"):
                head_consensus_values.append(layer_head_consensus(a))

            if metric_enabled("cross_head_source_mi"):
                cross_head_source_mi_values.append(cross_head_source_mi(a))

            if token_attention_influence is not None:
                # Sum attention received by each source token across query positions and heads.
                token_attention_influence += a.detach().cpu().sum(dim=(0, 1))

            for h in range(a.shape[0]):
                attn_head = a[h]

                attention_row = {
                    "prompt_name": name,
                    "layer": int(layer_idx),
                    "head": int(h),
                    "mean_attention_entropy": float(row_entropy[h].mean().item()),
                    "std_attention_entropy": float(row_entropy[h].std().item()),
                    "mean_row_sum_error": float((row_sums[h] - 1.0).abs().mean().item()),
                    "common_prefix_adjacent_kl_drift": attention_common_prefix_kl(attn_head),
                    "recent_attention_mass_width4": attention_recent_mass(attn_head, recent_width=4),
                    "self_attention_mass": attention_self_mass(attn_head),
                    "max_attention_value": float(attn_head.max().item()),
                    "min_attention_value": float(attn_head.min().item()),
                }

                if metric_enabled("attention_transport_distance"):
                    attention_row["attention_transport_distance_mean"] = attention_transport_distance(attn_head)

                if metric_enabled("attention_concentration_radius"):
                    attention_row["attention_concentration_radius_mean"] = attention_concentration_radius(attn_head)

                if metric_enabled("attention_spectral_gap"):
                    attention_row["attention_spectral_gap"] = attention_spectral_gap(attn_head)

                if metric_enabled("attention_stationarity_error"):
                    attention_row["attention_stationarity_error"] = attention_stationarity_error(attn_head)

                if metric_enabled("attention_reversibility_error"):
                    attention_row["attention_reversibility_error"] = attention_reversibility_error(attn_head)

                if metric_enabled("attention_entropy_variance"):
                    attention_row["attention_entropy_variance"] = float(row_entropy[h].var(unbiased=False).item())

                attention_rows.append(attention_row)

    if token_attention_influence is not None:
        influence_values = token_attention_influence.detach().cpu().tolist()
        for row in token_rows:
            pos = int(row["position"])
            row["token_attention_influence"] = float(influence_values[pos]) if pos < len(influence_values) else 0.0
            row["target_token_attention_influence"] = float(influence_values[pos + 1]) if pos + 1 < len(influence_values) else 0.0

    if metric_enabled("causal_chokepoint_score"):
        for row in token_rows:
            influence = row.get("token_attention_influence", 0.0)
            row["causal_chokepoint_score"] = float(influence) * float(row.get("nll_path_action", 0.0))

    hidden_rows = []

    if hidden_states is not None:
        prev_h = None
        prev_h_normed = None

        for layer_idx, hs in enumerate(hidden_states):
            h = hs[0].float()
            h_normed = rms_normalized_hidden(h)

            row = {
                "prompt_name": name,
                "hidden_block": int(layer_idx),
                "hidden_block_type": "embedding_output" if layer_idx == 0 else "layer_output",

                "hidden_l2_mean_raw": float(h.norm(dim=-1).mean().item()),
                "hidden_l2_std_raw": float(h.norm(dim=-1).std().item()),
                "hidden_abs_mean_raw": float(h.abs().mean().item()),
                "hidden_abs_max_raw": float(h.abs().max().item()),

                "hidden_l2_mean_rms_normed": float(h_normed.norm(dim=-1).mean().item()),
                "hidden_abs_mean_rms_normed": float(h_normed.abs().mean().item()),

                "adjacent_token_cosine_mean_raw": cosine_mean_adjacent(h),
                "adjacent_token_l2_step_mean_raw": l2_mean_adjacent(h),

                "adjacent_token_cosine_mean_rms_normed": cosine_mean_adjacent(h_normed),
                "adjacent_token_l2_step_mean_rms_normed": l2_mean_adjacent(h_normed),
            }

            if prev_h is not None and prev_h.shape == h.shape:
                diff = h - prev_h
                diff_normed = h_normed - prev_h_normed

                row["layer_to_layer_l2_delta_mean_raw"] = float(diff.norm(dim=-1).mean().item())
                row["layer_to_layer_abs_delta_mean_raw"] = float(diff.abs().mean().item())
                row["layer_to_layer_cosine_mean_raw"] = float(
                    F.cosine_similarity(prev_h, h, dim=-1).mean().item()
                )

                row["layer_to_layer_l2_delta_mean_rms_normed"] = float(
                    diff_normed.norm(dim=-1).mean().item()
                )
                row["layer_to_layer_abs_delta_mean_rms_normed"] = float(
                    diff_normed.abs().mean().item()
                )
                row["layer_to_layer_cosine_mean_rms_normed"] = float(
                    F.cosine_similarity(prev_h_normed, h_normed, dim=-1).mean().item()
                )
            else:
                row["layer_to_layer_l2_delta_mean_raw"] = 0.0
                row["layer_to_layer_abs_delta_mean_raw"] = 0.0
                row["layer_to_layer_cosine_mean_raw"] = 0.0
                row["layer_to_layer_l2_delta_mean_rms_normed"] = 0.0
                row["layer_to_layer_abs_delta_mean_rms_normed"] = 0.0
                row["layer_to_layer_cosine_mean_rms_normed"] = 0.0

            if metric_enabled("hidden_curvature"):
                row["hidden_curvature_mean_raw"] = hidden_curvature_mean(h)
                row["hidden_curvature_mean_rms_normed"] = hidden_curvature_mean(h_normed)

            if metric_enabled("layer_work_ratio"):
                row["layer_work_ratio_raw"] = row["layer_to_layer_l2_delta_mean_raw"] / (row["hidden_l2_mean_raw"] + EPS_METRIC)
                row["layer_work_ratio_rms_normed"] = row["layer_to_layer_l2_delta_mean_rms_normed"] / (row["hidden_l2_mean_rms_normed"] + EPS_METRIC)

            if metric_enabled("residual_overwrite_proxy"):
                row["residual_overwrite_proxy_raw"] = row["layer_to_layer_l2_delta_mean_raw"] / (row["hidden_l2_mean_raw"] + EPS_METRIC)
                row["residual_overwrite_proxy_rms_normed"] = row["layer_to_layer_l2_delta_mean_rms_normed"] / (row["hidden_l2_mean_rms_normed"] + EPS_METRIC)

            if metric_enabled("hidden_anisotropy"):
                row["hidden_anisotropy"] = hidden_anisotropy_value(h_normed)

            if metric_enabled("hidden_effective_rank"):
                row["hidden_effective_rank"] = hidden_effective_rank_value(h_normed)

            hidden_rows.append(row)
            prev_h = h
            prev_h_normed = h_normed

    if token_nll.numel() > 0:
        avg_path_action = float(token_nll.mean().item())
        total_path_action = float(token_nll.sum().item())
        avg_output_entropy = float(output_entropy.mean().item())
        avg_lse = float(log_partition.mean().item())
        max_token_nll = float(token_nll.max().item())
        min_token_nll = float(token_nll.min().item())
    else:
        avg_path_action = 0.0
        total_path_action = 0.0
        avg_output_entropy = 0.0
        avg_lse = 0.0
        max_token_nll = 0.0
        min_token_nll = 0.0

    summary = {
        "prompt_name": name,
        "prompt": prompt,
        "model_id": MODEL_ID,
        "device": device,
        "dtype": str(dtype),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "num_input_tokens": int(seq_len),
        "num_scored_tokens": int(max(seq_len - 1, 0)),
        "avg_path_action_nll": avg_path_action,
        "total_path_action_nll": total_path_action,
        "avg_output_entropy": avg_output_entropy,
        "avg_log_partition_lse": avg_lse,
        "max_token_nll": max_token_nll,
        "min_token_nll": min_token_nll,
        "num_hidden_state_blocks": len(hidden_states) if hidden_states is not None else 0,
        "num_attention_layers": len(attentions) if attentions is not None else 0,
        "attention_available": attentions is not None,
        "hidden_states_available": hidden_states is not None,
    }

    summary["avg_attention_entropy"] = mean_or_none([
        row.get("mean_attention_entropy") for row in attention_rows
    ])
    summary["avg_attention_kl_drift"] = mean_or_none([
        row.get("common_prefix_adjacent_kl_drift") for row in attention_rows
    ])
    summary["avg_recent_attention_mass_width4"] = mean_or_none([
        row.get("recent_attention_mass_width4") for row in attention_rows
    ])
    summary["avg_self_attention_mass"] = mean_or_none([
        row.get("self_attention_mass") for row in attention_rows
    ])

    summary["avg_hidden_l2_mean_raw"] = mean_or_none([
        row.get("hidden_l2_mean_raw") for row in hidden_rows
    ])
    summary["avg_hidden_adjacent_cosine_raw"] = mean_or_none([
        row.get("adjacent_token_cosine_mean_raw") for row in hidden_rows
    ])
    summary["avg_hidden_adjacent_l2_step_raw"] = mean_or_none([
        row.get("adjacent_token_l2_step_mean_raw") for row in hidden_rows
    ])
    summary["avg_hidden_l2_mean_rms_normed"] = mean_or_none([
        row.get("hidden_l2_mean_rms_normed") for row in hidden_rows
    ])
    summary["avg_hidden_adjacent_cosine_rms_normed"] = mean_or_none([
        row.get("adjacent_token_cosine_mean_rms_normed") for row in hidden_rows
    ])
    summary["avg_hidden_adjacent_l2_step_rms_normed"] = mean_or_none([
        row.get("adjacent_token_l2_step_mean_rms_normed") for row in hidden_rows
    ])

    if metric_enabled("action_spike_index"):
        summary["num_action_spikes"] = int(sum(row.get("action_spike_flag", 0) for row in token_rows))
        summary["max_action_spike_z"] = safe_max([row.get("action_spike_z") for row in token_rows])

    if metric_enabled("target_rank"):
        summary["avg_target_rank"] = safe_mean([row.get("target_rank") for row in token_rows])
        summary["max_target_rank"] = safe_max([row.get("target_rank") for row in token_rows])

    if metric_enabled("effective_support"):
        summary["avg_effective_support_exp_entropy"] = safe_mean([row.get("effective_support_exp_entropy") for row in token_rows])

    if metric_enabled("top_margin"):
        summary["avg_top1_minus_top2_prob_margin"] = safe_mean([row.get("top1_minus_top2_prob_margin") for row in token_rows])
        summary["avg_top1_minus_top2_logit_gap"] = safe_mean([row.get("top1_minus_top2_logit_gap") for row in token_rows])

    if metric_enabled("output_js_drift") and probs.numel() > 0:
        summary["avg_output_js_drift"] = output_js_drift_mean(probs)

    if metric_enabled("output_distribution_acceleration") and probs.numel() > 0:
        summary["avg_output_distribution_acceleration"] = output_js_acceleration_mean(probs)

    if metric_enabled("probability_curvature") and probs.numel() > 0:
        summary["avg_probability_curvature_trace"] = probability_curvature_trace_mean(probs)

    if metric_enabled("temperature_sensitivity") and pred_logits.numel() > 0:
        summary["avg_temperature_entropy_sensitivity"] = temperature_entropy_sensitivity(pred_logits)

    if metric_enabled("output_entropy_delta") and output_entropy.numel() > 0:
        summary.update(output_entropy_delta_stats(output_entropy))

    if metric_enabled("free_energy_variance") and log_partition.numel() > 0:
        summary["free_energy_variance"] = float(log_partition.var(unbiased=False).item())

    if metric_enabled("integrity_checks"):
        if probs.numel() > 0:
            summary["output_prob_row_sum_error"] = float((probs.sum(dim=-1) - 1.0).abs().mean().item())
        finite_values = [
            summary.get("avg_path_action_nll"),
            summary.get("avg_output_entropy"),
            summary.get("avg_log_partition_lse"),
        ]
        summary["finite_metric_check"] = int(all(math.isfinite(float(v)) for v in finite_values if v is not None))

    if metric_enabled("attention_transport_distance"):
        summary["avg_attention_transport_distance"] = safe_mean([row.get("attention_transport_distance_mean") for row in attention_rows])

    if metric_enabled("attention_concentration_radius"):
        summary["avg_attention_concentration_radius"] = safe_mean([row.get("attention_concentration_radius_mean") for row in attention_rows])

    if metric_enabled("attention_spectral_gap"):
        summary["avg_attention_spectral_gap"] = safe_mean([row.get("attention_spectral_gap") for row in attention_rows])

    if metric_enabled("attention_stationarity_error"):
        summary["avg_attention_stationarity_error"] = safe_mean([row.get("attention_stationarity_error") for row in attention_rows])

    if metric_enabled("attention_reversibility_error"):
        summary["avg_attention_reversibility_error"] = safe_mean([row.get("attention_reversibility_error") for row in attention_rows])

    if metric_enabled("attention_entropy_variance"):
        summary["avg_attention_entropy_variance"] = safe_mean([row.get("attention_entropy_variance") for row in attention_rows])

    if metric_enabled("head_disagreement"):
        summary["avg_head_js_disagreement"] = safe_mean(head_js_disagreement_values)
        summary["max_head_js_disagreement"] = safe_max(head_js_disagreement_values)

    if metric_enabled("head_consensus"):
        summary["avg_head_consensus"] = safe_mean(head_consensus_values)
        summary["max_head_consensus"] = safe_max(head_consensus_values)

    if metric_enabled("cross_head_source_mi"):
        summary["avg_cross_head_source_mi"] = safe_mean(cross_head_source_mi_values)
        summary["max_cross_head_source_mi"] = safe_max(cross_head_source_mi_values)

    if metric_enabled("hidden_curvature"):
        summary["avg_hidden_curvature_rms_normed"] = safe_mean([row.get("hidden_curvature_mean_rms_normed") for row in hidden_rows])

    if metric_enabled("hidden_anisotropy"):
        summary["avg_hidden_anisotropy"] = safe_mean([row.get("hidden_anisotropy") for row in hidden_rows])

    if metric_enabled("hidden_effective_rank"):
        summary["avg_hidden_effective_rank"] = safe_mean([row.get("hidden_effective_rank") for row in hidden_rows])

    if metric_enabled("layer_work_ratio"):
        summary["avg_layer_work_ratio_rms_normed"] = safe_mean([row.get("layer_work_ratio_rms_normed") for row in hidden_rows])

    if metric_enabled("residual_overwrite_proxy"):
        summary["avg_residual_overwrite_proxy_rms_normed"] = safe_mean([row.get("residual_overwrite_proxy_rms_normed") for row in hidden_rows])

    if metric_enabled("mlp_activation_sparsity"):
        summary.update(summarize_mlp_activation_records(mlp_activation_records))

    if metric_enabled("token_attention_influence") or metric_enabled("causal_chokepoint_score"):
        summary["max_token_attention_influence"] = safe_max([row.get("token_attention_influence") for row in token_rows])
        summary["avg_token_attention_influence"] = safe_mean([row.get("token_attention_influence") for row in token_rows])

    if metric_enabled("causal_chokepoint_score"):
        summary["max_causal_chokepoint_score"] = safe_max([row.get("causal_chokepoint_score") for row in token_rows])
        summary["avg_causal_chokepoint_score"] = safe_mean([row.get("causal_chokepoint_score") for row in token_rows])

    return summary, token_rows, attention_rows, hidden_rows

# -----------------------------
# 6. Runtime metric selection panel
# -----------------------------
# Operator:
#   Build two-column checkbox arrays for selecting advanced metrics after the model loads.
#   The report panel controls saved prompt probes. The chat panel controls live chat diagnostics.
#
# Modifier:
#   Report checkboxes choose which metrics are computed when Run transport metrics is pressed.
#   Chat checkboxes choose which metrics are computed for each chat input when the chat diagnostic
#   checkbox is enabled.
#
# Pedagogical note:
#   The checkbox panel is the control surface. ADVANCED_METRIC_FLAGS only defines initial defaults.
#   Runtime boxes are what matter after the notebook is loaded.

# Metric names are internal keys. Display names are human-facing labels.
# Disabled labels mark metrics that need extra experimental modes: paired prompts,
# saved attractor baselines, cache tracing, or training gradients.
METRIC_DISPLAY_NAMES = {
    "action_spike_index": "Action spike index",
    "target_rank": "Target rank",
    "confidence_error_split": "Confidence/error split",
    "attention_transport_distance": "Attention transport distance",
    "attention_concentration_radius": "Attention concentration radius",
    "head_disagreement": "Head disagreement",
    "head_consensus": "Head consensus",
    "layer_work_ratio": "Layer work ratio",
    "hidden_curvature": "Hidden curvature",
    "residual_overwrite_proxy": "Residual overwrite proxy",
    "mlp_activation_sparsity": "MLP activation sparsity",
    "prompt_retention_generation": "Prompt retention generation",
    "effective_support": "Effective support",
    "top_margin": "Top margin / logit gap",
    "output_js_drift": "Output JS drift",
    "token_attention_influence": "Token attention influence",
    "probability_curvature": "Probability curvature",
    "temperature_sensitivity": "Temperature sensitivity",
    "output_entropy_delta": "Output entropy delta",
    "attention_spectral_gap": "Attention spectral gap",
    "attention_stationarity_error": "Attention stationarity error",
    "attention_reversibility_error": "Attention reversibility error",
    "cross_head_source_mi": "Cross-head source MI",
    "causal_chokepoint_score": "Causal choke-point score",
    "attention_entropy_variance": "Attention entropy variance",
    "hidden_anisotropy": "Hidden anisotropy",
    "hidden_effective_rank": "Hidden effective rank",
    "output_distribution_acceleration": "Output distribution acceleration",
    "repetition_loop_index": "Repetition-loop index",
    "free_energy_variance": "Free-energy variance",
    "integrity_checks": "Integrity checks",
    "layer_contraction_expansion": "Layer contraction/expansion (paired mode needed)",
    "prompt_pair_separation": "Prompt-pair separation (paired mode needed)",
    "attractor_distance_blend": "Attractor distance/blend (reference needed)",
    "sliding_window_markov_cost": "Sliding-window Markov cost (baseline needed)",
    "cache_value_drift": "Cache/value drift (cache tracing needed)",
    "gradient_norm": "Gradient norm (training-only)",
    "gradient_alignment": "Gradient alignment (training-only)",
    "adam_update_diagnostics": "Adam update diagnostics (training-only)",
    "parameter_heatmap": "Parameter heat map (training-only)",
}

# These metrics are listed as roadmap items but disabled in this frozen single-pass script.
# They require data that is not present in one ordinary inference pass.
METRIC_PEDAGOGY = {
    "action_spike_index": {
        "operator": "standardize per-token path action a_t = -log p(x_{t+1}|x_{<=t})",
        "modifier": "adds z-score and threshold flag to token rows",
        "why": "finds local rupture points hidden by average loss",
    },
    "target_rank": {
        "operator": "rank the observed target token among all vocabulary logits",
        "modifier": "adds rank and top-k membership to token rows",
        "why": "separates near-miss prediction from complete target failure",
    },
    "confidence_error_split": {
        "operator": "cross-classify entropy H_t and path action a_t",
        "modifier": "adds a natural-language regime label per token",
        "why": "detects confident-wrong, diffuse-costly, expected, and broad-fit states",
    },
    "attention_transport_distance": {
        "operator": "compute mean attention-weighted distance sum_s alpha_ts |t-s|",
        "modifier": "adds distance to attention-head rows",
        "why": "measures how far information is transported through context",
    },
    "attention_concentration_radius": {
        "operator": "compute attention-weighted standard deviation around the row center",
        "modifier": "adds radius to attention-head rows",
        "why": "separates tight local routing from broad attention spread",
    },
    "head_disagreement": {
        "operator": "average pairwise Jensen-Shannon divergence between heads in a layer",
        "modifier": "adds layer-level head disagreement to summary",
        "why": "detects competing internal routing channels",
    },
    "head_consensus": {
        "operator": "average maximum source-position mass after averaging heads",
        "modifier": "adds layer-level consensus to summary",
        "why": "detects collective agreement over which token matters",
    },
    "layer_work_ratio": {
        "operator": "normalize layer-to-layer hidden displacement by hidden magnitude",
        "modifier": "adds work ratio to hidden rows",
        "why": "shows which hidden blocks actively transform representation",
    },
    "hidden_curvature": {
        "operator": "measure second-order bending of adjacent-token hidden trajectory",
        "modifier": "adds curvature to hidden rows",
        "why": "marks representational turns and regime changes",
    },
    "residual_overwrite_proxy": {
        "operator": "estimate layer displacement divided by current hidden scale",
        "modifier": "adds overwrite proxy to hidden rows",
        "why": "distinguishes small residual nudges from large representational rewrites",
    },
    "mlp_activation_sparsity": {
        "operator": "hook candidate MLP activations and count active fraction above threshold",
        "modifier": "adds MLP activation statistics to summary",
        "why": "shows whether feature-bank use is sparse or broad",
    },
    "prompt_retention_generation": {
        "operator": "replay prompt plus answer and measure generated-token attention to prompt positions",
        "modifier": "adds chat-only prompt retention line",
        "why": "checks whether the answer remains coupled to the user prompt",
    },
    "effective_support": {
        "operator": "compute exp(H_t) from output entropy",
        "modifier": "adds effective live-token count to token rows",
        "why": "turns entropy into an intuitive count of plausible continuations",
    },
    "top_margin": {
        "operator": "subtract top-2 probability/logit from top-1 probability/logit",
        "modifier": "adds margin and logit gap to token rows",
        "why": "measures winner stability in the output distribution",
    },
    "output_js_drift": {
        "operator": "compute Jensen-Shannon drift between adjacent output probability rows",
        "modifier": "adds output drift to summary",
        "why": "measures how fast the next-token law changes along the prompt",
    },
    "token_attention_influence": {
        "operator": "sum future attention received by each source token across layers and heads",
        "modifier": "adds influence scores to token rows",
        "why": "finds tokens reused heavily by later computation",
    },
    "probability_curvature": {
        "operator": "compute trace of simplex metric Diag(p)-pp^T as 1-sum p_i^2",
        "modifier": "adds curvature proxy to summary",
        "why": "measures sensitivity of probability rows to logit perturbation",
    },
    "temperature_sensitivity": {
        "operator": "finite-difference entropy under nearby softmax temperatures",
        "modifier": "adds entropy-temperature slope to summary",
        "why": "detects brittle distributions near decision boundaries",
    },
    "output_entropy_delta": {
        "operator": "differentiate output entropy across adjacent positions",
        "modifier": "adds mean and variance of entropy change",
        "why": "shows whether uncertainty is rising, falling, or unstable along the path",
    },
    "attention_spectral_gap": {
        "operator": "treat attention as a Markov kernel and estimate 1-|lambda_2|",
        "modifier": "adds spectral gap to attention rows",
        "why": "measures mixing versus persistent channel structure",
    },
    "attention_stationarity_error": {
        "operator": "estimate stationary distribution pi and compute ||pi P - pi||_1",
        "modifier": "adds stationarity error to attention rows",
        "why": "checks whether the frozen attention kernel behaves like a stable flow",
    },
    "attention_reversibility_error": {
        "operator": "compare forward and reverse stationary flows pi_i P_ij and pi_j P_ji",
        "modifier": "adds reversibility error to attention rows",
        "why": "measures directedness versus equilibrium-like attention transport",
    },
    "cross_head_source_mi": {
        "operator": "compute mutual information between head identity and source-position mass",
        "modifier": "adds head-source specialization to summary",
        "why": "detects whether heads divide labor over source positions",
    },
    "causal_chokepoint_score": {
        "operator": "multiply token influence by token path action",
        "modifier": "adds choke-point score to token rows",
        "why": "finds tokens that are both surprising and reused downstream",
    },
    "attention_entropy_variance": {
        "operator": "compute variance of attention entropy across token positions",
        "modifier": "adds volatility of focus to attention rows",
        "why": "detects heads alternating between focused and diffuse states",
    },
    "hidden_anisotropy": {
        "operator": "compute dominant covariance eigenvalue divided by total covariance energy",
        "modifier": "adds anisotropy to hidden rows",
        "why": "detects representational collapse into a dominant direction",
    },
    "hidden_effective_rank": {
        "operator": "compute entropy-rank from covariance eigenvalues",
        "modifier": "adds effective rank to hidden rows",
        "why": "estimates dimensionality actually used by hidden states",
    },
    "output_distribution_acceleration": {
        "operator": "differentiate adjacent output JS drift",
        "modifier": "adds distribution acceleration to summary",
        "why": "finds sudden changes in the next-token law",
    },
    "repetition_loop_index": {
        "operator": "count repeated token concentration in the generated response tail",
        "modifier": "adds chat-only repetition-loop line",
        "why": "detects early loop formation during generation",
    },
    "free_energy_variance": {
        "operator": "measure variance of log-sum-exp partition F_t across positions",
        "modifier": "adds partition volatility to summary",
        "why": "tracks instability in the normalization landscape",
    },
    "integrity_checks": {
        "operator": "check probability row sums and finite scalar metrics",
        "modifier": "adds numerical validity fields to summary",
        "why": "prevents interpretation of broken tensor/probability outputs",
    },
}

# These metrics are listed as roadmap items but disabled in this frozen single-pass script.
# They require data that is not present in one ordinary inference pass.
UNAVAILABLE_IN_FROZEN_SINGLE_PASS = {
    "layer_contraction_expansion",
    "prompt_pair_separation",
    "attractor_distance_blend",
    "sliding_window_markov_cost",
    "cache_value_drift",
    "gradient_norm",
    "gradient_alignment",
    "adam_update_diagnostics",
    "parameter_heatmap",
}

CHAT_RELEVANT_METRICS = set(ADVANCED_METRIC_FLAGS.keys())
CHAT_ONLY_METRICS = {"prompt_retention_generation", "repetition_loop_index"}
REPORT_RELEVANT_METRICS = [
    name for name in ADVANCED_METRIC_FLAGS.keys()
    if name not in CHAT_ONLY_METRICS
]


def selected_metrics_from_checkboxes(checkbox_map):
    return {name for name, box in checkbox_map.items() if box.value}


def make_metric_checkbox_grid(metric_names, initial_flags=None, columns=2):
    initial_flags = initial_flags or {}
    boxes = {}
    children = []

    for name in metric_names:
        label = METRIC_DISPLAY_NAMES.get(name, name)
        unavailable = name in UNAVAILABLE_IN_FROZEN_SINGLE_PASS
        box = widgets.Checkbox(
            value=False if unavailable else bool(initial_flags.get(name, False)),
            description=label,
            indent=False,
            disabled=unavailable,
            layout=widgets.Layout(width="360px"),
        )
        boxes[name] = box
        children.append(box)

    col_count = max(1, int(columns))
    col_len = math.ceil(len(children) / col_count)
    cols = []
    for i in range(col_count):
        col_children = children[i * col_len:(i + 1) * col_len]
        cols.append(widgets.VBox(col_children, layout=widgets.Layout(width="380px")))

    return boxes, widgets.HBox(cols)


report_metric_checkboxes, report_metric_grid = make_metric_checkbox_grid(
    REPORT_RELEVANT_METRICS,
    initial_flags=ADVANCED_METRIC_FLAGS,
    columns=2,
)

chat_metric_checkboxes, chat_metric_grid = make_metric_checkbox_grid(
    [name for name in ADVANCED_METRIC_FLAGS.keys() if name in CHAT_RELEVANT_METRICS],
    initial_flags=ADVANCED_METRIC_FLAGS,
    columns=2,
)

run_metrics_button = widgets.Button(
    description="Run transport metrics",
    button_style="success",
    tooltip="Run prompt probes using the selected advanced metric checkboxes",
)

select_core_report_button = widgets.Button(description="Core report on")
select_all_report_button = widgets.Button(description="All report on")
clear_report_button = widgets.Button(description="Clear report")
copy_report_to_chat_button = widgets.Button(description="Copy report to chat")
clear_chat_metrics_button = widgets.Button(description="Clear chat metrics")

metric_run_output = widgets.Output(
    layout=widgets.Layout(
        border="1px solid #999",
        height="380px",
        overflow_y="auto",
        padding="8px",
        width="100%",
    )
)

CORE_ADVANCED_METRICS = [
    "action_spike_index",
    "target_rank",
    "confidence_error_split",
    "attention_transport_distance",
    "attention_concentration_radius",
    "head_disagreement",
    "head_consensus",
    "layer_work_ratio",
    "hidden_curvature",
    "residual_overwrite_proxy",
    "mlp_activation_sparsity",
    "prompt_retention_generation",
]


def set_checkbox_values(checkbox_map, names, value):
    names = set(names)
    for name, box in checkbox_map.items():
        if name in names and not box.disabled:
            box.value = bool(value)


def set_all_checkbox_values(checkbox_map, value):
    for box in checkbox_map.values():
        if not box.disabled:
            box.value = bool(value)


def on_select_core_report_clicked(_):
    set_all_checkbox_values(report_metric_checkboxes, False)
    set_checkbox_values(report_metric_checkboxes, CORE_ADVANCED_METRICS, True)


def on_select_all_report_clicked(_):
    set_all_checkbox_values(report_metric_checkboxes, True)


def on_clear_report_clicked(_):
    set_all_checkbox_values(report_metric_checkboxes, False)


def on_copy_report_to_chat_clicked(_):
    selected = selected_metrics_from_checkboxes(report_metric_checkboxes)
    for name, box in chat_metric_checkboxes.items():
        box.value = name in selected


def on_clear_chat_metrics_clicked(_):
    set_all_checkbox_values(chat_metric_checkboxes, False)


select_core_report_button.on_click(on_select_core_report_clicked)
select_all_report_button.on_click(on_select_all_report_clicked)
clear_report_button.on_click(on_clear_report_clicked)
copy_report_to_chat_button.on_click(on_copy_report_to_chat_clicked)
clear_chat_metrics_button.on_click(on_clear_chat_metrics_clicked)


def run_all_prompt_probes(active_metrics):
    # Operator:
    #   Apply the selected metric context, run each calibration prompt, collect all row types,
    #   write durable artifacts, and print compact tables.
    #
    # Modifier:
    #   active_metrics is the set produced by the report checkbox panel.
    #
    # Pedagogical note:
    #   This is the batch microscope mode. It compares multiple prompt regimes under the same
    #   selected instrument configuration.
    set_metric_context(active_metrics)

    all_summaries = []
    all_token_rows = []
    all_attention_rows = []
    all_hidden_rows = []

    for prompt_name, prompt_text in PROMPTS.items():
        print("\nRunning probe:", prompt_name)

        summary, token_rows, attention_rows, hidden_rows = run_transport_probe(
            prompt=prompt_text,
            name=prompt_name,
            max_tokens=MAX_TOKENS,
            top_k=TOP_K,
        )

        all_summaries.append(summary)
        all_token_rows.extend(token_rows)
        all_attention_rows.extend(attention_rows)
        all_hidden_rows.extend(hidden_rows)

        print(
            "tokens:", summary["num_input_tokens"],
            "| avg path action:", round(summary["avg_path_action_nll"], 4),
            "| avg output entropy:", round(summary["avg_output_entropy"], 4),
            "| attention layers:", summary["num_attention_layers"],
        )

    comparison_cols = [
        "prompt_name",
        "num_input_tokens",
        "num_scored_tokens",
        "avg_path_action_nll",
        "total_path_action_nll",
        "avg_output_entropy",
        "avg_log_partition_lse",
        "avg_attention_entropy",
        "avg_attention_kl_drift",
        "avg_recent_attention_mass_width4",
        "avg_self_attention_mass",
        "avg_hidden_l2_mean_rms_normed",
        "avg_hidden_adjacent_cosine_rms_normed",
        "avg_hidden_adjacent_l2_step_rms_normed",
    ]

    for row in all_summaries:
        for key in row.keys():
            if key not in comparison_cols and key not in {"prompt", "model_id", "device", "dtype", "timestamp"}:
                comparison_cols.append(key)

    comparison_rows = [
        {col: row.get(col, "") for col in comparison_cols}
        for row in all_summaries
    ]

    token_surprise_rows = sort_rows(all_token_rows, "nll_path_action", reverse=True)
    volatile_heads_rows = sort_rows(all_attention_rows, "common_prefix_adjacent_kl_drift", reverse=True)
    stable_heads_rows = sort_rows(all_attention_rows, "common_prefix_adjacent_kl_drift", reverse=False)

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    summary_csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    comparison_path = os.path.join(OUTPUT_DIR, "prompt_comparison.csv")
    token_path = os.path.join(OUTPUT_DIR, "token_path_action.csv")
    attention_path = os.path.join(OUTPUT_DIR, "attention_transport.csv")
    hidden_path = os.path.join(OUTPUT_DIR, "hidden_additive_carrier.csv")
    token_surprise_path = os.path.join(OUTPUT_DIR, "token_surprise_ranking.csv")
    volatile_heads_path = os.path.join(OUTPUT_DIR, "volatile_attention_heads.csv")
    stable_heads_path = os.path.join(OUTPUT_DIR, "stable_attention_heads.csv")
    config_path = os.path.join(OUTPUT_DIR, "run_config.json")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    write_csv(summary_csv_path, all_summaries)
    write_csv(comparison_path, comparison_rows)
    write_csv(token_path, all_token_rows)
    write_csv(attention_path, all_attention_rows)
    write_csv(hidden_path, all_hidden_rows)
    write_csv(token_surprise_path, token_surprise_rows)
    write_csv(volatile_heads_path, volatile_heads_rows)
    write_csv(stable_heads_path, stable_heads_rows)

    run_config = {
        "model_id": MODEL_ID,
        "device": device,
        "dtype": str(dtype),
        "max_tokens": MAX_TOKENS,
        "top_k": TOP_K,
        "chat_max_new_tokens": CHAT_MAX_NEW_TOKENS,
        "chat_temperature": CHAT_TEMPERATURE,
        "chat_top_p": CHAT_TOP_P,
        "chat_repetition_penalty": CHAT_REPETITION_PENALTY,
        "prompts": PROMPTS,
        "advanced_metrics_enabled": sorted(active_metrics),
        "training": False,
        "pandas": False,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    zip_path = ZIP_NAME
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, OUTPUT_DIR)
                z.write(full_path, arcname=os.path.join(OUTPUT_DIR, arcname))

    print("\nSaved output directory:", OUTPUT_DIR)
    print("Saved ZIP:", zip_path)

    print_table(
        comparison_rows,
        columns=comparison_cols,
        limit=20,
        title="=== Prompt comparison ===",
    )

    surprise_cols = [
        "prompt_name",
        "position",
        "context_token_text",
        "target_token_text",
        "target_decoded",
        "nll_path_action",
        "output_entropy",
        "top1_text",
        "top1_prob",
    ]
    for col in [
        "action_spike_z",
        "action_spike_flag",
        "target_rank",
        "target_in_topk",
        "confidence_error_regime",
        "effective_support_exp_entropy",
        "top1_minus_top2_prob_margin",
        "top1_minus_top2_logit_gap",
        "token_attention_influence",
        "causal_chokepoint_score",
    ]:
        if any(col in row for row in token_surprise_rows):
            surprise_cols.append(col)

    print_table(
        token_surprise_rows,
        columns=surprise_cols,
        limit=25,
        title="=== Highest-surprise tokens ===",
    )

    attention_cols = [
        "prompt_name",
        "layer",
        "head",
        "mean_attention_entropy",
        "common_prefix_adjacent_kl_drift",
        "recent_attention_mass_width4",
        "self_attention_mass",
        "mean_row_sum_error",
    ]
    for col in [
        "attention_transport_distance_mean",
        "attention_concentration_radius_mean",
        "attention_spectral_gap",
        "attention_stationarity_error",
        "attention_reversibility_error",
        "attention_entropy_variance",
    ]:
        if any(col in row for row in volatile_heads_rows):
            attention_cols.append(col)

    print_table(
        volatile_heads_rows,
        columns=attention_cols,
        limit=25,
        title="=== Most volatile attention heads ===",
    )

    hidden_cols = [
        "prompt_name",
        "hidden_block",
        "hidden_block_type",
        "hidden_l2_mean_rms_normed",
        "hidden_abs_mean_rms_normed",
        "adjacent_token_cosine_mean_rms_normed",
        "adjacent_token_l2_step_mean_rms_normed",
        "layer_to_layer_cosine_mean_rms_normed",
        "layer_to_layer_l2_delta_mean_rms_normed",
    ]
    for col in [
        "hidden_curvature_mean_rms_normed",
        "layer_work_ratio_rms_normed",
        "residual_overwrite_proxy_rms_normed",
        "hidden_anisotropy",
        "hidden_effective_rank",
    ]:
        if any(col in row for row in all_hidden_rows):
            hidden_cols.append(col)

    print_table(
        all_hidden_rows,
        columns=hidden_cols,
        limit=40,
        title="=== Hidden-state additive carrier summary ===",
    )

    try:
        from google.colab import files
        files.download(zip_path)
    except Exception:
        print("Download skipped outside Colab. ZIP remains at:", zip_path)


def on_run_metrics_clicked(_):
    active_metrics = selected_metrics_from_checkboxes(report_metric_checkboxes)
    with metric_run_output:
        clear_output(wait=True)
        print("Selected report metrics:", sorted(active_metrics) if active_metrics else "base metrics only")
        run_all_prompt_probes(active_metrics)


run_metrics_button.on_click(on_run_metrics_clicked)

metric_panel = widgets.VBox([
    widgets.HTML("<b>Report metric checkboxes</b><br>Select advanced metrics, then press Run transport metrics."),
    report_metric_grid,
    widgets.HBox([select_core_report_button, select_all_report_button, clear_report_button, copy_report_to_chat_button]),
    run_metrics_button,
    metric_run_output,
])

print("\n=== Transport metric control panel ===")
display(metric_panel)

# ============================================================
# 11. Chat window using the same frozen instruct model
# ============================================================
# Operator:
#   Run ordinary sampled chat generation with the same frozen model. Separately, run the
#   transport probe on the user input and attach selected diagnostics to the answer.
#
# Modifier:
#   Chat metric checkboxes control only the diagnostics attached to chat turns. They do not
#   change generation parameters unless CHAT_TEMPERATURE, CHAT_TOP_P, or repetition penalty
#   are edited above.
#
# Pedagogical note:
#   Chat mode separates acting from measuring. The model generates a reply, while the probe
#   measures the input path and optional replay behavior.

chat_history = []


def build_chat_inputs(history, user_text):
    system_msg = (
        "Assistant answers directly and concisely. "
        "Assistant does not continue fake transcripts. "
        "Assistant does not invent previous users. "
        "Assistant can explain model diagnostics, transport metrics, code, and math."
    )

    messages = [{"role": "system", "content": system_msg}]

    for user_msg, assistant_msg in history[-4:]:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": user_text})

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = None
    else:
        text = None

    if text is None:
        text = system_msg + "\n\n"
        for user_msg, assistant_msg in history[-4:]:
            text += "User: " + user_msg.strip() + "\n"
            text += "Assistant: " + assistant_msg.strip() + "\n\n"
        text += "User: " + user_text.strip() + "\nAssistant:"

    return tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)


def clean_chat_response(response):
    response = response.strip()

    stop_markers = [
        "\nUser:",
        "\nSystem:",
        "\nAssistant:",
        "User:",
        "System:",
    ]

    for marker in stop_markers:
        if marker in response:
            response = response.split(marker)[0].strip()

    if response.lower().startswith("assistant:"):
        response = response[len("assistant:"):].strip()

    return response if response else "[empty response]"


def generate_chat_response(user_text):
    inputs = build_chat_inputs(chat_history, user_text)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=CHAT_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=CHAT_TEMPERATURE,
            top_p=CHAT_TOP_P,
            repetition_penalty=CHAT_REPETITION_PENALTY,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        generated[0, input_len:],
        skip_special_tokens=True,
    )

    return clean_chat_response(response)


def format_probe_summary(summary):
    parts = [
        f"path action={summary['avg_path_action_nll']:.4f}",
        f"output entropy={summary['avg_output_entropy']:.4f}",
    ]

    if summary.get("avg_attention_entropy") is not None:
        parts.append(f"attention entropy={summary['avg_attention_entropy']:.4f}")

    optional_fields = [
        ("num_action_spikes", "action spikes", "int"),
        ("max_action_spike_z", "max spike z", "float"),
        ("avg_target_rank", "avg target rank", "float"),
        ("avg_attention_transport_distance", "attn distance", "float"),
        ("avg_attention_concentration_radius", "attn radius", "float"),
        ("avg_head_js_disagreement", "head JS", "float"),
        ("avg_head_consensus", "head consensus", "float"),
        ("avg_layer_work_ratio_rms_normed", "layer work", "float"),
        ("avg_hidden_curvature_rms_normed", "hidden curvature", "float"),
        ("avg_residual_overwrite_proxy_rms_normed", "overwrite proxy", "float"),
        ("mlp_activation_active_fraction", "MLP active", "float"),
        ("avg_effective_support_exp_entropy", "effective support", "float"),
        ("avg_top1_minus_top2_prob_margin", "top margin", "float"),
        ("avg_output_js_drift", "output JS drift", "float"),
        ("avg_output_distribution_acceleration", "output accel", "float"),
        ("avg_probability_curvature_trace", "prob curvature", "float"),
        ("avg_temperature_entropy_sensitivity", "temp sensitivity", "float"),
        ("avg_output_entropy_delta", "entropy delta", "float"),
        ("output_entropy_delta_variance", "entropy delta var", "float"),
        ("avg_attention_spectral_gap", "attn spectral gap", "float"),
        ("avg_attention_stationarity_error", "stationarity err", "float"),
        ("avg_attention_reversibility_error", "reversibility err", "float"),
        ("avg_cross_head_source_mi", "head-source MI", "float"),
        ("avg_attention_entropy_variance", "attn entropy var", "float"),
        ("avg_hidden_anisotropy", "hidden anisotropy", "float"),
        ("avg_hidden_effective_rank", "hidden eff rank", "float"),
        ("free_energy_variance", "free-energy var", "float"),
        ("output_prob_row_sum_error", "prob row err", "float"),
        ("finite_metric_check", "finite check", "int"),
        ("max_causal_chokepoint_score", "max choke", "float"),
    ]

    for key, label, kind in optional_fields:
        value = summary.get(key)
        if value is None or value == "":
            continue
        if kind == "int":
            parts.append(f"{label}={int(value)}")
        else:
            parts.append(f"{label}={float(value):.4f}")

    return ", ".join(parts)


def chat_probe_for_last_user_message(user_text):
    try:
        summary, _, _, _ = run_transport_probe(
            prompt=user_text,
            name="chat_user_message",
            max_tokens=MAX_TOKENS,
            top_k=TOP_K,
        )
        return format_probe_summary(summary)
    except Exception as e:
        return f"diagnostic unavailable: {e}"


def prompt_retention_replay_probe(prompt_text, generated_text, max_tokens=MAX_TOKENS):
    if not metric_enabled("prompt_retention_generation"):
        return None

    try:
        prompt_ids = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        )["input_ids"]

        combined_text = prompt_text.strip() + "\n" + generated_text.strip()
        combined = tokenizer(
            combined_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_tokens,
        ).to(device)

        prompt_len = int(prompt_ids.shape[1])
        seq_len = int(combined["input_ids"].shape[1])

        if seq_len <= prompt_len or prompt_len <= 0:
            return None

        with torch.no_grad():
            outputs = model(
                **combined,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )

        if outputs.attentions is None:
            return None

        masses = []
        for attn in outputs.attentions:
            a = attn[0].float().clamp_min(EPS)
            response_rows = a[:, prompt_len:seq_len, :prompt_len]
            if response_rows.numel() > 0:
                masses.append(response_rows.sum(dim=-1).mean())

        if not masses:
            return None

        return float(torch.stack(masses).mean().item())

    except Exception:
        return None


chat_output = widgets.Output(
    layout=widgets.Layout(
        border="1px solid #999",
        height="420px",
        overflow_y="auto",
        padding="8px",
        width="100%",
    )
)

chat_input = widgets.Textarea(
    value="",
    placeholder="Type a message here. Press Send.",
    description="Input:",
    layout=widgets.Layout(width="100%", height="90px"),
)

send_button = widgets.Button(
    description="Send",
    button_style="primary",
    tooltip="Generate response",
)

clear_button = widgets.Button(
    description="Clear chat",
    button_style="",
    tooltip="Clear chat history",
)

diag_toggle = widgets.Checkbox(
    value=True,
    description="Show base + enabled advanced metrics for each user message",
)

chat_metric_controls = widgets.HBox([copy_report_to_chat_button, clear_chat_metrics_button])
controls = widgets.HBox([send_button, clear_button, diag_toggle])
chat_metric_panel = widgets.VBox([
    widgets.HTML("<b>Chat metric checkboxes</b><br>These advanced metrics apply only to chat input reports and replay checks."),
    chat_metric_grid,
    chat_metric_controls,
])
chat_box = widgets.VBox([chat_output, chat_input, controls, chat_metric_panel])


def render_chat():
    with chat_output:
        clear_output(wait=True)
        print("Frozen-weight chat window")
        print("Model:", MODEL_ID)
        print("Training: disabled")
        active_chat_metrics = selected_metrics_from_checkboxes(chat_metric_checkboxes)
        print("Advanced chat metrics:", sorted(active_chat_metrics) if active_chat_metrics else "none enabled")
        print("Metric source: chat-window checkboxes")
        print("-" * 60)

        if not chat_history:
            print("No messages yet.")
            return

        for idx, (user_msg, assistant_msg) in enumerate(chat_history, start=1):
            print(f"\nUser {idx}:")
            print(user_msg)
            print(f"\nAssistant {idx}:")
            print(assistant_msg)
            print("-" * 60)


def on_send_clicked(_):
    user_text = chat_input.value.strip()
    if not user_text:
        return

    chat_input.value = ""

    with chat_output:
        print("\nGenerating...")

    diag_line = ""
    if diag_toggle.value:
        active_chat_metrics = selected_metrics_from_checkboxes(chat_metric_checkboxes)
        set_metric_context(active_chat_metrics)
        diag_line = chat_probe_for_last_user_message(user_text)

    try:
        response = generate_chat_response(user_text)
    except Exception as e:
        response = f"[generation error: {e}]"

    retention_mass = None
    if diag_toggle.value:
        active_chat_metrics = selected_metrics_from_checkboxes(chat_metric_checkboxes)
        set_metric_context(active_chat_metrics)
        retention_mass = prompt_retention_replay_probe(user_text, response, max_tokens=MAX_TOKENS)

    repetition_loop_value = None
    if diag_toggle.value and metric_enabled("repetition_loop_index"):
        repetition_loop_value = repetition_loop_index_for_text(response)

    if diag_line:
        response = response + "\n\n[transport metrics for input: " + diag_line + "]"

    if retention_mass is not None:
        response = response + f"\n[prompt retention replay: prompt_attention_mass={retention_mass:.4f}]"

    if repetition_loop_value is not None:
        response = response + f"\n[repetition-loop index: {repetition_loop_value:.4f}]"

    chat_history.append((user_text, response))
    render_chat()


def on_clear_clicked(_):
    chat_history.clear()
    render_chat()


send_button.on_click(on_send_clicked)
clear_button.on_click(on_clear_clicked)

print("\n=== Chat window ===")
display(chat_box)
render_chat()

# ============================================================
# Appendix: why these metrics come from statistical mechanics and metric spaces
# ============================================================
# Natural-language map:
# A language model turns text into a path through states. At each token position it builds
# scores, normalizes those scores into probabilities, moves information through attention,
# and pays a cost for the actual next token. This is the same structural pattern used in
# statistical mechanics: raw energies or scores become Gibbs weights, Gibbs weights become
# probability distributions, and negative log-probability becomes an additive action or cost.
#
# The metric-space side enters because hidden states, attention rows, and probability rows
# all live in spaces where distance matters. Hidden states live in Euclidean-like vector spaces.
# Attention rows and output rows live on probability simplexes. Divergences, entropy, curvature,
# transport distance, cosine distance, L2 distance, spectral gap, and effective rank are all ways
# of measuring movement, spread, concentration, or collapse inside those spaces.
#
# Dense formal map:
# Tokens:
#     x_{1:n} in Sigma^n
#
# Additive embedding carrier:
#     X_t^(0) = E(x_t) + P(t),        X_t^(0) in R^d
#
# Layer operator:
#     X^(l+1) = X^(l) + Phi_l(X^(l))
#
# Query-key-value tensor operators:
#     q_t^(l,h) = W_Q^(l,h) X_t^(l)
#     k_s^(l,h) = W_K^(l,h) X_s^(l)
#     v_s^(l,h) = W_V^(l,h) X_s^(l)
#
# Additive attention score:
#     A_ts^(l,h) = <q_t^(l,h), k_s^(l,h)> / sqrt(d_h)
#
# Gibbs/free-energy normalization:
#     F_t^(l,h) = log sum_r exp(A_tr^(l,h))
#     alpha_ts^(l,h) = exp(A_ts^(l,h) - F_t^(l,h))
#     sum_s alpha_ts^(l,h) = 1
#
# Stochastic transport operator:
#     y_t^(l,h) = sum_s alpha_ts^(l,h) v_s^(l,h)
#
# Output logits and simplex row:
#     z_t = W_O X_t^(L)
#     p_t(i) = exp(z_ti) / sum_j exp(z_tj)
#
# Sequence probability and action:
#     P(x_{1:n}) = product_t p_t(x_{t+1})
#     A_path(x_{1:n}) = -log P(x_{1:n}) = sum_t -log p_t(x_{t+1})
#
# Entropy and uncertainty:
#     H(p_t) = -sum_i p_t(i) log p_t(i)
#     N_eff(t) = exp(H(p_t))
#
# Confidence/error logic:
#     low H and low action    -> confident expected token
#     low H and high action   -> confident wrong/misaligned token
#     high H and high action  -> diffuse and costly uncertainty
#     high H and low action   -> broad distribution but target still fit
#
# Metric-space objects:
#     Hidden states:        h_t in R^d, measured by L2, cosine, covariance rank.
#     Output rows:          p_t in Delta^{|V|-1}, measured by entropy, JS, KL, Fisher trace.
#     Attention rows:       alpha_t in Delta^{T-1}, measured by entropy, transport distance,
#                           spectral gap, stationarity, reversibility, and head divergence.
#
# Probability-simplex curvature:
#     G(p) = Diag(p) - p p^T
#     trace(G(p)) = 1 - sum_i p_i^2
# This is a local sensitivity proxy. It is large when mass can still move among alternatives,
# and small when the row is saturated or collapsed.
#
# Attention transport distance:
#     D_t = sum_s alpha_ts |t-s|
# This measures how far information is moved through context, not merely how focused attention is.
#
# Attention concentration radius:
#     c_t = sum_s alpha_ts s
#     R_t = sqrt(sum_s alpha_ts (s-c_t)^2)
# This measures spread around the attention center of mass.
#
# Attention Markov-kernel reading:
#     P = alpha_head
#     spectral_gap = 1 - |lambda_2(P)|
#     stationarity_error = ||pi P - pi||_1
#     reversibility_error = sum_ij |pi_i P_ij - pi_j P_ji|
# The head is interpreted as a row-stochastic operator. These metrics ask whether it mixes,
# stabilizes, or carries directed nonequilibrium flow.
#
# Hidden trajectory geometry:
#     v_t = h_{t+1} - h_t
#     curvature_t = ||v_{t+1} - v_t|| / (||v_t|| + eps)
# This detects turns in representation space, often corresponding to local regime shifts.
#
# Hidden covariance geometry:
#     C = covariance({h_t})
#     anisotropy = lambda_max(C) / trace(C)
#     effective_rank = exp(-sum_i w_i log w_i), where w_i=lambda_i/sum_j lambda_j
# These metrics ask whether the hidden trajectory uses many dimensions or collapses into a few.
#
# Head specialization:
#     I(H;S) = sum_{h,s} p(h,s) log[p(h,s)/(p(h)p(s))]
# where p(h,s) is derived from total attention mass head h assigns to source position s.
# High value means heads specialize by source position; low value means heads behave similarly.
#
# Choke-point logic:
#     influence_s = sum_{layers,heads,t} alpha_t,s^(layer,head)
#     choke_s = influence_s * action_s
# A token is structurally important when it is both costly/surprising and reused downstream.
#
# Statistical-mechanics interpretation:
#     logits/scores       -> negative-energy-like additive coordinates
#     exp(score)          -> Boltzmann/Gibbs weight
#     logsumexp(score)    -> free-energy/partition normalizer
#     softmax(score)      -> Gibbs distribution on the simplex
#     -log p(target)      -> local action / surprise cost
#     sum_t -log p        -> path action over a token trajectory
#
# Metric-space interpretation:
#     L2/cosine           -> movement in additive hidden vector space
#     KL/JS               -> movement between probability distributions
#     entropy             -> spread inside a simplex
#     spectral gap        -> mixing rate of a stochastic operator
#     curvature/rank      -> local geometry and dimensional use
#
# Logic summary:
#     If a quantity is a signed vector, treat it as additive geometry.
#     If a quantity is positive and normalized, treat it as a simplex distribution.
#     If a quantity is a row of attention, treat it as a stochastic transport operator.
#     If a quantity is a token probability along a sequence, log-lift it into path action.
#     If two distributions must be compared, use divergence or metric-space proxies.
#     If hidden states must be compared, use vector-space geometry and covariance structure.
#
# Practical reading:
#     The script is not trying to prove what the model "means."
#     It measures where the model concentrates, spreads, transports, bends, stabilizes,
#     destabilizes, agrees across heads, disagrees across heads, and pays prediction cost.
#     That is why the same forward pass can be read as a statistical-mechanics object,
#     a metric-space trajectory, and a transport operator system.
# ============================================================
