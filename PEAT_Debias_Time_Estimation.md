# PEAT-Debias — GPU Hour Estimation Reference

**Actual hardware:** NVIDIA A100 SXM4 80 GB (Akash Network, dseq 26628620)  
**Actual wall-clock total:** ~18 hours (pipeline launched ~May 5 evening UTC, finished May 6 01:08:51 UTC)  
**Pipeline cells completed:** 136 / 136 (0 errors, 0 failures)  
**Precision policy:** bfloat16 throughout — NO 4-bit or 8-bit quantisation on any model or baseline  

> **H100 note:** The user refers to "H100 GPU" but this experiment ran on an A100 SXM4 80 GB.
> The A100 SXM4 80 GB has ~312 TFLOPS (bf16) vs the H100 SXM5's ~989 TFLOPS (bf16).
> Scale factor: **H100 ≈ 3× faster** than A100 SXM4 for transformer training.
> All A100 timings below can be divided by ~3.0 for an H100 SXM5 estimate.

---

## 1. Model Size Reference

| Tag | HuggingFace ID | Type | Params | VRAM (bf16) | Attention |
|-----|---------------|------|--------|-------------|-----------|
| bert-base | google-bert/bert-base-uncased | Encoder | 110 M | ~0.4 GB | eager |
| modernbert-base | answerdotai/ModernBERT-base | Encoder | 149 M | ~0.6 GB | sdpa |
| nomicbert | nomic-ai/nomic-bert-2048 | Encoder | 137 M | ~0.5 GB | eager |
| qwen2.5-1.5b | Qwen/Qwen2.5-1.5B-Instruct | Causal | 1.5 B | ~3.0 GB | sdpa |
| gemma-3-4b | google/gemma-3-4b-it | Causal | 4.3 B | ~8.6 GB | sdpa |
| llama-3.1-8b | meta-llama/Meta-Llama-3.1-8B-Instruct | Causal | 8.0 B | ~16 GB | sdpa |

**LoRA adapter**: rank=4, alpha=8, dropout=0.0, last 2 transformer layers only, bf16.  
Trainable params per model: ~0.05–0.2% of total (≈6 MB adapter state per checkpoint).

---

## 2. Pipeline Stage Breakdown

Total: **136 cells** across 6 stages.

| Stage | Description | Cells | Approx. A100 Time |
|-------|-------------|-------|-------------------|
| 0 | Dry run + dataset preflight | 2 | ~5 min |
| 1 | PEAT training — 4 core models | 16 | ~8 h |
| 2 | PEAT scaling — 2 large models | 8 | ~4 h |
| 3 | Baselines — 9 methods × 4 models × 3 seeds | 108 | ~6 h |
| 4 | Aggregation (5 tables) | 1 | ~2 min |
| 5 | Figures (2 PDFs) | 1 | ~1 min |
| — | Model downloads (HuggingFace, first run only) | — | ~20–40 min |
| **Total** | | **136** | **~18–19 h** |

---

## 3. Stage 1 — PEAT Training (Core Models): Deep Breakdown

Each of the 4 core models (bert-base, modernbert-base, nomicbert, qwen2.5-1.5b) runs through
the full PEAT pipeline: Phase A (Successive Halving), Phase B (Bootstrap selection),
Phase C (Final training × 3 seeds), and evaluation (× 3 seeds).

### 3.1 Phase A — Successive Halving (SHA)

Grid: λ₁ × λ₂ ∈ {1e-3, 1e-2, 1e-1, 1.0, 10.0}² = **25 configurations**.

| SHA Round | Configs evaluated | Cumulative epochs per config | New epochs this round | Configs kept |
|-----------|------------------|-----------------------------|-----------------------|--------------|
| Round 0 | 25 | 1 | 1 | top 12 |
| Round 1 | 12 | 3 | 2 | top 4 |
| Round 2 | 4 | 5 | 2 | 4 (finalists) |

**Total epoch-passes across all SHA configs:**
- Round 0: 25 configs × 1 epoch = **25 epoch-passes**
- Round 1: 12 configs × 2 new epochs = **24 epoch-passes**
- Round 2: 4 configs × 2 new epochs = **8 epoch-passes**
- **Total SHA: 57 epoch-passes per model**

**Memory strategy:** The frozen reference model (θ₀) stays on GPU for all 41 configs.
Adapter states saved as CPU tensors (~6 MB each) — no full model reload between configs.

**Validation SS evaluation** after each config in each round: ~1 full StereoSet pass per config.

### 3.2 Phase B — Bootstrap Selection

- 4 finalist configs from SHA
- 1,000 bootstrap resamples on validation set per finalist (pure CPU numpy — negligible GPU time)
- 1 SS evaluation pass per finalist (4 forward passes total, ~2–5 min)

### 3.3 Phase C — Final Training

- 3 seeds (42, 123, 456) × 5 epochs each = **15 epoch-passes**
- Model resets to LoRA-init for each seed
- Frozen θ₀ kept on GPU for all 3 seeds

### 3.4 Evaluation

After final training: `evaluate_full()` per seed = **3 evaluation passes**.

Each evaluation pass includes:
- StereoSet SS: stratified 1,100-row sample, forward pass on all sentence pairs
- GLUE average (encoder models only): 8 task fine-tunes + eval
- WikiText-103 PPL (causal models only): perplexity pass over held-out text
- CrowS-Pairs LLM-as-judge (causal models only): 495 pairs × LLM judge API call

### 3.5 Per-Model Time Estimate (A100 SXM4 80 GB)

Training dataset: StereoSet ~8,000 training pairs (after 80/20 split of 29K total, or 880 train rows when stratified 1,100-row eval set is used). Batch sizes: 64 (encoders), 16 (decoders).

| Sub-phase | bert-base | modernbert-base | nomicbert | qwen2.5-1.5b |
|-----------|-----------|----------------|-----------|--------------|
| SHA (57 epoch-passes) | ~25 min | ~30 min | ~30 min | ~90 min |
| Bootstrap eval (4 configs) | ~2 min | ~2 min | ~2 min | ~8 min |
| Final training (15 epoch-passes) | ~7 min | ~8 min | ~8 min | ~25 min |
| Evaluation × 3 seeds | ~10 min | ~10 min | ~10 min | ~30 min |
| **Total per model** | **~45 min** | **~50 min** | **~50 min** | **~2.5 h** |

**Stage 1 total (4 models): ~5.5–6.5 h on A100**  
(Sequential; models loaded/unloaded one at a time)

---

## 4. Stage 2 — PEAT Scaling (Large Models): Deep Breakdown

Gemma-3-4b and Llama-3.1-8b skip SHA entirely. They reuse `c_best` transferred from
qwen2.5-1.5b (justified: no re-search needed, only scaling verification).

Each scaling model runs only **Phase C + Evaluation**:
- Final training: 3 seeds × 5 epochs = 15 epoch-passes
- Evaluation × 3 seeds (SS + CrowS-Pairs LLM-judge + WikiText-103 PPL)

| Sub-phase | gemma-3-4b | llama-3.1-8b |
|-----------|------------|-------------|
| Final training (15 epoch-passes, batch=16) | ~60 min | ~90 min |
| Evaluation × 3 seeds | ~45 min | ~60 min |
| **Total per model** | **~1.75 h** | **~2.5 h** |

**Stage 2 total (2 models): ~4–5 h on A100**

---

## 5. Stage 3 — Baselines: Deep Breakdown

9 baseline methods × 4 core models × 3 seeds = **108 cells**.
Base model is loaded **once per model** and shared across all 9 baselines.

### 5.1 Baseline Method Complexity

| Method | Approach | GPU Usage | Fine-tunes? | Approx. time per model per seed (A100) |
|--------|----------|-----------|-------------|---------------------------------------|
| `base` | No mitigation; inference only | Forward pass | No | ~3–5 min |
| `cda` | Counterfactual Data Augmentation; trains on augmented data | Full fine-tune | Yes | ~8–12 min |
| `self_debias` | Inference-time steering with self-generated debiased text | Forward pass | No | ~5–8 min |
| `auto_debias` | Bias direction projection; lightweight | Forward pass + projection | No | ~4–6 min |
| `bias_edit` | Model edit (locate + update bias neurons); encoder last layer | Forward pass | No | ~5–8 min |
| `fair_steer` | Steering vectors in activation space | Forward pass | No | ~4–6 min |
| `bias_unlearn` | Gradient-based unlearning over bias-related associations | Gradient pass | Yes (shallow) | ~8–12 min |
| `know_bias` | Knowledge editing via rank-one updates | Forward pass | No | ~4–6 min |
| `lora_vanilla_sft` | Standard LoRA SFT on counterfactual pairs (ablation) | LoRA fine-tune | Yes | ~8–15 min |

**Note on qwen2.5-1.5b:** Causal models take ~2–3× longer per baseline due to autoregressive
generation during LLM-as-judge evaluation of CrowS-Pairs.

### 5.2 Per-Model Baseline Times (all 9 methods × 3 seeds)

| Model | Approx. A100 time |
|-------|------------------|
| bert-base | ~45–60 min |
| modernbert-base | ~45–60 min |
| nomicbert | ~45–60 min |
| qwen2.5-1.5b | ~120–150 min |

**Stage 3 total: ~5–6 h on A100** (sequential; one model loaded at a time)

### 5.3 Known Anomaly

**BiasUnlearn on qwen2.5-1.5b** produced SS = 33.47 (deeply anti-stereotypical = model
collapse / over-correction). This is a known failure mode of gradient-based unlearning on
causal LMs at the standard hyperparameters used. It does not represent a genuine debiasing
result and should be excluded from aggregate comparisons.

---

## 6. Evaluation Cost Breakdown (per model per seed)

| Evaluation type | Models | Dataset size | Approx. A100 time |
|-----------------|--------|-------------|-------------------|
| StereoSet SS | All 6 | Stratified 1,100 rows (100/category × 11 categories) | ~2–5 min |
| GLUE average (8 tasks) | Encoders only (3 models) | Task-specific (2K–393K rows) | ~3–8 min |
| WikiText-103 PPL | Decoders only (3 models) | 245,569 tokens held-out | ~1–3 min |
| CrowS-Pairs LLM-judge | Decoders only (3 models) | Stratified 495 pairs (55/type × 9 types) | ~5–15 min (DeepSeek API latency) |

**CrowS-Pairs note:** LLM-as-judge is API-bound (DeepSeek as fallback; GCP Gemini 4× keys
all failed with ThinkingConfig attribute error throughout the run). Latency dominated by
network round-trips, not GPU compute.

**Llama CrowS-Pairs reliability note:** llama-3.1-8b had 65–74% abstention in LLM-judge
parsing (verbose output style). CrowS-Choice SS unreliable for llama; use StereoSet SS
(55.79) for paper reporting.

---

## 7. Dataset Download / Preflight Times (first run only)

These are one-time costs on a fresh VM. Cached on subsequent runs.

| Dataset / Asset | Size | Download time (VM, ~1 Gbps) |
|----------------|------|-----------------------------|
| StereoSet (HuggingFace) | ~8 MB | ~5 s |
| CrowS-Pairs CSV | ~1 MB | ~2 s |
| GLUE tasks (all 8 via HuggingFace) | ~500 MB | ~2–3 min |
| WikiText-103 (HuggingFace) | ~180 MB | ~30 s |
| bert-base-uncased weights | ~440 MB | ~30 s |
| ModernBERT-base weights | ~600 MB | ~45 s |
| nomic-bert-2048 weights | ~550 MB | ~40 s |
| qwen2.5-1.5b-instruct weights | ~3.1 GB | ~3 min |
| gemma-3-4b-it weights | ~8.6 GB | ~8 min |
| llama-3.1-8b-instruct weights | ~16 GB | ~15 min |
| Python packages (pip install) | ~2 GB | ~5–10 min |
| **Total first-run overhead** | **~32 GB** | **~35–45 min** |

Model weights are gated on HuggingFace — gemma-3-4b and llama-3.1-8b require an accepted
HuggingFace access token (`HF_Classic_Token` in `.env`).

---

## 8. Full Run Cost Summary

### A100 SXM4 80 GB (actual hardware)

| Component | Wall-clock hours | GPU-hours (1 GPU) |
|-----------|-----------------|-------------------|
| Stage 0: Preflight | 0.08 | 0.08 |
| Stage 1: PEAT core training (4 models) | ~6.0 | 6.0 |
| Stage 2: PEAT scaling (2 large models) | ~4.5 | 4.5 |
| Stage 3: Baselines (108 cells) | ~6.0 | 6.0 |
| Stage 4–5: Aggregation + figures | 0.05 | 0.05 |
| Downloads / setup (first run only) | ~0.6 | 0.6 |
| **Total (observed)** | **~18 h** | **~18 GPU-hours** |

### H100 SXM5 80 GB (estimated)

Scaling factor ~3× (312 → 989 TFLOPS bf16). Download / API-bound time does not scale.

| Component | H100 estimated hours |
|-----------|---------------------|
| Stage 1: PEAT core training | ~2.0 h |
| Stage 2: PEAT scaling | ~1.5 h |
| Stage 3: Baselines | ~2.0 h |
| Downloads / setup | ~0.5 h (network-bound, unchanged) |
| **Total (H100 estimate)** | **~6–7 h** |

### Cost Estimate (Akash Network)

| GPU | Approx. rate (Akash spot) | Total GPU-hours | Estimated cost |
|-----|--------------------------|-----------------|----------------|
| A100 SXM4 80 GB | ~$1.50–2.00 / hr | 18 h | **~$27–36** |
| H100 SXM5 80 GB | ~$3.50–4.50 / hr | 7 h | **~$25–32** |

> Note: Akash spot pricing fluctuates. The actual deployment (dseq 26628620) ran on
> provider `akash12v6dhc8awlwhv438jjyw80eguhgtm735mfv3fx` at the A100 rate.
> Deployment lasted approximately 5–6 days total (including setup days), but the
> core 136-cell pipeline ran in ~18 hours.

---

## 9. How to Re-Run (Future Estimation Checklist)

For anyone re-running or extending this pipeline, here is the time cost per new component:

| New experiment type | Estimated A100 hours | H100 hours |
|--------------------|--------------------|------------|
| Add 1 new encoder model (PEAT full: SHA + final + eval) | ~1 h | ~0.4 h |
| Add 1 new causal model (PEAT full: SHA + final + eval) | ~3–5 h | ~1–2 h |
| Add 1 new causal scaling model (Phase C + eval only) | ~2.5 h | ~0.9 h |
| Add 1 new baseline method (all 4 models × 3 seeds) | ~1 h | ~0.4 h |
| Add 1 new seed to existing models (PEAT only) | ~1 h (4 models) | ~0.4 h |
| Re-run single model + all baselines from scratch | ~3–4 h | ~1–1.5 h |
| Full pipeline from scratch (fresh VM, no cache) | ~18–20 h | ~7–8 h |

### Key bottlenecks in order of GPU consumption:

1. **SHA grid search** (57 epoch-passes per model) — largest single consumer for PEAT.
   Reducing to fewer lambda grid points cuts this proportionally.
2. **Causal model baselines** — each seed on qwen/gemma/llama takes 2–3× longer than encoders
   due to generation-based evaluation.
3. **Evaluation (GLUE)** — GLUE average over 8 tasks adds meaningful overhead for encoder models.
4. **LLM-as-judge (CrowS-Pairs)** — network latency dominated; not GPU-bound.

### Checkpoint resume:
All cells write state to `Code/state/run_state.json`. Pipeline resumes cleanly from
interruption — you only re-run incomplete cells, not the full 136.
