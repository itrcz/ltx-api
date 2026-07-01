# Closed-source → open-source GPU migration: unit economics

**Question:** what does it cost to keep routing closed-source LLM traffic through the
proxy (paying OpenAI / Anthropic / others), versus serving open-source analogs on our
own GPUs (5090 / A100 / H200) — per 100k users.

**Baseline data:** monthly token consumption by model class. Latest full month =
**May 2026**. Current load ≈ **40k users**; everything is scaled ×2.5 to **per-100k-users**.

> Each data row = `month, category, input_tokens, output_tokens, total_tokens`.
> Workload is **input-heavy (~9:1 in:out)** across every category — this is the single
> biggest driver of the result: prefill is cheap on a GPU, so self-hosting wins big.

---

## 1. Closed → open-source analog map (GPU placement)

| Proxy category | Open-source analog | Tier | GPU | $/GPU-hr |
|---|---|---|---|---|
| **OpenAI** (GPT-4o/5 class) | Llama-3.3-70B / Qwen2.5-72B / DeepSeek-V3 | Large | 1×**H200** | 3.50 |
| **Anthropic** (Claude Sonnet/Opus) | Qwen2.5-72B / DeepSeek-V3 (R1 for Opus-class) | Large | 1×**H200** | 3.50 |
| **Other Closed Source** (Gemini Flash/Pro etc.) | Qwen2.5-32B / Gemma-3-27B | Medium | 1×**A100-80** | 1.50 |
| *(mini/cheap tier, optional split-off)* | Llama-3.1-8B / Qwen2.5-7B | Small | 1×**RTX 5090** | 0.70 |

`Grom` and the existing `Open Source` tiers already run on our own GPUs — they are **not**
part of the migration; only the three closed-source categories are.

**Throughput assumptions** (aggregate, vLLM/SGLang, continuous batching, per replica):

| Tier | decode tok/s | prefill tok/s |
|---|---|---|
| Large (70B, H200) | 1,500 | 20,000 |
| Medium (32B, A100-80) | 1,800 | 25,000 |
| Small (8B, 5090) | 3,000 | 40,000 |

**Proxy list prices** (blended per category, early-2026 public rates, $/1M in / out):
OpenAI `2.00 / 8.00` · Anthropic `3.50 / 15.00` · Other Closed Source `1.25 / 5.00`.

---

## 2. Proxy cost vs GPU cost — May 2026, 40k users

| Category | Total tokens | Proxy $/mo | $/1M (proxy) | GPU-hr/mo | GPU $/mo (compute) | $/1M (GPU) |
|---|--:|--:|--:|--:|--:|--:|
| OpenAI | 4.90B | **$12,141** | 2.48 | 135 | $473 | 0.097 |
| Anthropic | 4.26B | **$19,780** | 4.65 | 132 | $461 | 0.108 |
| Other Closed Source | 3.44B | **$6,050** | 1.76 | 105 | $158 | 0.046 |
| **Total** | **12.6B** | **$37,970** | — | **372** | **$1,091** | — |

- **Pure-compute basis** (perfectly autoscaled GPUs): **$1,091/mo** vs **$37,970/mo** → **97% cheaper**.
- **Realistic always-on floor** (1 replica per tier kept warm 24/7 = 2×H200 + 1×A100):
  **$6,205/mo**. Even this floor is **~84% cheaper** than proxy.
- Why the gap between compute and floor? Utilization is only **14–19%** at 40k users —
  the volume per category is small relative to a single GPU's capacity. The GPU cost here
  is dominated by *idle*, not by tokens.

---

## 3. Unit economics per 100k users (×2.5)

| Metric | Proxy | GPU (compute) | GPU (3 always-on replicas) |
|---|--:|--:|--:|
| Spend / month | **$94,925** | $2,728 | **$6,205** |
| Spend / year | **$1.14M** | $33k | $74k |
| Cost / user / month | **$0.949** | $0.027 | $0.062 |
| GPU-hours / month | — | 930 | 2,190 |

**Headline:** moving the three closed-source categories to open-source GPU serving cuts
the per-100k-user bill from **~$95k/mo to ~$6k/mo** — roughly **$89k/mo (~$1.07M/yr) saved**.
At 100k users the 3 always-on replicas run at ~47% utilization, so the floor and the
compute cost converge — no extra GPUs needed to reach 100k.

---

## 4. Capacity — "how many users per model / GPU"

At 50% average utilization (headroom for peaks), one replica serves:

| Category → analog | Eff. throughput | Tokens/user/mo | **Users / replica** | Replicas for 100k |
|---|--:|--:|--:|--:|
| OpenAI → 70B (H200) | ~18,800 tok/s | 122k | **~201,000** | 0.50 |
| Anthropic → 70B (H200) | ~15,100 tok/s | 106k | **~186,000** | 0.54 |
| Other CS → 32B (A100) | ~13,300 tok/s | 86k | **~203,000** | 0.49 |

So **one GPU per tier covers 100k users with ~50% headroom**. "Users to add per model" to
fill each replica to 100% (from the 40k baseline): roughly **+360k OpenAI**, **+330k
Anthropic**, **+365k Other-CS** users before a second replica is needed. The current 40k
load only uses ~15–19% of one GPU per tier — there is large spare capacity to absorb growth
before any GPU spend increases.

**Minimum fleet to migrate all closed-source traffic (any user count up to ~150k):**
2× H200 + 1× A100-80 (+ optional 1× 5090 if mini-tier is split off).

---

## 5. Caveats (the "+-" in "how much approx. on GPU")

- **Throughput is the swing factor.** If real aggregate decode is half my estimate, GPU
  cost roughly doubles — still <10% of proxy. The conclusion is robust to ±2× error.
- **Quality gap not modeled.** 70B open models ≈ GPT-4o-class, but Opus/GPT-5-frontier
  tasks may regress; keep a proxy fallback for the hardest traffic.
- **Floor cost dominates at low volume.** Below ~40k users, always-on GPUs are mostly idle
  — use serverless/autoscale (like this project's RunPod setup) to track the compute basis.
- **Excludes** engineering/ops, model storage, eval/QA, and the proxy's customer-facing
  margin (proxy price ≠ our cost — these are list prices we'd pay providers).
- Prices/throughput are early-2026 public estimates; plug in real blended rates to tighten.

---

## 6. H200 vs Huawei Ascend — and running DeepSeek V4

### 6.1 Chip-for-chip

| Spec | **NVIDIA H200** | **Huawei Ascend 910C** | **Ascend 950PR** (2026) |
|---|---|---|---|
| Process | TSMC 4N | SMIC 7nm (N+2) | next-gen |
| Compute | ~990 TFLOPS FP16 / ~2 PFLOPS FP8 | ~800 TFLOPS FP16 / ~1,054 TFLOPS INT8 | ~1.56 PFLOPS (claimed) |
| Memory | 141 GB HBM3e | ~64–128 GB HBM | larger HBM |
| Mem bandwidth | **4.8 TB/s** | 3.2 TB/s | — |
| Aggregate "TPP" | 15,840 | 12,032 (~76% of H200) | — |
| Single-chip inference | baseline | **~60% of H100** (so ~45–50% of H200) | targeting H200-class |
| Purchase | ~$25–30k | **~¥190k ≈ $26k** (China-only) | — |
| Cloud rent | **$2–4/GPU-hr** (median ~$3.95; Vast spot ~$1) | not rentable outside China | — |

**Takeaway per chip:** H200 wins clearly on raw single-card inference (~2× the 910C) and on
memory bandwidth, which is what decode throughput tracks. The 910C's pitch is **not** the
single card — it's **rack-scale + price + sovereignty**.

### 6.2 Rack-scale: CloudMatrix384

Huawei's answer to per-chip weakness is brute scale-out. **CloudMatrix384** = 384× 910C on an
optical Unified Bus, ~300 PFLOPS BF16 (beats GB200 NVL72's ~180). Measured DeepSeek serving
(arXiv 2506.12708):

- **Prefill: 6,688 tok/s per NPU** (4K prompt)
- **Decode: 1,943 tok/s per NPU** (<50 ms TPOT) — *exceeds published SGLang-on-H100 / DeepSeek-on-H800 compute efficiency*
- DeepSeek's own reported serving cost on Ascend: **~¥1/1M input (V3), ~¥4/1M (R1)** ≈ $0.14–0.55/1M, vs ~$7/1M for R1 on NVIDIA-class API.

The per-NPU decode (1,943 tok/s) is actually **higher than my single-H200 estimate (1,500)**
for these MoE workloads — because CloudMatrix throws an entire 384-chip fabric at expert
parallelism. So at fleet scale, "910C is half an H200" inverts: per-token cost on a full
CloudMatrix is reportedly **~90% below H100**.

### 6.3 Can you run DeepSeek V4?

Yes — V4 is open (Apache 2.0) and DeepSeek/Huawei support both NVIDIA and Ascend. Footprint
(FP8, weights only; KV cache on top — V4's CSA/HCA attention cuts KV to ~10% of V3.2):

| Model | Total / active | FP8 weights | **On H200** | **On Ascend 910C** |
|---|---|--:|---|---|
| **V4-Flash** | 284B / 13B | ~284 GB | **4× H200** (1 node, ~$14/hr) | 8× 910C |
| **V4-Pro** | 1.6T / 49B | ~1.6 TB | **16× H200** (2 nodes, ~$56/hr) | CloudMatrix partition (Huawei post-trained V4-Pro on 910C) |

- **V4-Flash is the practical sweet spot** for the migration: frontier quality, fits a single
  4×H200 node, MoE-13B-active so decode is cheap (~3,000–5,000 tok/s aggregate node).
- **V4-Pro** needs multi-node — only worth self-hosting at very high volume or for sovereignty.
- Reality check: **DeepSeek V4 API list is ~$0.30/1M (Flash) to ~$0.87/1M output (Pro)** —
  so cheap that self-hosting V4 only pays off above ~very high sustained load, or when you
  *can't* use the API (data residency, China-export, air-gap).

### 6.4 H200 vs Ascend — decision

| Factor | H200 | Ascend 910C |
|---|---|---|
| Per-chip perf | ✅ ~2× | ❌ |
| $/token at rack scale | good | ✅ ~90% below H100 (CloudMatrix) |
| Capex/chip | ~$26k | ✅ similar list, but China-subsidized |
| **Availability (EU/US)** | ✅ Vast/RunPod, $2–4/hr | ❌ effectively China-only; not rentable in EU |
| Software (vLLM/SGLang/CUDA) | ✅ mature | ⚠️ CANN/MindIE — DeepSeek-optimized but smaller ecosystem |
| DeepSeek V4 support | ✅ | ✅ (Huawei post-trained V4-Pro on 910C) |
| Sovereignty / sanctions-proof | ❌ | ✅ |

**Recommendation for this project:** stay on **H200 (rent at $2–4/hr on Vast/RunPod)** for
the migration — Ascend isn't practically rentable outside China and the software path is
heavier. Serve **DeepSeek-V4-Flash on a 4×H200 node** as the frontier open analog (replaces
the heaviest OpenAI/Anthropic traffic); keep 70B/32B on single H200/A100 for the bulk.
Revisit Ascend/CloudMatrix only if (a) you need a China region, (b) sovereignty becomes a
requirement, or (c) you reach CloudMatrix-scale volume where the ~90% per-token saving beats
the ecosystem friction.

**Sources:** [Tom's Hardware – 910C 60% of H100](https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseek-research-suggests-huaweis-ascend-910c-delivers-60-percent-nvidia-h100-inference-performance) · [Tom's Hardware – Huawei post-trained DeepSeek-V4-Pro on 910C](https://www.tomshardware.com/tech-industry/artificial-intelligence/huawei-led-team-claims-it-post-trained-deepseeks-1-6-trillion-parameter-models-on-ascend-910c-chips) · [arXiv 2506.12708 – CloudMatrix384 serving](https://arxiv.org/abs/2506.12708) · [getdeploying – H200 cloud pricing](https://getdeploying.com/gpus/nvidia-h200) · [Thundercompute – H200 price June 2026](https://www.thundercompute.com/blog/nvidia-h200-pricing) · [DeepSeek V4 specs (morphllm)](https://www.morphllm.com/deepseek-v4) · [DeepSeek-V4-Pro on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) · [TechPowerUp – 910C vs H200](https://www.techpowerup.com/343932/huawei-ascend-910c-accelerators-maturation-allegedly-spurred-nvidia-h200-export-reversal) · [gpuvec – Huawei+DeepSeek cost](https://gpuvec.com/posts/huawei_and_deepseek)

---

*Source month: May 2026 (latest full). Baseline 40k users → scaled ×2.5 to per-100k.
GPU rates: 5090 $0.70/hr · A100-80 $1.50/hr · H200 $3.50/hr. Hardware section: prices/specs
are early/mid-2026 public estimates — validate before procurement.*
