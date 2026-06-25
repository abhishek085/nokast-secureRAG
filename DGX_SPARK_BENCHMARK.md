# I ran a full LLM-security research pipeline on a single NVIDIA DGX Spark 🖥️⚡

I built **nokast-secureRAG** — a context-aware prompt-injection defense for RAG — end
to end on **one NVIDIA DGX Spark (GB10 Grace-Blackwell, 121 GB unified memory)** sitting
on my desk. No cloud, no cluster. Generating data with a **35B** model, auditing it with
a **120B** model, and fine-tuning a **0.5B** model — all on the same little box.

Here's how it actually performed.

## TL;DR

- 🧠 Ran a **120B-parameter model** (NVFP4) locally — peak **116 GB** of the 121 GB unified memory.
- ⏱️ **~3 hours** of active compute for the whole pipeline (generate → judge → train → eval).
- 🔌 **~0.11 kWh total** — about one pot of coffee. Averaged **~37 W**.
- 🌡️ Never hotter than **74 °C**; held **94–96% GPU utilization** for hours straight.
- 🔢 Processed **~4.3M tokens** of generation + auditing, produced **5,233** labeled examples and a fine-tuned model.

## The workload

A four-stage teacher → judge → student pipeline, run **one model at a time** (the 35B
and 120B can't co-reside in 121 GB):

| Stage | Model (NVFP4) | What it did |
|---|---|---|
| **Generate** | Qwen3.6-35B-A3B | created 5,233 synthetic (query, context) security examples |
| **Judge** | Nemotron-3-Super-120B-A12B | independently re-labeled all 5,233 to cross-check quality |
| **Train** | Qwen2.5-0.5B + LoRA | fine-tuned the defense model on the agreed data |
| **Evaluate** | 0.5B + baselines | scored detection / FPR / latency on a held-out test set |

## Measured performance

All numbers sampled live during the runs (GPU utilization, temperature, power, and
clocks from `nvidia-smi`; unified-memory usage from the OS; energy by integrating power
over time).

| Stage | Duration | GPU util (mean/max) | Temp (mean/max) | Power (mean) | Energy | Throughput |
|---|---|---|---|---|---|---|
| Generate (35B) | 56.5 min | 95.7% / 96% | 64.7 °C / 73 °C | 37.2 W | 35.1 Wh | **757 tok/s** out |
| Judge (120B) | 111.5 min | 95.8% / 96% | 65.7 °C / 74 °C | 36.5 W | 67.9 Wh | 265 tok/s out |
| Train (0.5B LoRA) | 14.6 min | 94.0% / 96% | 62.6 °C / 73 °C | 37.1 W | 9.0 Wh | 13.1 samples/s |
| Evaluate | 1.4 min | 65.5% / 95% | 53.0 °C / 59 °C | 29.9 W | 0.7 Wh | — |
| **Total** | **~3.1 hours** | sustained 94–96% | **≤ 74 °C** | **~37 W avg** | **~112.7 Wh** | **~4.3M tokens** |

## Why these numbers are wild

- **A 120B model on a desktop.** In NVFP4 the Nemotron-120B occupied **102 GB** of GPU
  working memory and ran a sustained **265 tok/s** of structured output — on a device you
  can carry in one hand.
- **Sips power.** It averaged **~37 W** and peaked at **49 W**. Idle draw was **~10.5 W**.
  The entire 3-hour pipeline used **~0.11 kWh** — less than running a 40 W bulb for the
  same time, and roughly one brew of a coffee maker.
- **Cool and quiet.** Three hours of 94–96% GPU utilization and it never crossed **74 °C**,
  holding a steady **~2.4 GHz** SM clock the whole way.
- **No babysitting, no cloud.** Schema-constrained generation ran 12–20 concurrent
  requests through vLLM with **zero engine restarts** during the 56-minute generation run.

## The result it produced

The pipeline yielded a **0.5B** prompt-injection detector that hits **0.994 detection
recall at 0.026 false-positive rate and 37 ms latency** — and correctly handles
"same-sentence, different-context" attacks that keyword filters can't. Model:
**https://huggingface.co/abhishek085/nokast-secureRAG-0.5B**

## Methodology / honesty notes

- Models are NVFP4 (4-bit) quantized; the GB10 has native FP4 support.
- Telemetry is sampled at 1 Hz; energy is a trapezoidal integral of measured power, so it
  counts GPU package power, not whole-wall draw.
- Throughput is at the concurrency/batch sizes I used, not a tuned max-throughput
  benchmark. Model download and load/warm-up time are excluded from the stage durations.
- "One model at a time" is a real constraint at this memory size, not a limitation of
  the workload.

---

*A complete generate-audit-train-evaluate LLM pipeline — including a 120B-parameter
auditor — on a single ~37 W desktop device, in about 3 hours, for ~0.11 kWh. The DGX
Spark is a genuinely capable local-AI research machine.*
