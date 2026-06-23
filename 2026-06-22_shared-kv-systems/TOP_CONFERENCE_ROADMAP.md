# Top-Conference Gap Analysis And Roadmap

This note records the stricter paper-positioning decision after comparing the current
guarded shared-prefix result with recent KV-cache compression and shared-cache work.

## Strict Current Verdict

The current method is not yet strong enough for a standalone top-conference method or
systems paper.

The most accurate current positioning is:

> an empirical systems study and guarded storage design for applying TurboQuant to a
> Shared-KV shared-prefix table.

It has a valid system problem and useful full-run evidence, but the technical core is
still vulnerable to the critique:

> existing prefix sharing + existing KV quantizer + conventional high-precision
> sink/recent boundary protection.

This is stronger than a trivial application because it identifies the shared-prefix
lifecycle, records whole-prefix failure, measures full quality/storage/runtime, and
separates persistent and transient memory. It is still not enough for a strong top-tier
claim without a new policy, kernel, or theory-backed module.

## Prior-Art Pressure

| Work | Pressure on current claim | Required response |
| --- | --- | --- |
| KVTC | Directly targets compact storage for reusable KV caches across shared-prefix prompts, with on/off-GPU storage and long-context/reasoning evaluation. | Do not claim first shared-prefix KV compression. Compare against KVTC-style compression/storage metrics or explain a sharper architectural difference. |
| SKVQ | Uses sliding-window KV quantization and preserves recent tokens in high precision. | Fixed fp16 tail guard is not novel by itself. |
| InnerQ | Maintains high-precision recent and sink-token windows while optimizing hardware-aware KV quantization. | Fixed first/last guard is a known pattern unless made shared-prefix-specific and adaptive. |
| SpectrumKV | Uses per-token mixed precision for KV-cache transfer, protecting attention sinks/high-importance tokens at fp16. | Need stronger policy than manually chosen fixed guards. |
| KVmix | Uses gradient-based layer importance for mixed-precision KV-cache allocation. | Need beat or complement layer-/token-sensitivity baselines. |
| TriAxialKV | Uses calibrated multi-axis token tags, mixed precision, memory management, and fused Triton decode kernels with end-to-end throughput. | A top-tier systems claim needs an end-to-end runtime path, not only storage accounting. |

Representative references:

- KVTC: https://arxiv.org/abs/2511.01815
- SKVQ: https://arxiv.org/abs/2405.06219
- InnerQ: https://arxiv.org/html/2602.23200v2
- SpectrumKV: https://arxiv.org/html/2606.08635v1
- KVmix: https://ojs.aaai.org/index.php/AAAI/article/view/40422
- TriAxialKV: https://arxiv.org/abs/2605.17170

## Baseline Matrix Needed Before Strong Claims

The current evidence compares fp16 Shared-KV, independent TurboQuant 2.5/3.5, and the
guarded TurboQuant shared-prefix policy. That is not enough.

Baseline rule: for any method with an official GitHub repository, publishable baseline
numbers must come from cloning and running the official project. Local reimplementations
are allowed only as sanity checks or adapters and must not be presented as official
baseline evidence. See [BASELINE_REPRODUCTION_PLAN.md](BASELINE_REPRODUCTION_PLAN.md).

Required baselines:

| Category | Baselines | Metrics to align |
| --- | --- | --- |
| Reusable KV storage | KVTC, CacheGen-style compression, offloaded prefix-cache storage | compression ratio, on-GPU bytes, off-GPU bytes, load bandwidth, decompression latency, accuracy |
| Low-bit KV quantization | TurboQuant, SKVQ, InnerQ, KIVI/KVQuant if available | score, bit budget, logical bytes, actual HBM, dequant overhead |
| Mixed precision allocation | KVmix, SpectrumKV, TriAxialKV, PM-KVQ/RateQuant if feasible | allocation policy, protected tokens/layers, memory-quality frontier |
| Shared-prefix systems | fp16 Shared-KV, vLLM/SGLang prefix cache where practical | TTFT, inter-token latency, throughput, cache hit latency, resident prefix count |
| Ablations | whole-prefix TQ, fixed guard, adaptive guard, local-only quantization, K-only/V-only | isolate method contribution from exact sharing and from TurboQuant itself |

Required measurement split:

```text
M_independent_fp16
M_shared_fp16
M_shared_guarded

incremental_gain =
  (M_shared_fp16 - M_shared_guarded) / M_shared_fp16
```

The report must also separate:

- quantized middle-prefix payload;
- fp16 boundary guards;
- TurboQuant metadata/codebooks/scales/rotation or correction state;
- request-local suffix/decode KV;
- temporary dequantization buffers;
- dense materialized prefix cache, if used;
- allocator fragmentation and peak HBM when measured.

## Method Gap

Current fixed policy:

```text
first 512 shared-prefix tokens: fp16
middle shared-prefix segment:  TurboQuant 3.5
last 2048 shared-prefix tokens: fp16
request-local KV:              fp16
```

This is a useful empirical point but not a strong new method. The main weakness is that
the guard is a fixed heuristic. To become a top-conference contribution, the method needs
one of the following modules.

### M1: Reuse-Aware Shared-Prefix Precision Allocation

Define each prefix block `b` with expected reuse count `R_b`, storage saving `S_b`,
encoding cost `C_enc(b)`, per-use decode cost `C_dec(b)`, and quality risk `DeltaQ_b`.

One possible objective:

```text
maximize_pi
  sum_b R_b * S_b(pi_b)
  - alpha * sum_b C_enc(b, pi_b)
  - beta  * sum_b R_b * C_dec(b, pi_b)
  - gamma * Risk(b, pi_b)

subject to
  DeltaQ(pi) <= epsilon
  PeakHBM(pi) <= B
```

where `pi_b` selects fp16, TurboQuant 4.0, TurboQuant 3.5, TurboQuant 2.5, offload, or
evict/recompute for each shared-prefix block.

This would make the shared-prefix lifecycle essential to the algorithm, not just a
placement choice.

### M2: Cross-Suffix Sensitivity Guard

The repeated-prefix setting differs from standard streaming KV quantization because the
same compressed prefix is reused under multiple suffixes. A useful policy should measure
not only average sensitivity, but also cross-suffix worst-case risk.

Possible block risk:

```text
Risk(b) =
  E_s [ L(y_s; q_b(K,V)) - L(y_s; K,V) ]
  + eta * Var_s [ L(y_s; q_b(K,V)) - L(y_s; K,V) ]
  + rho * max_s Delta_s(b)
```

where suffix `s` ranges over branches that share the prefix.

Then choose guards/adaptive precision by:

```text
pi_b =
  argmin_p Memory(b, p)
  subject to Risk(b, p) <= tau_b
```

This would turn the guard from "first/last tokens are fp16" into a shared-prefix-specific
protection rule based on cross-suffix sensitivity.

### M3: Non-Materializing Compressed-Prefix Attention

Current runtime is not closed because the quantized middle prefix is materialized for
stock attention. A stronger systems contribution would compute attention over:

```text
fp16 start guard + compressed middle prefix + fp16 end guard + fp16 local KV
```

without reconstructing a full dense prefix tensor. The kernel should merge online
softmax states from guarded and compressed regions and report:

- TTFT;
- inter-token latency;
- throughput;
- peak HBM;
- resident shared-prefix count;
- cache hit/load latency;
- end-to-end serving throughput under repeated-prefix arrivals.

## Experimental Gates For A Top-Tier Submission

Minimum gates before claiming a strong paper:

1. Baseline frontier includes KVTC or an equivalent reusable-KV storage baseline.
2. At least two mixed-precision/token-protection baselines are run from official code
   where available, on the same repeated-context workload or a clearly aligned slice.
3. Quality is reported as comparable, not improved, unless supported by significance
   tests over seeds/models.
4. Storage savings are reported incrementally over fp16 Shared-KV, not only over
   independent fp16.
5. Runtime includes end-to-end serving metrics, not only per-row latency.
6. The proposed method has a principle or objective that cannot be reduced to
   "fixed sink/recent guard plus existing quantizer."
7. Results include at least one additional model or context/reuse regime beyond the
   current single evidence point.

## Revised Paper Positioning

Current acceptable title style:

> An Empirical Study of Guarded Shared-Prefix KV Compression for Repeated Long-Context
> Serving

Target title after method upgrade:

> Reuse-Aware Mixed-Precision Shared-Prefix KV Storage for Long-Context LLM Serving

or, if the kernel is completed:

> Non-Materializing Attention over Compressed Shared-Prefix KV Caches

## Immediate Next Step

Do not spend more full runs on fixed guard lengths. The next useful work item is to
prototype `M2` on a small calibration split:

1. Generate multiple suffix branches per shared prefix.
2. Quantize candidate prefix blocks independently.
3. Estimate cross-suffix score/logit/attention-output sensitivity.
4. Use the sensitivity scores to select protected blocks under a memory budget.
5. Compare against fixed `512/2048` guard, whole-prefix TurboQuant, and local-only
   quantization on the same 40-row then 600-row gates.
