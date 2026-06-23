# Related Work And Novelty Boundary

This note records the current novelty audit for the Shared-KV systems direction.

## Main Conclusion

Prefix sharing, prefix-tree KV reuse, and shared-prefix decode kernels are already
well-covered. A broad claim such as "shared KV cache for repeated prompts" is not novel.

The defensible contribution is narrower:

> a guarded TurboQuant-compressed shared-prefix table, evaluated on an
> answer-preserving repeated long-context workload, with explicit accounting for the
> shared prefix table, request-local KV, metadata, and quality drift against an fp16
> Shared-KV reference.

The full guarded-prefix validation is now complete, so this is a storage-quality
result. It is still not a complete speedup claim because the current prototype either
repeatedly materializes quantized prefix tensors or uses an extra transient dense
materialization cache.

## Prefix Sharing And Shared-Prefix Kernels

| Work | What it already covers | Implication |
| --- | --- | --- |
| vLLM / PagedAttention | Paged KV management and block-level prefix-cache reuse across requests. | We cannot claim KV block reuse or automatic prefix caching as new. Our accounting must include allocator/block metadata eventually. |
| SGLang / RadixAttention | Runtime KV reuse through radix-prefix organization for structured LLM programs. | Prefix-tree cache reuse is a known serving primitive. A fixed repeat-group table is a simpler workload-specific representation. |
| Hydragen | Exact shared-prefix attention separates shared-prefix and unique-suffix attention. | Non-materializing attention must preserve exact softmax semantics. |
| ChunkAttention | Prefix-aware KV cache with chunk/prefix-tree organization and two-phase attention. | Dynamic prefix-aware cache management is prior art; our novelty cannot be the prefix table alone. |
| FlashInfer Cascade | Shared-prefix batch decode with partial attention state merging. | The right kernel target is online-softmax state merge, not Python `cat(prefix, local)`. |
| DeFT | Tree-structured inference avoids redundant shared-prefix KV I/O. | Tree/group scheduling is related prior art for broader shared-prefix workloads. |
| CoDec | Prefix-shared decoding needs dedicated kernels and workload balancing. | A reportable speedup claim needs a real shared-prefix decode path. The current materialized prototype is storage-first. |
| PAT | Packs queries by shared prefix and merges prefix/suffix attention states. | Our split-attention parity failure is a known hard systems boundary, not a minor implementation detail. |
| SparseX | Segment-level KV-cache sharing for interleaved, non-prefix serving patterns on top of vLLM-style infrastructure. | Broad KV reuse beyond exact prefixes is active recent work; our contribution should stay on exact repeated-prefix storage compression and evidence, not general cache reuse. |

Representative sources:

- vLLM/PagedAttention: https://arxiv.org/abs/2309.06180
- vLLM prefix caching: https://docs.vllm.ai/en/latest/design/prefix_caching/
- SGLang/RadixAttention: https://arxiv.org/abs/2312.07104
- Hydragen: https://arxiv.org/abs/2402.05099
- ChunkAttention: https://arxiv.org/abs/2402.15220
- FlashInfer Cascade: https://flashinfer.ai/2024/02/02/cascade-inference.html
- DeFT: https://arxiv.org/abs/2404.00242
- CoDec: https://arxiv.org/abs/2505.17694
- PAT: https://arxiv.org/abs/2511.22333
- SparseX: https://arxiv.org/abs/2606.01751

## KV Storage, Transfer, And Cache Systems

| Work | What it already covers | Implication |
| --- | --- | --- |
| CacheBlend / EPIC | Non-prefix cache reuse for RAG-like contexts with repair/recomputation. | This is a different correctness model. Do not mix it with exact-prefix claims. |
| MemServe | Distributed KV cache pool and global prompt-tree locality. | Long-term storage claims should include placement and movement, not only logical bytes. |
| Mooncake | KVCache-centric disaggregated serving and tiered memory. | TurboQuant-compressed shared prefixes are best framed as a memory-layer primitive after exact sharing is validated. |
| Marconi | Prefix-cache admission/eviction for hybrid models. | Cache value should account for reuse, memory footprint, and compute saved. |
| CacheGen-style compression | Compresses KV cache for transfer/storage. | Compression plus caching is prior art broadly; our claim needs the guarded shared-prefix policy and LongBench-style quality evidence. |
| Probabilistic Language Tries | Recent sequential KV-compression proposal combines probabilistic prefix deduplication with predictive delta coding. | It increases pressure on broad "shared-prefix compression" claims. Our current result must stay empirical and narrow: exact repeated prefixes, guarded TurboQuant middle-prefix storage, and measured quality/storage/runtime accounting. |

Representative sources:

- CacheBlend: https://arxiv.org/abs/2405.16444
- EPIC: https://arxiv.org/abs/2410.15332
- MemServe: https://arxiv.org/abs/2406.17565
- Mooncake: https://arxiv.org/abs/2407.00079
- Marconi: https://arxiv.org/abs/2411.19379
- CacheGen: https://arxiv.org/abs/2310.07240
- Probabilistic Language Tries: https://arxiv.org/abs/2604.15356

## Codec Work Is Also Crowded

| Work | What it already covers | Implication |
| --- | --- | --- |
| TurboQuant | Online random-access vector quantization with KV-cache evaluations. | A simple "apply TurboQuant to KV" claim is not new. The new part must be the shared-prefix storage policy and evidence. |
| OSCAR | Attention-aware covariance rotations for deployable INT2 KV-cache quantization. | Rotation/calibration variants need strong full-task evidence and are not the current strongest novelty. |
| OScaR | Canalized rotation and token scaling for extreme KV-cache quantization. | Plain rotation/scale contributions have novelty pressure. |
| FibQuant | Vector codebook compression under the normalized rotation interface. | Block/vector codebook variants are not the best near-term path. |

Representative sources:

- TurboQuant: https://arxiv.org/abs/2504.19874
- OSCAR: https://arxiv.org/abs/2605.17757
- OScaR: https://arxiv.org/abs/2605.19660
- FibQuant: https://arxiv.org/abs/2605.11478

## Current Novelty Candidate

The strongest differentiable claim is:

> On an answer-preserving repeated long-context QA workload, fp16 Shared-KV exposes a
> systems-vs-codec Pareto point: exact sharing preserves fp16 quality with large latency
> and persistent-memory gains, while independent TurboQuant at similar memory is lower
> quality and slower. The storage decomposition then identifies the shared prefix table
> as the dominant remaining compression target, and guarded TurboQuant-compressed prefix
> storage is evaluated against the fp16 Shared-KV reference.

This yields two possible reportable layers:

1. Already strong: fp16 Shared-KV reference as a systems observation and evaluation
   protocol.
2. Now validated: guarded TurboQuant shared-prefix table as a storage-quality method
   with full 2400-row LongBench evidence against the fp16 Shared-KV reference.
3. Diagnostic only: dense materialized prefix caching shows that most current latency
   overhead comes from repeated dequantization/materialization, but it pays for this
   with extra transient memory and is not a final fused-kernel result.

The current implementation does not yet support a speedup claim for the guarded
TurboQuant prefix table because it still dequantizes/materializes prefix K/V in Python.

## What Not To Claim

- Do not claim prefix caching or shared-prefix attention as novel.
- Do not claim non-materializing shared-prefix decode; the Python split path failed the
  full row-level parity protocol.
- Do not claim TurboQuant improves latency in the guarded prefix prototype; current
  latency is worse than fp16 Shared-KV reference.
- Do not claim a general KV-cache compression improvement unless full-task quality beats
  the existing TurboQuant baselines.
