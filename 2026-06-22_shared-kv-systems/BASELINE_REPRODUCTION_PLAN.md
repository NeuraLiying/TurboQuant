# Official Baseline Reproduction Plan

This plan records a hard rule for publishable baseline results:

> If a baseline method has an official GitHub repository, the reportable baseline result
> must be produced by cloning and running that original project. Local reimplementation
> results are allowed only as sanity checks or adapters, not as paper-quality baseline
> evidence.

This differs from TurboQuant because TurboQuant is the method we are extending inside
this repository. For unrelated baselines, implementation drift would make comparisons
untrustworthy.

## Baseline Code Policy

1. Clone the official upstream repository when available.
2. Record repository URL, commit hash, license, environment, model checkpoint, dataset,
   command line, and raw output path.
3. Keep upstream code isolated under an external-baseline directory, for example:

   ```text
   external_baselines/<method_name>/
   ```

4. Do not patch upstream algorithm code unless required to fix environment breakage. Any
   patch must be logged as a diff and cannot change the method.
5. Use this repo only for:
   - preparing the repeated-context dataset in an upstream-compatible format;
   - launching official commands;
   - parsing raw baseline logs into a common result table;
   - running our own method and ablations.
6. If no official implementation exists, mark the baseline as `paper-only` and do not
   present a local reproduction as an official baseline. At most, report it as an
   approximate reimplementation with a separate label.

## Initial Baseline Registry

| Method | Official code status | Source / evidence | Action |
| --- | --- | --- | --- |
| CacheGen | Official repo found | `https://github.com/UChi-JCL/CacheGen` is linked from the SIGCOMM paper. | Clone official repo and run its documented compression/loading path. |
| SKVQ | Official repo found | `https://github.com/cat538/SKVQ` identifies itself as the official implementation. | Clone official repo and run LongBench-compatible KV quantization baseline if supported. |
| SpectrumKV | Official repo found | Paper artifact section links `https://github.com/YangSteve1223/kvcache-lab`. | Clone official repo; adapt only dataset/input wrappers if necessary. |
| KVTC | No official repo found yet | arXiv/OpenReview paper describes reusable KV transform coding but does not list a GitHub link in the arXiv metadata. | Search author/NVIDIA release; if unavailable, classify as paper-only or contact authors. |
| InnerQ | No official repo found yet | arXiv HTML describes method; no GitHub artifact link found in the paper text. | Search author releases; if unavailable, do not claim official reproduced baseline. |
| KVmix | No official repo found yet | AAAI/arXiv paper found; no official repo confirmed. | Search author releases; otherwise paper-only. |
| TriAxialKV | No official repo found yet | arXiv paper reports fused Triton system; no GitHub link found in arXiv metadata. | Search author releases; otherwise paper-only. |
| KVQuant | Official repo found | `https://github.com/squeezeailab/kvquant`. | Candidate low-bit KV quantization baseline if model/dataset support aligns. |
| kvpress | Official/library repo found | `https://github.com/NVIDIA/kvpress` implements multiple KV compression methods. | Useful framework baseline, but not a substitute for each method's official repo if one exists. |

## Required Result Metadata

Every publishable baseline run must produce a sidecar file:

```json
{
  "method": "SKVQ",
  "official_repo": "https://github.com/cat538/SKVQ",
  "commit": "...",
  "local_path": "external_baselines/SKVQ",
  "environment": "...",
  "model": "...",
  "dataset": "...",
  "command": "...",
  "raw_output": "...",
  "adapter_or_patch": "none or path to diff",
  "notes": "..."
}
```

## Immediate Execution Order

1. Clone and smoke-test official CacheGen and SKVQ first because their repositories are
   confirmed and they directly pressure the current claims.
2. Clone SpectrumKV next because it provides a recent mixed-precision policy with an
   artifact repository and detailed raw logs.
3. Continue searching for official KVTC, InnerQ, KVmix, and TriAxialKV repositories.
4. Only after official baselines are runnable, compare them against our improved
   shared-prefix policy on the same model/data slice.
