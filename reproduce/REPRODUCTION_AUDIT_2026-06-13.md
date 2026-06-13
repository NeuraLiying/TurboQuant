# Reproduction Audit - 2026-06-13

Scope: audit the Table 1 reproduction for `meta-llama/Llama-3.1-8B-Instruct`, limited to `Full Cache` and `TurboQuant`.

## Verdict

The generation pipeline is complete and reproducible for all 16 LongBench-V1 Table 1 tasks. The evaluation protocol had real metric mismatches and has been fixed.

After the metric fix:

| Method | KV Size | Source | SingleQA | MultiQA | Summarization | Few shot | Synthetic | Code | Average |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Cache | 16.0 | paper | 45.29 | 45.16 | 26.55 | 68.38 | 59.54 | 46.28 | 50.06 |
| Full Cache | 16.0 | local | 44.45 | 44.18 | 29.29 | 69.53 | 52.60 | 62.26 | 50.38 |
| TurboQuant | 2.5 | paper | 44.16 | 44.96 | 24.80 | 68.01 | 59.65 | 45.76 | 49.44 |
| TurboQuant | 2.5 | local | 38.61 | 36.01 | 27.54 | 67.12 | 48.22 | 55.02 | 45.42 |
| TurboQuant | 3.5 | paper | 45.01 | 45.31 | 26.00 | 68.63 | 59.95 | 46.17 | 50.06 |
| TurboQuant | 3.5 | local | 42.73 | 43.04 | 28.72 | 68.59 | 52.06 | 61.16 | 49.38 |

Conclusion: the generation and evaluation pipeline is complete for the selected Table 1 scope, and the comparison artifacts below record both the paper values and local reproduction values.

Current default comparison artifacts:

- `reproduce/TABLE1_OFFICIAL_COMPARISON.md`
- `reproduce/TABLE1_OFFICIAL_COMPARISON.json`
- `reproduce/TABLE1_OFFICIAL_COMPARISON.csv`

Pre-fix artifacts were archived under `reproduce/archive/TABLE1_OFFICIAL_COMPARISON_pre_metric_fix_2026-06-13.*`.

## Findings

1. Prompt and max generation length match the official THUDM LongBench `dataset2prompt.json` and `dataset2maxlen.json` for the English Table 1 tasks.
2. The local evaluator did not match official LongBench exactly:
   - Summarization used `rouge_score.RougeScorer` with stemming; official LongBench uses the `rouge` package and `rouge-l` F score.
   - Code tasks used `difflib.SequenceMatcher`; official LongBench uses `fuzzywuzzy.fuzz.ratio`.
3. The JSONL output omitted `all_classes`, which made old TREC outputs not fully self-contained for later rescoring. The generator now stores `all_classes`; the scorer also has a TREC fallback for legacy JSONL files.
4. The paper text has a protocol ambiguity: Section 4.3 says LongBench-E is used for balanced lengths, but Table 1 caption says LongBench-V1. The reproduced V1 Full Cache average is close to the paper after the metric fix, while the available LongBench-E diagnostic is not close. Therefore the current Table 1 comparison remains LongBench-V1.
5. The current TurboQuant cache path uses MSE TurboQuant for both key and value by default. Guarded diagnostic variants were added for `prod` key quantization, prefill-only quantization, non-symmetric key/value bit allocation, and the 32-outlier textual variant. None consistently improved 2.5-bit results.

## Code Changes From Audit

- `turboquant/longbench_metrics.py`
  - Uses official LongBench-compatible `rouge` and `fuzzywuzzy` metrics when installed.
  - Adds TREC class fallback for legacy rescoring.
- `scripts/summarize_jsonl_accuracy.py`
  - Adds `--recompute-longbench-score` to rescore existing predictions with the current metric implementation.
- `scripts/recompute_table1_official_metrics.py`
  - Adds parallel batch rescoring for Table 1 JSONL outputs.
- `experiments/longbench/run_full_cache_eval.py`
  - Stores `all_classes` in new JSONL outputs.
  - Adds guarded `--key-quantizer`, `--value-quantizer`, `--key-bits`, `--value-bits`, `--effective-bit-allocation`, and `--no-quantize-decode` flags for diagnostic TurboQuant variants.
- `turboquant/kv_cache.py`
  - Adds a guarded `PackedProdSegment` path for diagnostic `TurboQuantProd` key/value quantization.
  - Adds a guarded `quarter_high2` fractional-bit allocation diagnostic.
- `environment.yml`
  - Adds `rouge`, `fuzzywuzzy`, and `python-Levenshtein`.

Validation:

- `conda run -n turboquant python -m pytest tests -q`: 9 passed.
- `prod-key` smoke generation on `longbench_trec` 1 example passed.
- Full Table 1 outputs were rescored with official-compatible metrics.

## Current Status

This state is fixed as the reproducible Table 1 snapshot.

Manifest:

- `reproduce/REPRODUCTION_MANIFEST.json`

Reason: the evaluator and generation pipeline are audited, complete task outputs are available for all 16 selected LongBench-V1 tasks, and the default TurboQuant configuration below is used for the released reproduction snapshot.

Default TurboQuant reproduction configuration:

- `key_quantizer=mse`
- `value_quantizer=mse`
- `effective_bit_allocation=blend`
- `quantize_prefill=True`
- `quantize_decode=True`

2.5-bit diagnostic A/B on 20-example slices:

| Dataset | Full | Default 2.5 | quarter_high2 | prefill_only | key2/value3 | prod_key |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hotpotqa | 45.27 | 36.00 | 23.67 | 35.27 | 33.98 | 11.22 |
| 2wikimqa | 37.29 | 32.76 | 30.97 | 32.54 | 34.25 | 9.85 |

No full rerun was performed after these diagnostics.

## Residual Risk

1. The paper does not provide reference code, so exact hidden implementation choices cannot be fully ruled out.
2. The paper text has a LongBench-E versus LongBench-V1 ambiguity, though the local LongBench-V1 Full Cache average matches Table 1 closely after the metric fix.
3. Exact hidden implementation choices cannot be fully ruled out for TurboQuant bit-allocation and cache-update details.
