#!/usr/bin/env python3
"""Run full-cache LongBench generation over a local Arrow shard."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from time import perf_counter

import torch
import yaml
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import modeling_llama

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from turboquant.kv_cache import (
    OUTLIER_POLICIES,
    TOKEN_PROTECTION_POLICIES,
    TurboQuantDynamicCache,
    make_kv_config_from_effective_bits,
)
from turboquant.longbench_metrics import category_for_dataset, score_prediction
from turboquant.longbench_prompts import (
    build_longbench_prompt,
    dataset_name_from_row,
    max_new_tokens_for,
    should_apply_chat_template,
)


QUANTIZER_CHOICES = [
    "mse",
    "unit_mse",
    "lsq_mse",
    "lsq_unit_mse",
    "gain_mse",
    "regular_gain_mse",
    "regular_half_gain_mse",
    "regular_selected_gain_mse",
    "regular_clipped_gain_mse",
    "adaptive_regular_gain_mse",
    "attention_adaptive_regular_gain_mse",
    "clipped_gain_mse",
    "selected_gain_mse",
    "lowbit_gain_mse",
    "lowbit_half_gain_mse",
    "lowbit_clipped_gain_mse",
    "lowbit_selected_gain_mse",
    "dot_gain_mse",
    "hadamard_mse",
    "srht_mse",
    "calibrated_rotation_mse",
    "head_rotation_mse",
    "head_hadamard_mse",
    "outlier_hadamard_mse",
    "attention_adaptive_outlier_hadamard_mse",
    "attention_outlier_hadamard_mse",
    "bitwidth_attention_outlier_hadamard_mse",
    "entropy_guarded_outlier_hadamard_mse",
    "adaptive_outlier_hadamard_mse",
    "vector_adaptive_outlier_hadamard_mse",
    "margin_vector_outlier_hadamard_mse",
    "hadamard_residual_mse",
    "attention_hadamard_residual_mse",
    "attention_weighted_hadamard_residual_mse",
    "regular_outlier_hadamard_mse",
    "outlier_only_hadamard_mse",
    "attention_scale_mse",
    "regular_attention_scale_mse",
    "rotation_bank_mse",
    "attention_rotation_bank_mse",
    "segment_rotation_bank_mse",
    "rotated_outlier_mse",
    "attention_rotated_outlier_mse",
    "attention_adaptive_rotated_outlier_mse",
    "attention_adaptive_paired_rotated_outlier_mse",
    "entropy_guarded_paired_rotated_outlier_mse",
    "calibrated_rotated_outlier_mse",
    "paired_rotated_outlier_mse",
    "shared_rotated_outlier_mse",
    "rms_rotation_mse",
    "rms_regular_gain_mse",
    "centered_mse",
    "auto_mse",
    "attention_auto_mse",
    "distortion_regime_mse",
    "gain_unit_regime_mse",
    "rate_regime_mse",
    "hadamard_rate_regime_mse",
    "rate_hadamard_value_mse",
    "prod",
    "mse_block2",
    "learned_mse_block2",
    "learned_unit_mse_block2",
    "uniform_token",
    "uniform_channel",
]


def install_attention_error_cache_patch() -> None:
    """Pass Llama query states into TurboQuant cache updates for saliency-aware protection."""

    def forward_with_query_cache(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = modeling_llama.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "query_states": query_states,
                "scaling": self.scaling,
            }
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface = modeling_llama.eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = modeling_llama.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    modeling_llama.LlamaAttention.forward = forward_with_query_cache


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset_from_config(data_cfg: dict) -> Dataset:
    arrow_files = data_cfg.get("arrow_files") or []
    parquet_files = data_cfg.get("parquet_files") or []
    if arrow_files:
        datasets = [Dataset.from_file(path) for path in arrow_files]
    elif parquet_files:
        datasets = [Dataset.from_parquet(path) for path in parquet_files]
    else:
        raise ValueError("dataset config must contain either arrow_files or parquet_files")
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def build_legacy_prompt(row: dict) -> str:
    context = row["context"].strip()
    question = (row.get("question") or row.get("input") or "").strip()
    answer_prefix = (row.get("answer_prefix") or "Answer:").strip()
    if context.endswith(question):
        return f"{context}\n{answer_prefix}"
    return f"{context}\n\n{question}\n{answer_prefix}"


def truncate_middle(input_ids: torch.Tensor, max_input_tokens: int | None) -> torch.Tensor:
    if max_input_tokens is None or input_ids.shape[-1] <= max_input_tokens:
        return input_ids
    half = max_input_tokens // 2
    return torch.cat([input_ids[..., :half], input_ids[..., -half:]], dim=-1)


def apply_chat_template_if_needed(tokenizer, prompt: str, dataset_name: str, mode: str) -> str:
    if not should_apply_chat_template(dataset_name, mode):
        return prompt
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def is_code_completion_prompt(prompt: str, dataset_name: str | None = None) -> bool:
    if dataset_name in {"lcc", "repobench-p"}:
        return True
    head = prompt[:2048].lower()
    if "please complete the code given below" in head:
        return True
    code_markers = [
        "public class ",
        "def ",
        "import ",
        "#include",
        "using system;",
        "namespace ",
        "class ",
        "function ",
        "src/",
    ]
    marker_hits = sum(marker in head for marker in code_markers)
    return marker_hits >= 3 and ("complete the code" in head or dataset_name in {"lcc", "repobench-p"})


def prompt_structure_features(prompt: str) -> dict[str, int]:
    lower = prompt.lower()
    return {
        "passage_count": len(re.findall(r"\bpassage\s+\d+\s*:", lower)),
        "question_marks": prompt.count("?"),
    }


def contains_answer(prediction: str, answers: list[str]) -> bool:
    pred = normalize_text(prediction)
    return any(normalize_text(answer) in pred for answer in answers if answer)


def load_completed(output_path: Path) -> set[str]:
    completed: set[str] = set()
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            completed.add(str(record["index"]))
    return completed


def parse_layer_bits(text: str | None) -> tuple[float, ...] | None:
    if text is None:
        return None
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("layer bit schedule cannot be empty")
    return values


def parse_layer_ints(text: str | None) -> tuple[int, ...] | None:
    if text is None:
        return None
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("layer integer schedule cannot be empty")
    return values


def parse_layer_quantizers(text: str | None) -> tuple[str, ...] | None:
    if text is None:
        return None
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("layer quantizer schedule cannot be empty")
    invalid = [value for value in values if value not in QUANTIZER_CHOICES]
    if invalid:
        raise ValueError(f"unsupported layer quantizer(s): {invalid}")
    return values


def load_layer_channel_scores(path: str | None, key: str) -> tuple[tuple[float, ...], ...] | None:
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    values = data[key] if isinstance(data, dict) and key in data else data
    if not isinstance(values, list) or not values:
        raise ValueError(f"{key} channel scores must be a non-empty list")
    return tuple(tuple(float(item) for item in row) for row in values)


def load_layer_rotation_matrices(path: str | None, key: str) -> tuple[tuple[tuple[float, ...], ...], ...] | None:
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    values = data[key] if isinstance(data, dict) and key in data else data
    if not isinstance(values, list) or not values:
        raise ValueError(f"{key} rotation matrices must be a non-empty list")
    return tuple(
        tuple(tuple(float(item) for item in matrix_row) for matrix_row in matrix)
        for matrix in values
    )


def resolve_quantizer_preset(baseline_mode: str, key_quantizer: str, value_quantizer: str) -> tuple[str, str]:
    if baseline_mode == "naive_int":
        return "uniform_token", "uniform_token"
    if baseline_mode == "kivi_style":
        return "uniform_channel", "uniform_token"
    return key_quantizer, value_quantizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", default=str(PROJECT_ROOT / "configs/paths.yaml"))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/llama_first.yaml"))
    parser.add_argument("--dataset-key", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", choices=["single", "auto", "balanced"], default="single")
    parser.add_argument("--max-memory", action="append", default=[])
    parser.add_argument("--offload-folder", default=str(PROJECT_ROOT / "reproduce/offload"))
    parser.add_argument("--cache-mode", choices=["full", "turboquant"], default="full")
    parser.add_argument("--kv-bits", type=float, default=16.0)
    parser.add_argument("--fixed-kv-seed", type=int, default=None)
    parser.add_argument("--key-bits", type=float, default=None)
    parser.add_argument("--value-bits", type=float, default=None)
    parser.add_argument("--key-quantizer", choices=QUANTIZER_CHOICES, default="mse")
    parser.add_argument("--value-quantizer", choices=QUANTIZER_CHOICES, default="mse")
    parser.add_argument(
        "--long-prompt-key-quantizer",
        choices=QUANTIZER_CHOICES,
        default=None,
        help="Use this key quantizer when the encoded prompt length is greater than --long-prompt-threshold.",
    )
    parser.add_argument(
        "--long-prompt-value-quantizer",
        choices=QUANTIZER_CHOICES,
        default=None,
        help="Use this value quantizer when the encoded prompt length is greater than --long-prompt-threshold.",
    )
    parser.add_argument(
        "--long-prompt-threshold",
        type=int,
        default=None,
        help="Prompt-token threshold for switching to the long-prompt quantizers.",
    )
    parser.add_argument(
        "--long-prompt-max-tokens",
        type=int,
        default=None,
        help="Do not switch quantizers when the encoded prompt length is greater than this value.",
    )
    parser.add_argument(
        "--long-prompt-exclude-code",
        action="store_true",
        help="Do not switch quantizers for code-completion prompts when --long-prompt-threshold is active.",
    )
    parser.add_argument(
        "--long-prompt-max-passages",
        type=int,
        default=None,
        help="Do not switch quantizers when a Passage-style prompt contains more than this many passages.",
    )
    parser.add_argument(
        "--long-prompt-max-question-marks",
        type=int,
        default=None,
        help="Do not switch quantizers when the prompt contains more than this many question marks.",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=["turboquant", "naive_int", "kivi_style"],
        default="turboquant",
        help="Convenience presets: naive_int uses per-token uniform K/V; kivi_style uses per-channel K and per-token V.",
    )
    parser.add_argument("--layer-key-bits", default=None, help="Comma-separated key bit schedule by layer.")
    parser.add_argument("--layer-value-bits", default=None, help="Comma-separated value bit schedule by layer.")
    parser.add_argument("--layer-key-quantizers", default=None, help="Comma-separated key quantizer schedule by layer.")
    parser.add_argument("--layer-value-quantizers", default=None, help="Comma-separated value quantizer schedule by layer.")
    parser.add_argument("--layer-key-rotation-indices", default=None, help="Comma-separated key rotation seed index schedule.")
    parser.add_argument("--layer-value-rotation-indices", default=None, help="Comma-separated value rotation seed index schedule.")
    parser.add_argument("--effective-bit-allocation", choices=["blend", "quarter_high2"], default="blend")
    parser.add_argument("--outlier-policy", choices=sorted(OUTLIER_POLICIES), default="dynamic_absmean")
    parser.add_argument("--key-outlier-policy", choices=sorted(OUTLIER_POLICIES), default=None)
    parser.add_argument("--value-outlier-policy", choices=sorted(OUTLIER_POLICIES), default=None)
    parser.add_argument("--layer-key-channel-scores", default=None, help="JSON file with per-layer key channel scores.")
    parser.add_argument("--layer-value-channel-scores", default=None, help="JSON file with per-layer value channel scores.")
    parser.add_argument("--layer-key-rotation-matrices", default=None, help="JSON file with per-layer key rotation matrices.")
    parser.add_argument("--layer-value-rotation-matrices", default=None, help="JSON file with per-layer value rotation matrices.")
    parser.add_argument("--sensitivity-score-power", type=float, default=1.0)
    parser.add_argument("--token-protection-policy", choices=sorted(TOKEN_PROTECTION_POLICIES), default="none")
    parser.add_argument("--protected-start-tokens", type=int, default=4)
    parser.add_argument("--token-quant-bits", type=int, default=None)
    parser.add_argument("--token-protection-target-ratio", type=float, default=None)
    parser.add_argument("--token-protection-targets", choices=["both", "key", "value"], default="both")
    parser.add_argument("--attention-error-query-tokens", type=int, default=1)
    parser.add_argument("--attention-entropy-threshold", type=float, default=0.80)
    parser.add_argument("--rotation-bank-size", type=int, default=4)
    parser.add_argument("--outlier-hadamard-block-size", type=int, default=16)
    parser.add_argument("--no-quantize-decode", action="store_true")
    parser.add_argument("--codebook-grid-size", type=int, default=20_001)
    parser.add_argument("--turboquant-fast-materialized-eval", action="store_true")
    parser.add_argument("--prompt-mode", choices=["longbench", "legacy"], default="longbench")
    parser.add_argument("--chat-template-mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "reproduce/runs/longbench_full_cache_eval.jsonl"))
    args = parser.parse_args()
    if (
        args.token_protection_policy == "attention_error_budget"
        or args.outlier_policy in {"attention_error_gain", "joint_attention_error_gain"}
        or args.key_outlier_policy in {"attention_error_gain", "joint_attention_error_gain"}
        or args.value_outlier_policy in {"attention_error_gain", "joint_attention_error_gain"}
        or args.key_quantizer == "attention_auto_mse"
        or args.value_quantizer == "attention_auto_mse"
        or args.long_prompt_key_quantizer == "attention_auto_mse"
        or args.long_prompt_value_quantizer == "attention_auto_mse"
        or args.key_quantizer == "attention_hadamard_residual_mse"
        or args.value_quantizer == "attention_hadamard_residual_mse"
        or args.long_prompt_key_quantizer == "attention_hadamard_residual_mse"
        or args.long_prompt_value_quantizer == "attention_hadamard_residual_mse"
        or args.key_quantizer == "attention_weighted_hadamard_residual_mse"
        or args.value_quantizer == "attention_weighted_hadamard_residual_mse"
        or args.long_prompt_key_quantizer == "attention_weighted_hadamard_residual_mse"
        or args.long_prompt_value_quantizer == "attention_weighted_hadamard_residual_mse"
        or args.key_quantizer == "attention_outlier_hadamard_mse"
        or args.value_quantizer == "attention_outlier_hadamard_mse"
        or args.long_prompt_key_quantizer == "attention_outlier_hadamard_mse"
        or args.long_prompt_value_quantizer == "attention_outlier_hadamard_mse"
        or args.key_quantizer == "attention_adaptive_outlier_hadamard_mse"
        or args.value_quantizer == "attention_adaptive_outlier_hadamard_mse"
        or args.long_prompt_key_quantizer == "attention_adaptive_outlier_hadamard_mse"
        or args.long_prompt_value_quantizer == "attention_adaptive_outlier_hadamard_mse"
        or args.key_quantizer == "bitwidth_attention_outlier_hadamard_mse"
        or args.value_quantizer == "bitwidth_attention_outlier_hadamard_mse"
        or args.long_prompt_key_quantizer == "bitwidth_attention_outlier_hadamard_mse"
        or args.long_prompt_value_quantizer == "bitwidth_attention_outlier_hadamard_mse"
        or args.key_quantizer == "entropy_guarded_outlier_hadamard_mse"
        or args.value_quantizer == "entropy_guarded_outlier_hadamard_mse"
        or args.long_prompt_key_quantizer == "entropy_guarded_outlier_hadamard_mse"
        or args.long_prompt_value_quantizer == "entropy_guarded_outlier_hadamard_mse"
        or args.key_quantizer == "attention_rotation_bank_mse"
        or args.value_quantizer == "attention_rotation_bank_mse"
        or args.long_prompt_key_quantizer == "attention_rotation_bank_mse"
        or args.long_prompt_value_quantizer == "attention_rotation_bank_mse"
        or args.key_quantizer == "attention_rotated_outlier_mse"
        or args.value_quantizer == "attention_rotated_outlier_mse"
        or args.long_prompt_key_quantizer == "attention_rotated_outlier_mse"
        or args.long_prompt_value_quantizer == "attention_rotated_outlier_mse"
        or args.key_quantizer == "attention_adaptive_rotated_outlier_mse"
        or args.value_quantizer == "attention_adaptive_rotated_outlier_mse"
        or args.long_prompt_key_quantizer == "attention_adaptive_rotated_outlier_mse"
        or args.long_prompt_value_quantizer == "attention_adaptive_rotated_outlier_mse"
        or args.key_quantizer == "attention_adaptive_paired_rotated_outlier_mse"
        or args.value_quantizer == "attention_adaptive_paired_rotated_outlier_mse"
        or args.long_prompt_key_quantizer == "attention_adaptive_paired_rotated_outlier_mse"
        or args.long_prompt_value_quantizer == "attention_adaptive_paired_rotated_outlier_mse"
        or args.key_quantizer == "entropy_guarded_paired_rotated_outlier_mse"
        or args.value_quantizer == "entropy_guarded_paired_rotated_outlier_mse"
        or args.long_prompt_key_quantizer == "entropy_guarded_paired_rotated_outlier_mse"
        or args.long_prompt_value_quantizer == "entropy_guarded_paired_rotated_outlier_mse"
        or args.key_quantizer == "attention_scale_mse"
        or args.value_quantizer == "attention_scale_mse"
        or args.long_prompt_key_quantizer == "attention_scale_mse"
        or args.long_prompt_value_quantizer == "attention_scale_mse"
        or args.key_quantizer == "regular_attention_scale_mse"
        or args.value_quantizer == "regular_attention_scale_mse"
        or args.long_prompt_key_quantizer == "regular_attention_scale_mse"
        or args.long_prompt_value_quantizer == "regular_attention_scale_mse"
        or args.key_quantizer == "attention_adaptive_regular_gain_mse"
        or args.value_quantizer == "attention_adaptive_regular_gain_mse"
        or args.long_prompt_key_quantizer == "attention_adaptive_regular_gain_mse"
        or args.long_prompt_value_quantizer == "attention_adaptive_regular_gain_mse"
    ):
        install_attention_error_cache_patch()
    layer_key_bits = parse_layer_bits(args.layer_key_bits)
    layer_value_bits = parse_layer_bits(args.layer_value_bits)
    layer_key_quantizers = parse_layer_quantizers(args.layer_key_quantizers)
    layer_value_quantizers = parse_layer_quantizers(args.layer_value_quantizers)
    layer_key_rotation_indices = parse_layer_ints(args.layer_key_rotation_indices)
    layer_value_rotation_indices = parse_layer_ints(args.layer_value_rotation_indices)
    layer_key_channel_scores = load_layer_channel_scores(args.layer_key_channel_scores, "layer_key_channel_scores")
    layer_value_channel_scores = load_layer_channel_scores(args.layer_value_channel_scores, "layer_value_channel_scores")
    layer_key_rotation_matrices = load_layer_rotation_matrices(
        args.layer_key_rotation_matrices,
        "layer_key_rotation_matrices",
    )
    layer_value_rotation_matrices = load_layer_rotation_matrices(
        args.layer_value_rotation_matrices,
        "layer_value_rotation_matrices",
    )
    resolved_key_quantizer, resolved_value_quantizer = resolve_quantizer_preset(
        args.baseline_mode,
        args.key_quantizer,
        args.value_quantizer,
    )

    paths = load_yaml(Path(args.paths))
    cfg = load_yaml(Path(args.config))
    model_cfg = paths["models"][cfg["model_key"]]
    dataset_key = args.dataset_key or cfg["longbench"]["primary_dataset_key"]
    data_cfg = paths["datasets"][dataset_key]

    dataset = load_dataset_from_config(data_cfg)
    full_dataset_len = len(dataset)
    start_index = max(args.start_index, 0)
    end_index = args.end_index if args.end_index is not None else full_dataset_len
    end_index = min(end_index, full_dataset_len)
    if start_index > end_index:
        raise ValueError(f"start-index {start_index} is greater than end-index {end_index}")
    if args.max_examples is not None:
        end_index = min(end_index, start_index + args.max_examples)
    selected_indices = list(range(start_index, end_index))
    dataset = dataset.select(selected_indices)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(output_path) if args.resume else set()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["snapshot"], local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg["hardware"].get("dtype") == "bfloat16" else torch.float16
    max_memory = None
    if args.max_memory:
        max_memory = {}
        for item in args.max_memory:
            key, value = item.split("=", 1)
            max_memory[int(key) if key.isdigit() else key] = value
    device_map = {"": args.device} if args.device_map == "single" else args.device_map
    offload_folder = Path(args.offload_folder)
    if args.device_map != "single":
        offload_folder.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["snapshot"],
        local_files_only=True,
        torch_dtype=dtype,
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=str(offload_folder),
        attn_implementation="sdpa",
    )
    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None

    correct = 0
    total = 0
    skipped = 0
    started = perf_counter()
    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for offset, row in enumerate(dataset):
            idx = selected_indices[offset]
            if str(idx) in completed:
                skipped += 1
                continue
            dataset_name = dataset_name_from_row(row, data_cfg.get("requested_config") or data_cfg["config"])
            if args.prompt_mode == "longbench":
                prompt = build_longbench_prompt(row, dataset_name)
                max_new_tokens = max_new_tokens_for(dataset_name)
            else:
                prompt = build_legacy_prompt(row)
                max_new_tokens = int(row.get("max_new_tokens") or 64)
            prompt = apply_chat_template_if_needed(tokenizer, prompt, dataset_name, args.chat_template_mode)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
            inputs["input_ids"] = truncate_middle(inputs["input_ids"], args.max_input_tokens)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = truncate_middle(inputs["attention_mask"], args.max_input_tokens)
            input_device = model.get_input_embeddings().weight.device
            inputs = inputs.to(input_device)
            example_key_quantizer = resolved_key_quantizer
            example_value_quantizer = resolved_value_quantizer
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            code_completion_prompt = is_code_completion_prompt(prompt, dataset_name)
            structure_features = prompt_structure_features(prompt)
            passage_count = structure_features["passage_count"]
            question_marks = structure_features["question_marks"]
            passage_gate_blocked = (
                passage_count > 0
                and args.long_prompt_max_passages is not None
                and passage_count > args.long_prompt_max_passages
            )
            question_gate_blocked = (
                args.long_prompt_max_question_marks is not None
                and question_marks > args.long_prompt_max_question_marks
            )
            if (
                args.long_prompt_threshold is not None
                and prompt_tokens > args.long_prompt_threshold
                and (args.long_prompt_max_tokens is None or prompt_tokens <= args.long_prompt_max_tokens)
                and not (args.long_prompt_exclude_code and code_completion_prompt)
                and not passage_gate_blocked
                and not question_gate_blocked
            ):
                if args.long_prompt_key_quantizer is not None:
                    example_key_quantizer = args.long_prompt_key_quantizer
                if args.long_prompt_value_quantizer is not None:
                    example_value_quantizer = args.long_prompt_value_quantizer
            past_key_values = None
            if args.cache_mode == "turboquant":
                kv_cfg = make_kv_config_from_effective_bits(
                    args.kv_bits,
                    seed=args.fixed_kv_seed if args.fixed_kv_seed is not None else idx,
                    codebook_grid_size=args.codebook_grid_size,
                    key_quantizer=example_key_quantizer,
                    value_quantizer=example_value_quantizer,
                    effective_bit_allocation=args.effective_bit_allocation,
                    outlier_policy=args.outlier_policy,
                    key_outlier_policy=args.key_outlier_policy,
                    value_outlier_policy=args.value_outlier_policy,
                    fast_materialized_eval=args.turboquant_fast_materialized_eval,
                    layer_key_quantizers=layer_key_quantizers,
                    layer_value_quantizers=layer_value_quantizers,
                    layer_key_bits=layer_key_bits,
                    layer_value_bits=layer_value_bits,
                    layer_key_rotation_indices=layer_key_rotation_indices,
                    layer_value_rotation_indices=layer_value_rotation_indices,
                    layer_key_channel_scores=layer_key_channel_scores,
                    layer_value_channel_scores=layer_value_channel_scores,
                    layer_key_rotation_matrices=layer_key_rotation_matrices,
                    layer_value_rotation_matrices=layer_value_rotation_matrices,
                    sensitivity_score_power=args.sensitivity_score_power,
                    token_protection_policy=args.token_protection_policy,
                    protected_start_tokens=args.protected_start_tokens,
                    token_quant_bits=args.token_quant_bits,
                    token_protection_target_ratio=args.token_protection_target_ratio,
                    token_protection_targets=args.token_protection_targets,
                    attention_error_query_tokens=args.attention_error_query_tokens,
                    attention_entropy_threshold=args.attention_entropy_threshold,
                    rotation_bank_size=args.rotation_bank_size,
                    outlier_hadamard_block_size=args.outlier_hadamard_block_size,
                )
                kv_cfg = type(kv_cfg)(
                    key_bits=args.key_bits if args.key_bits is not None else kv_cfg.key_bits,
                    value_bits=args.value_bits if args.value_bits is not None else kv_cfg.value_bits,
                    seed=kv_cfg.seed,
                    codebook_grid_size=kv_cfg.codebook_grid_size,
                    quantize_prefill=kv_cfg.quantize_prefill,
                    quantize_decode=not args.no_quantize_decode,
                    key_quantizer=kv_cfg.key_quantizer,
                    value_quantizer=kv_cfg.value_quantizer,
                    effective_bit_allocation=kv_cfg.effective_bit_allocation,
                    outlier_policy=kv_cfg.outlier_policy,
                    key_outlier_policy=kv_cfg.key_outlier_policy,
                    value_outlier_policy=kv_cfg.value_outlier_policy,
                    fast_materialized_eval=kv_cfg.fast_materialized_eval,
                    layer_key_quantizers=kv_cfg.layer_key_quantizers,
                    layer_value_quantizers=kv_cfg.layer_value_quantizers,
                    layer_key_bits=kv_cfg.layer_key_bits,
                    layer_value_bits=kv_cfg.layer_value_bits,
                    layer_key_rotation_indices=kv_cfg.layer_key_rotation_indices,
                    layer_value_rotation_indices=kv_cfg.layer_value_rotation_indices,
                    layer_key_channel_scores=kv_cfg.layer_key_channel_scores,
                    layer_value_channel_scores=kv_cfg.layer_value_channel_scores,
                    layer_key_rotation_matrices=kv_cfg.layer_key_rotation_matrices,
                    layer_value_rotation_matrices=kv_cfg.layer_value_rotation_matrices,
                    sensitivity_score_power=kv_cfg.sensitivity_score_power,
                    token_protection_policy=kv_cfg.token_protection_policy,
                    protected_start_tokens=kv_cfg.protected_start_tokens,
                    token_quant_bits=kv_cfg.token_quant_bits,
                    token_protection_target_ratio=kv_cfg.token_protection_target_ratio,
                    token_protection_targets=kv_cfg.token_protection_targets,
                    attention_error_query_tokens=kv_cfg.attention_error_query_tokens,
                    attention_entropy_threshold=kv_cfg.attention_entropy_threshold,
                    rotation_bank_size=kv_cfg.rotation_bank_size,
                    outlier_hadamard_block_size=kv_cfg.outlier_hadamard_block_size,
                )
                past_key_values = TurboQuantDynamicCache(kv_cfg)
            example_started = perf_counter()
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_beams": 1,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
                "past_key_values": past_key_values,
            }
            if dataset_name == "samsum":
                newline_ids = tokenizer.encode("\n", add_special_tokens=False)
                if newline_ids:
                    generate_kwargs["min_length"] = int(inputs["input_ids"].shape[-1]) + 1
                    generate_kwargs["eos_token_id"] = [tokenizer.eos_token_id, newline_ids[-1]]
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    **generate_kwargs,
                )
            latency = perf_counter() - example_started
            new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)
            answers = row["answers"]
            is_correct = contains_answer(prediction, answers)
            longbench_score = score_prediction(dataset_name, prediction, answers, row.get("all_classes"))
            longbench_category = category_for_dataset(dataset_name)
            cache_stats = {}
            if past_key_values is not None and hasattr(past_key_values, "storage_nbytes"):
                storage_nbytes = int(past_key_values.storage_nbytes())
                materialized_nbytes = int(past_key_values.materialized_nbytes())
                cache_stats = {
                    "cache_storage_nbytes": storage_nbytes,
                    "cache_materialized_nbytes": materialized_nbytes,
                    "cache_storage_ratio": storage_nbytes / materialized_nbytes if materialized_nbytes else None,
                }
                if hasattr(past_key_values, "compression_summary"):
                    cache_stats["cache_compression_summary"] = past_key_values.compression_summary()
                if hasattr(past_key_values, "fast_materialized_nbytes"):
                    cache_stats["cache_fast_materialized_nbytes"] = int(past_key_values.fast_materialized_nbytes())
            correct += int(is_correct)
            total += 1
            record = {
                "index": idx,
                "id": row.get("_id"),
                "dataset": row.get("dataset"),
                "task": row.get("task"),
                "longbench_dataset": dataset_name,
                "longbench_category": longbench_category,
                "prompt_mode": args.prompt_mode,
                "chat_template_mode": args.chat_template_mode,
                "max_input_tokens": args.max_input_tokens,
                "cache_mode": args.cache_mode,
                "kv_bits": args.kv_bits,
                "fixed_kv_seed": args.fixed_kv_seed,
                "key_bits": args.key_bits,
                "value_bits": args.value_bits,
                "baseline_mode": args.baseline_mode,
                "key_quantizer": past_key_values.config.key_quantizer if past_key_values is not None else args.key_quantizer,
                "value_quantizer": past_key_values.config.value_quantizer if past_key_values is not None else args.value_quantizer,
                "layer_key_bits": list(layer_key_bits) if layer_key_bits is not None else None,
                "layer_value_bits": list(layer_value_bits) if layer_value_bits is not None else None,
                "layer_key_quantizers": list(layer_key_quantizers) if layer_key_quantizers is not None else None,
                "layer_value_quantizers": list(layer_value_quantizers) if layer_value_quantizers is not None else None,
                "layer_key_rotation_indices": list(layer_key_rotation_indices) if layer_key_rotation_indices is not None else None,
                "layer_value_rotation_indices": (
                    list(layer_value_rotation_indices) if layer_value_rotation_indices is not None else None
                ),
                "long_prompt_threshold": args.long_prompt_threshold,
                "long_prompt_max_tokens": args.long_prompt_max_tokens,
                "long_prompt_exclude_code": args.long_prompt_exclude_code,
                "long_prompt_max_passages": args.long_prompt_max_passages,
                "long_prompt_max_question_marks": args.long_prompt_max_question_marks,
                "code_completion_prompt": code_completion_prompt,
                "prompt_passage_count": passage_count,
                "prompt_question_marks": question_marks,
                "passage_gate_blocked": passage_gate_blocked,
                "question_gate_blocked": question_gate_blocked,
                "long_prompt_key_quantizer": args.long_prompt_key_quantizer,
                "long_prompt_value_quantizer": args.long_prompt_value_quantizer,
                "effective_key_quantizer": example_key_quantizer,
                "effective_value_quantizer": example_value_quantizer,
                "effective_bit_allocation": args.effective_bit_allocation,
                "outlier_policy": args.outlier_policy,
                "key_outlier_policy": args.key_outlier_policy,
                "value_outlier_policy": args.value_outlier_policy,
                "layer_key_channel_scores": args.layer_key_channel_scores,
                "layer_value_channel_scores": args.layer_value_channel_scores,
                "layer_key_rotation_matrices": args.layer_key_rotation_matrices,
                "layer_value_rotation_matrices": args.layer_value_rotation_matrices,
                "sensitivity_score_power": args.sensitivity_score_power,
                "token_protection_policy": args.token_protection_policy,
                "protected_start_tokens": args.protected_start_tokens,
                "token_quant_bits": args.token_quant_bits,
                "token_protection_target_ratio": args.token_protection_target_ratio,
                "token_protection_targets": args.token_protection_targets,
                "attention_error_query_tokens": args.attention_error_query_tokens,
                "attention_entropy_threshold": args.attention_entropy_threshold,
                "rotation_bank_size": args.rotation_bank_size,
                "outlier_hadamard_block_size": args.outlier_hadamard_block_size,
                "quantize_decode": not args.no_quantize_decode,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(new_tokens.shape[-1]),
                "max_new_tokens": max_new_tokens,
                "latency_seconds": latency,
                "prediction": prediction,
                "answers": answers,
                "all_classes": row.get("all_classes"),
                "contains_answer": is_correct,
                "longbench_score": longbench_score,
                **cache_stats,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if args.progress_every > 0 and total % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "completed_this_run": total,
                            "skipped": skipped,
                            "dataset_index": idx,
                            "requested": len(selected_indices),
                            "latency_seconds": round(latency, 4),
                            "contains_answer": is_correct,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    elapsed = perf_counter() - started
    # Summary for this invocation only. For resumed aggregate metrics, use the
    # JSONL records directly.
    summary = {
        "model": model_cfg["repo_id"],
        "dataset_key": dataset_key,
        "dataset": data_cfg["repo_id"],
        "config": data_cfg["config"],
        "split": data_cfg["split"],
        "cache_mode": args.cache_mode,
        "kv_bits": args.kv_bits,
        "fixed_kv_seed": args.fixed_kv_seed,
        "key_bits": args.key_bits,
        "value_bits": args.value_bits,
        "baseline_mode": args.baseline_mode,
        "key_quantizer": resolved_key_quantizer,
        "value_quantizer": resolved_value_quantizer,
        "layer_key_bits": list(layer_key_bits) if layer_key_bits is not None else None,
        "layer_value_bits": list(layer_value_bits) if layer_value_bits is not None else None,
        "layer_key_quantizers": list(layer_key_quantizers) if layer_key_quantizers is not None else None,
        "layer_value_quantizers": list(layer_value_quantizers) if layer_value_quantizers is not None else None,
        "layer_key_rotation_indices": list(layer_key_rotation_indices) if layer_key_rotation_indices is not None else None,
        "layer_value_rotation_indices": list(layer_value_rotation_indices) if layer_value_rotation_indices is not None else None,
        "long_prompt_threshold": args.long_prompt_threshold,
        "long_prompt_max_tokens": args.long_prompt_max_tokens,
        "long_prompt_key_quantizer": args.long_prompt_key_quantizer,
        "long_prompt_value_quantizer": args.long_prompt_value_quantizer,
        "effective_bit_allocation": args.effective_bit_allocation,
        "outlier_policy": args.outlier_policy,
        "key_outlier_policy": args.key_outlier_policy,
        "value_outlier_policy": args.value_outlier_policy,
        "layer_key_channel_scores": args.layer_key_channel_scores,
        "layer_value_channel_scores": args.layer_value_channel_scores,
        "layer_key_rotation_matrices": args.layer_key_rotation_matrices,
        "layer_value_rotation_matrices": args.layer_value_rotation_matrices,
        "sensitivity_score_power": args.sensitivity_score_power,
        "token_protection_policy": args.token_protection_policy,
        "protected_start_tokens": args.protected_start_tokens,
        "token_quant_bits": args.token_quant_bits,
        "token_protection_target_ratio": args.token_protection_target_ratio,
        "token_protection_targets": args.token_protection_targets,
        "attention_error_query_tokens": args.attention_error_query_tokens,
        "attention_entropy_threshold": args.attention_entropy_threshold,
        "rotation_bank_size": args.rotation_bank_size,
        "outlier_hadamard_block_size": args.outlier_hadamard_block_size,
        "quantize_decode": not args.no_quantize_decode,
        "device_map": args.device_map,
        "max_memory": args.max_memory,
        "offload_folder": args.offload_folder if args.device_map != "single" else None,
        "prompt_mode": args.prompt_mode,
        "chat_template_mode": args.chat_template_mode,
        "max_input_tokens": args.max_input_tokens,
        "turboquant_fast_materialized_eval": args.turboquant_fast_materialized_eval,
        "full_dataset_len": full_dataset_len,
        "start_index": start_index,
        "end_index": end_index,
        "num_examples_requested": len(selected_indices),
        "num_examples_run": total,
        "num_examples_skipped": skipped,
        "contains_answer_accuracy_this_run": correct / total if total else 0.0,
        "elapsed_seconds": elapsed,
        "output": str(output_path),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
