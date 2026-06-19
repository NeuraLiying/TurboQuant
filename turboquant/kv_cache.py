"""Transformers cache wrappers for TurboQuant KV experiments."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import ceil, floor
from operator import mul
from typing import Any, Optional

import torch
from transformers.cache_utils import Cache

from .core import (
    BlockMSEQuantized,
    MSEQuantized,
    ProdQuantized,
    TurboQuantBlockMSE,
    TurboQuantLearnedBlockMSE,
    TurboQuantMSE,
    TurboQuantProd,
    hadamard_orthogonal,
    random_orthogonal,
)

QUANTIZER_KINDS = {
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
}
EFFECTIVE_BIT_ALLOCATION_POLICIES = {"blend", "quarter_high2"}
OUTLIER_POLICIES = {
    "dynamic_absmean",
    "error_gain",
    "attention_error_gain",
    "joint_error_gain",
    "joint_attention_error_gain",
    "static_score",
    "sensitivity_error_gain",
    "head_dynamic_absmean",
    "head_error_gain",
}
TOKEN_PROTECTION_POLICIES = {"none", "sink_recent_budget", "norm_aware_budget", "attention_error_budget"}


@dataclass(frozen=True)
class KVQuantConfig:
    key_bits: float = 4
    value_bits: float = 4
    seed: int = 0
    codebook_grid_size: int = 20_001
    quantize_prefill: bool = True
    quantize_decode: bool = True
    key_quantizer: str = "mse"
    value_quantizer: str = "mse"
    effective_bit_allocation: str = "blend"
    outlier_policy: str = "dynamic_absmean"
    key_outlier_policy: str | None = None
    value_outlier_policy: str | None = None
    fast_materialized_eval: bool = False
    layer_key_quantizers: tuple[str, ...] | None = None
    layer_value_quantizers: tuple[str, ...] | None = None
    layer_key_bits: tuple[float, ...] | None = None
    layer_value_bits: tuple[float, ...] | None = None
    layer_key_rotation_indices: tuple[int, ...] | None = None
    layer_value_rotation_indices: tuple[int, ...] | None = None
    layer_key_channel_scores: tuple[tuple[float, ...], ...] | None = None
    layer_value_channel_scores: tuple[tuple[float, ...], ...] | None = None
    layer_key_rotation_matrices: tuple[tuple[tuple[float, ...], ...], ...] | None = None
    layer_value_rotation_matrices: tuple[tuple[tuple[float, ...], ...], ...] | None = None
    sensitivity_score_power: float = 1.0
    token_protection_policy: str = "none"
    protected_start_tokens: int = 4
    token_quant_bits: int | None = None
    token_protection_target_ratio: float | None = None
    token_protection_targets: str = "both"
    attention_error_query_tokens: int = 1
    attention_entropy_threshold: float = 0.80
    rotation_bank_size: int = 4
    rotation_bank_seed_stride: int = 100_003
    outlier_hadamard_block_size: int = 16

    def __post_init__(self) -> None:
        if self.key_quantizer not in QUANTIZER_KINDS:
            raise ValueError(f"unsupported key quantizer: {self.key_quantizer}")
        if self.value_quantizer not in QUANTIZER_KINDS:
            raise ValueError(f"unsupported value quantizer: {self.value_quantizer}")
        if self.effective_bit_allocation not in EFFECTIVE_BIT_ALLOCATION_POLICIES:
            raise ValueError(f"unsupported effective-bit allocation: {self.effective_bit_allocation}")
        if self.outlier_policy not in OUTLIER_POLICIES:
            raise ValueError(f"unsupported outlier policy: {self.outlier_policy}")
        if self.key_outlier_policy is not None and self.key_outlier_policy not in OUTLIER_POLICIES:
            raise ValueError(f"unsupported key outlier policy: {self.key_outlier_policy}")
        if self.value_outlier_policy is not None and self.value_outlier_policy not in OUTLIER_POLICIES:
            raise ValueError(f"unsupported value outlier policy: {self.value_outlier_policy}")
        if self.layer_key_quantizers is not None:
            if not self.layer_key_quantizers:
                raise ValueError("layer_key_quantizers must be non-empty when provided")
            for quantizer in self.layer_key_quantizers:
                if quantizer not in QUANTIZER_KINDS:
                    raise ValueError(f"unsupported layer key quantizer: {quantizer}")
        if self.layer_value_quantizers is not None:
            if not self.layer_value_quantizers:
                raise ValueError("layer_value_quantizers must be non-empty when provided")
            for quantizer in self.layer_value_quantizers:
                if quantizer not in QUANTIZER_KINDS:
                    raise ValueError(f"unsupported layer value quantizer: {quantizer}")
        if self.layer_key_bits is not None and not self.layer_key_bits:
            raise ValueError("layer_key_bits must be non-empty when provided")
        if self.layer_value_bits is not None and not self.layer_value_bits:
            raise ValueError("layer_value_bits must be non-empty when provided")
        if self.layer_key_rotation_indices is not None and not self.layer_key_rotation_indices:
            raise ValueError("layer_key_rotation_indices must be non-empty when provided")
        if self.layer_value_rotation_indices is not None and not self.layer_value_rotation_indices:
            raise ValueError("layer_value_rotation_indices must be non-empty when provided")
        for name, schedule in (
            ("layer_key_rotation_indices", self.layer_key_rotation_indices),
            ("layer_value_rotation_indices", self.layer_value_rotation_indices),
        ):
            if schedule is not None:
                for rotation_idx in schedule:
                    if rotation_idx < 0 or rotation_idx > 255:
                        raise ValueError(f"{name} values must be in [0, 255]")
        if self.layer_key_channel_scores is not None and not self.layer_key_channel_scores:
            raise ValueError("layer_key_channel_scores must be non-empty when provided")
        if self.layer_value_channel_scores is not None and not self.layer_value_channel_scores:
            raise ValueError("layer_value_channel_scores must be non-empty when provided")
        if self.layer_key_rotation_matrices is not None and not self.layer_key_rotation_matrices:
            raise ValueError("layer_key_rotation_matrices must be non-empty when provided")
        if self.layer_value_rotation_matrices is not None and not self.layer_value_rotation_matrices:
            raise ValueError("layer_value_rotation_matrices must be non-empty when provided")
        if self.sensitivity_score_power < 0:
            raise ValueError("sensitivity_score_power must be non-negative")
        if self.token_protection_policy not in TOKEN_PROTECTION_POLICIES:
            raise ValueError(f"unsupported token protection policy: {self.token_protection_policy}")
        if self.token_protection_targets not in {"both", "key", "value"}:
            raise ValueError(f"unsupported token protection targets: {self.token_protection_targets}")
        if self.protected_start_tokens < 0:
            raise ValueError("protected_start_tokens must be non-negative")
        if self.token_quant_bits is not None and self.token_quant_bits <= 0:
            raise ValueError("token_quant_bits must be positive when provided")
        if self.token_protection_target_ratio is not None and self.token_protection_target_ratio <= 0:
            raise ValueError("token_protection_target_ratio must be positive when provided")
        if self.attention_error_query_tokens <= 0:
            raise ValueError("attention_error_query_tokens must be positive")
        if not 0.0 <= self.attention_entropy_threshold <= 1.0:
            raise ValueError("attention_entropy_threshold must be in [0, 1]")
        if self.rotation_bank_size <= 0 or self.rotation_bank_size > 256:
            raise ValueError("rotation_bank_size must be in [1, 256]")
        if self.rotation_bank_seed_stride <= 0:
            raise ValueError("rotation_bank_seed_stride must be positive")
        if self.outlier_hadamard_block_size <= 0 or self.outlier_hadamard_block_size > 256:
            raise ValueError("outlier_hadamard_block_size must be in [1, 256]")


@dataclass
class PackedMSESegment:
    """Packed TurboQuantMSE representation for one cache update segment."""

    packed_indices: torch.Tensor
    norms: torch.Tensor
    shape: tuple[int, ...]
    bits: int
    dtype: torch.dtype
    block_size: int = 1
    quantizer_kind: str = "mse"
    packed_rotation_ids: torch.Tensor | None = None
    rotation_id_bits: int | None = None
    rotation_bank_size: int | None = None

    @property
    def sequence_length(self) -> int:
        return self.shape[-2]

    @property
    def index_shape(self) -> tuple[int, ...]:
        if self.block_size == 1:
            return self.shape
        return (*self.shape[:-1], self.shape[-1] // self.block_size)

    @property
    def nbytes(self) -> int:
        rotation_nbytes = (
            0 if self.packed_rotation_ids is None else self.packed_rotation_ids.numel() * self.packed_rotation_ids.element_size()
        )
        return (
            self.packed_indices.numel() * self.packed_indices.element_size()
            + self.norms.numel() * self.norms.element_size()
            + rotation_nbytes
        )


@dataclass
class PackedProdSegment:
    """Packed TurboQuantProd representation for one cache update segment."""

    packed_mse_indices: torch.Tensor
    mse_norms: torch.Tensor
    packed_qjl_signs: torch.Tensor
    residual_norms: torch.Tensor
    shape: tuple[int, ...]
    bits: int
    dtype: torch.dtype

    @property
    def sequence_length(self) -> int:
        return self.shape[-2]

    @property
    def nbytes(self) -> int:
        return (
            self.packed_mse_indices.numel() * self.packed_mse_indices.element_size()
            + self.mse_norms.numel() * self.mse_norms.element_size()
            + self.packed_qjl_signs.numel() * self.packed_qjl_signs.element_size()
            + self.residual_norms.numel() * self.residual_norms.element_size()
        )


@dataclass
class UniformAffineSegment:
    """Packed symmetric affine integer quantization segment."""

    packed_values: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    bits: int
    dtype: torch.dtype
    granularity: str

    @property
    def sequence_length(self) -> int:
        return self.shape[-2]

    @property
    def nbytes(self) -> int:
        return self.packed_values.numel() * self.packed_values.element_size() + self.scales.numel() * self.scales.element_size()


@dataclass
class CenteredSegment:
    """Segment that stores a full-precision sequence mean plus quantized residuals."""

    center: torch.Tensor
    residual: CacheSegment
    shape: tuple[int, ...]
    dtype: torch.dtype
    bits: int

    @property
    def sequence_length(self) -> int:
        return self.shape[-2]

    @property
    def nbytes(self) -> int:
        return self.center.numel() * self.center.element_size() + self.residual.nbytes


@dataclass
class OutlierHadamardSegment:
    """Packed segment using a data-dependent permutation plus block-Hadamard rotation."""

    packed_indices: torch.Tensor
    norms: torch.Tensor
    permutation: torch.Tensor
    signs: torch.Tensor
    shape: tuple[int, ...]
    bits: int
    dtype: torch.dtype
    block_size: int
    quantizer_kind: str = "outlier_hadamard_mse"

    @property
    def sequence_length(self) -> int:
        return self.shape[-2]

    @property
    def nbytes(self) -> int:
        return (
            self.packed_indices.numel() * self.packed_indices.element_size()
            + self.norms.numel() * self.norms.element_size()
            + self.permutation.numel() * self.permutation.element_size()
            + self.signs.numel() * self.signs.element_size()
        )


@dataclass
class RmsScaledSegment:
    """Segment that applies per-channel RMS preconditioning before TurboQuant."""

    scales: torch.Tensor
    residual: CacheSegment
    shape: tuple[int, ...]
    dtype: torch.dtype
    bits: int

    @property
    def sequence_length(self) -> int:
        return self.shape[-2]

    @property
    def nbytes(self) -> int:
        return self.scales.numel() * self.scales.element_size() + self.residual.nbytes


@dataclass
class VectorAdaptiveRotationSegment:
    """Per-vector mixture of baseline TurboQuant and an alternate rotation path."""

    baseline: CacheSegment
    candidate: CacheSegment
    packed_candidate_mask: torch.Tensor
    selector_shape: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: torch.dtype
    bits: int
    candidate_kind: str

    @property
    def sequence_length(self) -> int:
        return self.shape[-2]

    @property
    def nbytes(self) -> int:
        return self.baseline.nbytes + self.candidate.nbytes + self.packed_candidate_mask.numel() * self.packed_candidate_mask.element_size()


@dataclass
class HeadRotationMSESegment:
    """Packed TurboQuant segment using a deterministic independent rotation per KV head."""

    head_segments: tuple[CacheSegment, ...]
    shape: tuple[int, ...]
    dtype: torch.dtype
    bits: float

    @property
    def sequence_length(self) -> int:
        return self.shape[-2]

    @property
    def nbytes(self) -> int:
        return sum(segment.nbytes for segment in self.head_segments)


@dataclass
class HeadHadamardSegment:
    """Segment quantized after an orthogonal Hadamard mix across KV heads."""

    residual: CacheSegment
    shape: tuple[int, ...]
    dtype: torch.dtype
    bits: float

    @property
    def sequence_length(self) -> int:
        return self.shape[-2]

    @property
    def nbytes(self) -> int:
        return self.residual.nbytes


@dataclass
class HadamardResidualSegment:
    """Low-bit TurboQuant plus sparse Hadamard residual-sign correction."""

    base: CacheSegment
    packed_residual_signs: torch.Tensor
    residual_scales: torch.Tensor
    residual_indices: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    base_bits: int
    requested_bits: float

    @property
    def sequence_length(self) -> int:
        return self.shape[-2]

    @property
    def bits(self) -> float:
        return float(self.requested_bits)

    @property
    def nbytes(self) -> int:
        return (
            self.base.nbytes
            + self.packed_residual_signs.numel() * self.packed_residual_signs.element_size()
            + self.residual_scales.numel() * self.residual_scales.element_size()
            + self.residual_indices.numel() * self.residual_indices.element_size()
        )

    @property
    def effective_index_bits(self) -> float:
        return float(self.base_bits) + float(self.residual_indices.numel()) / float(self.shape[-1])


@dataclass
class OutlierMSESegment:
    """Two-part packed TurboQuantMSE segment for non-integer effective bits."""

    regular: CacheSegment
    outlier: CacheSegment
    regular_indices: torch.Tensor
    outlier_indices: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    requested_bits: float
    policy: str

    @property
    def sequence_length(self) -> int:
        return self.shape[-2]

    @property
    def regular_bits(self) -> int:
        return self.regular.bits

    @property
    def outlier_bits(self) -> int:
        return self.outlier.bits

    @property
    def nbytes(self) -> int:
        return (
            self.regular.nbytes
            + self.outlier.nbytes
            + self.regular_indices.numel() * self.regular_indices.element_size()
            + self.outlier_indices.numel() * self.outlier_indices.element_size()
        )

    @property
    def effective_index_bits(self) -> float:
        dimension = self.shape[-1]
        return (
            self.regular_indices.numel() * self.regular.bits + self.outlier_indices.numel() * self.outlier.bits
        ) / dimension


@dataclass
class RotatedOutlierMSESegment:
    """Fractional-bit TurboQuant segment with high-bit coordinates selected after rotation."""

    packed_regular_indices: torch.Tensor
    regular_norms: torch.Tensor
    packed_outlier_indices: torch.Tensor
    outlier_norms: torch.Tensor
    regular_indices: torch.Tensor
    outlier_indices: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    regular_bits: int
    outlier_bits: int
    requested_bits: float
    quantizer_kind: str = "rotated_outlier_mse"

    @property
    def sequence_length(self) -> int:
        return self.shape[-2]

    @property
    def bits(self) -> float:
        return float(self.requested_bits)

    @property
    def nbytes(self) -> int:
        return (
            self.packed_regular_indices.numel() * self.packed_regular_indices.element_size()
            + self.regular_norms.numel() * self.regular_norms.element_size()
            + self.packed_outlier_indices.numel() * self.packed_outlier_indices.element_size()
            + self.outlier_norms.numel() * self.outlier_norms.element_size()
            + self.regular_indices.numel() * self.regular_indices.element_size()
            + self.outlier_indices.numel() * self.outlier_indices.element_size()
        )

    @property
    def effective_index_bits(self) -> float:
        dimension = self.shape[-1]
        return (
            self.regular_indices.numel() * self.regular_bits + self.outlier_indices.numel() * self.outlier_bits
        ) / dimension


@dataclass
class HeadAdaptiveOutlierMSESegment:
    """Outlier segment with independent high-bit channel choices per KV head."""

    head_segments: tuple[OutlierMSESegment, ...]
    shape: tuple[int, ...]
    dtype: torch.dtype
    requested_bits: float
    policy: str

    @property
    def sequence_length(self) -> int:
        return self.shape[-2]

    @property
    def nbytes(self) -> int:
        return sum(segment.nbytes for segment in self.head_segments)

    @property
    def effective_index_bits(self) -> float:
        if not self.head_segments:
            return 0.0
        return sum(segment.effective_index_bits for segment in self.head_segments) / len(self.head_segments)


@dataclass
class RawTensorSegment:
    """Fallback segment for unquantized cache settings."""

    tensor: torch.Tensor

    @property
    def sequence_length(self) -> int:
        return self.tensor.shape[-2]

    @property
    def nbytes(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()


@dataclass
class TokenProtectedSegment:
    """Mixed segment with selected sequence positions stored in full precision."""

    raw_tensor: torch.Tensor
    quantized: CacheSegment
    raw_indices: torch.Tensor
    quantized_indices: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    target_bits: float
    quant_bits: int
    policy: str

    @property
    def sequence_length(self) -> int:
        return self.shape[-2]

    @property
    def nbytes(self) -> int:
        return (
            self.raw_tensor.numel() * self.raw_tensor.element_size()
            + self.quantized.nbytes
            + self.raw_indices.numel() * self.raw_indices.element_size()
            + self.quantized_indices.numel() * self.quantized_indices.element_size()
        )

    @property
    def effective_token_bits(self) -> float:
        seq_len = self.shape[-2]
        if seq_len == 0:
            return 0.0
        raw_count = int(self.raw_indices.numel())
        quant_count = int(self.quantized_indices.numel())
        return (16 * raw_count + self.quant_bits * quant_count) / seq_len


CacheSegment = (
    PackedMSESegment
    | PackedProdSegment
    | UniformAffineSegment
    | CenteredSegment
    | OutlierHadamardSegment
    | RmsScaledSegment
    | VectorAdaptiveRotationSegment
    | HeadRotationMSESegment
    | HeadHadamardSegment
    | HadamardResidualSegment
    | OutlierMSESegment
    | RotatedOutlierMSESegment
    | HeadAdaptiveOutlierMSESegment
    | RawTensorSegment
    | TokenProtectedSegment
)


def _numel(shape: tuple[int, ...]) -> int:
    return int(reduce(mul, shape, 1))


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack non-negative integer indices into a little-endian uint8 bitstream."""

    if bits <= 0 or bits > 8:
        raise ValueError("pack_indices supports 1..8 bits")
    flat = indices.reshape(-1).to(dtype=torch.int32)
    if flat.numel() == 0:
        return torch.empty(0, device=indices.device, dtype=torch.uint8)
    if torch.any(flat < 0) or torch.any(flat >= (1 << bits)):
        raise ValueError(f"indices contain values outside the {bits}-bit range")

    num_bytes = (flat.numel() * bits + 7) // 8
    bit_positions = torch.arange(flat.numel(), device=indices.device, dtype=torch.int64) * bits
    byte_positions = bit_positions // 8
    bit_offsets = (bit_positions % 8).to(dtype=torch.int32)

    packed = torch.zeros(num_bytes, device=indices.device, dtype=torch.int32)
    low = torch.bitwise_and(torch.bitwise_left_shift(flat, bit_offsets), 0xFF)
    packed.index_add_(0, byte_positions, low)

    overflow = bit_offsets + bits - 8
    overflow_mask = overflow > 0
    if bool(overflow_mask.any()):
        high = torch.bitwise_right_shift(flat[overflow_mask], 8 - bit_offsets[overflow_mask])
        packed.index_add_(0, byte_positions[overflow_mask] + 1, high)

    return packed.to(dtype=torch.uint8)


def unpack_indices(packed: torch.Tensor, *, bits: int, shape: tuple[int, ...]) -> torch.Tensor:
    """Unpack a little-endian uint8 bitstream created by `pack_indices`."""

    if bits <= 0 or bits > 8:
        raise ValueError("unpack_indices supports 1..8 bits")
    total_values = _numel(shape)
    if total_values == 0:
        return torch.empty(shape, device=packed.device, dtype=torch.int16)

    packed_i32 = packed.to(dtype=torch.int32)
    expected_bytes = (total_values * bits + 7) // 8
    if packed_i32.numel() != expected_bytes:
        raise ValueError(f"packed length mismatch: expected {expected_bytes}, got {packed_i32.numel()}")

    bit_positions = torch.arange(total_values, device=packed.device, dtype=torch.int64) * bits
    byte_positions = bit_positions // 8
    bit_offsets = (bit_positions % 8).to(dtype=torch.int32)

    values = torch.bitwise_right_shift(packed_i32[byte_positions], bit_offsets)
    overflow = bit_offsets + bits - 8
    overflow_mask = overflow > 0
    if bool(overflow_mask.any()):
        high = torch.bitwise_left_shift(packed_i32[byte_positions[overflow_mask] + 1], 8 - bit_offsets[overflow_mask])
        values[overflow_mask] = torch.bitwise_or(values[overflow_mask], high)

    values = torch.bitwise_and(values, (1 << bits) - 1)
    return values.reshape(shape).to(dtype=torch.int16)


class TurboQuantDynamicCache(Cache):
    """Dynamic cache that stores TurboQuant-compressed K/V segments.

    The cache keeps only packed quantization indices plus per-vector norms for
    bit-widths below 16. During `update()` it materializes dequantized K/V
    tensors just long enough for the current attention call. This is still a
    Python-level reproduction cache, not an optimized kernel implementation,
    but it no longer stores full-precision K/V tensors internally.
    """

    def __init__(self, config: KVQuantConfig | None = None) -> None:
        super().__init__()
        self.config = config or KVQuantConfig()
        self._seen_tokens = 0
        self.key_cache: list[list[CacheSegment]] = []
        self.value_cache: list[list[CacheSegment]] = []
        self._seq_lengths: list[int] = []
        self._quantizers: dict[
            tuple[str, str, int, torch.dtype, int, int, str, int],
            TurboQuantMSE | TurboQuantProd | TurboQuantBlockMSE | TurboQuantLearnedBlockMSE,
        ] = {}
        self._materialized_key_cache: list[torch.Tensor | None] = []
        self._materialized_value_cache: list[torch.Tensor | None] = []

    def __len__(self) -> int:
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield self[layer_idx]

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self.key_cache):
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")
        return self._materialize_layer(layer_idx)

    def _ensure_layer(self, layer_idx: int) -> None:
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append([])
            self.value_cache.append([])
            self._seq_lengths.append(0)
            self._materialized_key_cache.append(None)
            self._materialized_value_cache.append(None)

    @staticmethod
    def _srht_rotation_matrix(
        dimension: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
    ) -> torch.Tensor:
        transform = hadamard_orthogonal(dimension, device=device, dtype=dtype, seed=seed)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + 31_337)
        permutation = torch.randperm(dimension, generator=gen, device=device)
        return transform.index_select(0, permutation)

    def _get_quantizer(
        self,
        *,
        quantizer_kind: str,
        name: str,
        dimension: int,
        bits: int,
        layer_idx: int,
        device: torch.device,
        dtype: torch.dtype,
        rotation_bank_index: int = 0,
    ) -> TurboQuantMSE | TurboQuantProd | TurboQuantBlockMSE | TurboQuantLearnedBlockMSE:
        device_index = device.index if device.index is not None else -1
        is_rotation_bank = quantizer_kind in {
            "rotation_bank_mse",
            "attention_rotation_bank_mse",
            "segment_rotation_bank_mse",
        }
        configured_rotation_idx = self._configured_rotation_index(layer_idx, name)
        effective_rotation_idx = rotation_bank_index if is_rotation_bank else configured_rotation_idx
        head_rotation_idx = self._head_rotation_index(name)
        key_name = f"{name}:rot{effective_rotation_idx}" if (is_rotation_bank or effective_rotation_idx) else name
        key = (quantizer_kind, key_name, dimension, dtype, bits, layer_idx, device.type, device_index)
        quantizer = self._quantizers.get(key)
        if quantizer is None:
            side_seed_offset = 0 if name == "key" or name.startswith("key_head") else 500
            seed = (
                self.config.seed
                + 1_000 * layer_idx
                + side_seed_offset
                + self.config.rotation_bank_seed_stride * effective_rotation_idx
                + self.config.rotation_bank_seed_stride * head_rotation_idx
            )
            if quantizer_kind == "prod":
                quantizer = TurboQuantProd(
                    dimension,
                    bits,
                    seed=seed,
                    device=device,
                    dtype=dtype,
                    codebook_grid_size=self.config.codebook_grid_size,
                )
            elif quantizer_kind == "mse_block2":
                quantizer = TurboQuantBlockMSE(
                    dimension,
                    bits,
                    block_size=2,
                    seed=seed,
                    device=device,
                    dtype=dtype,
                    codebook_grid_size=self.config.codebook_grid_size,
                )
            elif quantizer_kind in {"learned_mse_block2", "learned_unit_mse_block2"}:
                quantizer = TurboQuantLearnedBlockMSE(
                    dimension,
                    bits,
                    block_size=2,
                    seed=seed,
                    device=device,
                    dtype=dtype,
                    project_unit_norm=quantizer_kind == "learned_unit_mse_block2",
                )
            elif quantizer_kind == "hadamard_mse":
                transform = hadamard_orthogonal(dimension, device=device, dtype=dtype, seed=seed)
                quantizer = TurboQuantMSE(
                    dimension,
                    bits,
                    seed=seed,
                    device=device,
                    dtype=dtype,
                    codebook_grid_size=self.config.codebook_grid_size,
                    transform_matrix=transform,
                    inverse_transform_matrix=transform.T,
                )
            elif quantizer_kind == "srht_mse":
                transform = self._srht_rotation_matrix(dimension, device=device, dtype=dtype, seed=seed)
                quantizer = TurboQuantMSE(
                    dimension,
                    bits,
                    seed=seed,
                    device=device,
                    dtype=dtype,
                    codebook_grid_size=self.config.codebook_grid_size,
                    transform_matrix=transform,
                    inverse_transform_matrix=transform.T,
                )
            elif quantizer_kind == "calibrated_rotation_mse":
                transform = self._calibrated_rotation_matrix(layer_idx, name, dimension, device, dtype)
                quantizer = TurboQuantMSE(
                    dimension,
                    bits,
                    seed=seed,
                    device=device,
                    dtype=dtype,
                    codebook_grid_size=self.config.codebook_grid_size,
                    transform_matrix=transform,
                    inverse_transform_matrix=transform.T,
                )
            else:
                base_quantizer_kind = "mse" if is_rotation_bank else quantizer_kind
                quantizer = TurboQuantMSE(
                    dimension,
                    bits,
                    seed=seed,
                    device=device,
                    dtype=dtype,
                    codebook_grid_size=self.config.codebook_grid_size,
                    project_unit_norm=base_quantizer_kind in {"unit_mse", "lsq_unit_mse"},
                    reconstruction_scale=(
                        "lsq"
                        if base_quantizer_kind in {"lsq_mse", "lsq_unit_mse"}
                        else "norm_gain"
                        if base_quantizer_kind == "gain_mse"
                        else "norm_gain"
                        if base_quantizer_kind == "regular_gain_mse"
                        else "half_gain"
                        if base_quantizer_kind == "regular_half_gain_mse"
                        else "clipped_gain"
                        if base_quantizer_kind == "regular_clipped_gain_mse"
                        else "clipped_gain"
                        if base_quantizer_kind == "clipped_gain_mse"
                        else "selected_gain"
                        if base_quantizer_kind == "selected_gain_mse"
                        else "norm_gain"
                        if base_quantizer_kind == "lowbit_gain_mse" and bits <= 2
                        else "half_gain"
                        if base_quantizer_kind == "lowbit_half_gain_mse" and bits <= 2
                        else "clipped_gain"
                        if base_quantizer_kind == "lowbit_clipped_gain_mse" and bits <= 2
                        else "selected_gain"
                        if base_quantizer_kind == "lowbit_selected_gain_mse" and bits <= 2
                        else "dot_gain"
                        if base_quantizer_kind == "dot_gain_mse"
                        else "norm"
                    ),
                )
            self._quantizers[key] = quantizer
        return quantizer

    def _quantize_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: int,
        name: str,
        quantizer_kind: str,
        layer_idx: int,
        query_states: torch.Tensor | None = None,
        key_states_for_attention: torch.Tensor | None = None,
        scaling: float = 1.0,
    ) -> CacheSegment:
        if bits >= 16:
            return RawTensorSegment(tensor=states.detach())
        if quantizer_kind == "distortion_regime_mse":
            return self._quantize_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind="gain_mse" if bits <= 2 else "learned_unit_mse_block2",
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "gain_unit_regime_mse":
            return self._quantize_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind="gain_mse" if bits <= 2 else "unit_mse",
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "rate_regime_mse":
            return self._quantize_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind="gain_mse" if bits <= 2 else "learned_unit_mse_block2",
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "hadamard_rate_regime_mse":
            return self._quantize_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind=(
                    "margin_vector_outlier_hadamard_mse"
                    if bits <= 2 and name == "value"
                    else "mse"
                    if bits <= 2
                    else "learned_unit_mse_block2"
                ),
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "rate_hadamard_value_mse":
            if name == "value" and bits <= 2:
                target_quantizer = "margin_vector_outlier_hadamard_mse"
            elif name == "value" and layer_idx >= 16:
                target_quantizer = "outlier_hadamard_mse"
            else:
                target_quantizer = "mse"
            return self._quantize_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind=target_quantizer,
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "rms_rotation_mse":
            if states.shape[-2] <= 1:
                return self._quantize_to_segment(
                    states,
                    bits=bits,
                    name=f"{name}_rms_base",
                    quantizer_kind="mse",
                    layer_idx=layer_idx,
                )
            rms = states.detach().to(dtype=torch.float32).square().mean(dim=tuple(range(states.ndim - 1))).sqrt()
            rms = rms.clamp_min(torch.finfo(torch.float32).eps)
            geo = torch.exp(torch.log(rms).mean()).clamp_min(torch.finfo(torch.float32).eps)
            scales = (rms / geo).to(device=states.device, dtype=states.dtype)
            conditioned = states / scales
            residual_segment = self._quantize_to_segment(
                conditioned,
                bits=bits,
                name=f"{name}_rms",
                quantizer_kind="mse",
                layer_idx=layer_idx,
            )
            return RmsScaledSegment(
                scales=scales.detach(),
                residual=residual_segment,
                shape=tuple(states.shape),
                dtype=states.dtype,
                bits=bits,
            )
        if quantizer_kind == "head_rotation_mse":
            return self._quantize_head_rotation_effective_to_segment(
                states,
                bits=float(bits),
                name=name,
                layer_idx=layer_idx,
            )
        if quantizer_kind == "centered_mse":
            center = states.detach().mean(dim=-2, keepdim=True)
            residual = states - center
            residual_segment = self._quantize_to_segment(
                residual,
                bits=bits,
                name=f"{name}_centered",
                quantizer_kind="mse",
                layer_idx=layer_idx,
            )
            return CenteredSegment(
                center=center,
                residual=residual_segment,
                shape=tuple(states.shape),
                dtype=states.dtype,
                bits=bits,
            )
        if quantizer_kind == "attention_scale_mse":
            return self._quantize_attention_scale_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                query_states=query_states,
                scaling=scaling,
            )
        if quantizer_kind in {"uniform_token", "uniform_channel"}:
            return self._quantize_uniform_to_segment(
                states,
                bits=bits,
                granularity="channel" if quantizer_kind == "uniform_channel" else "token",
            )
        if quantizer_kind == "auto_mse":
            return self._quantize_auto_mse_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
            )
        if quantizer_kind in {"rotation_bank_mse", "attention_rotation_bank_mse"}:
            return self._quantize_rotation_bank_mse_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                attention_aware=quantizer_kind == "attention_rotation_bank_mse",
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "outlier_hadamard_mse":
            return self._quantize_outlier_hadamard_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
            )
        if quantizer_kind == "attention_adaptive_outlier_hadamard_mse":
            return self._quantize_attention_adaptive_outlier_hadamard_effective_to_segment(
                states,
                bits=float(bits),
                name=name,
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "adaptive_outlier_hadamard_mse":
            return self._quantize_adaptive_outlier_hadamard_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
            )
        if quantizer_kind == "vector_adaptive_outlier_hadamard_mse":
            return self._quantize_vector_adaptive_outlier_hadamard_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
            )
        if quantizer_kind == "margin_vector_outlier_hadamard_mse":
            return self._quantize_vector_adaptive_outlier_hadamard_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                activation_margin=0.95,
            )
        original_shape = tuple(states.shape)
        dimension = original_shape[-1]
        flat = states.reshape(-1, dimension)
        quantizer = self._get_quantizer(
            quantizer_kind=quantizer_kind,
            name=name,
            dimension=dimension,
            bits=bits,
            layer_idx=layer_idx,
            device=states.device,
            dtype=states.dtype,
        )
        quantized = quantizer.quantize(flat)
        if isinstance(quantized, (MSEQuantized, BlockMSEQuantized)):
            pack_bits = (
                quantizer.block_bits
                if isinstance(quantizer, (TurboQuantBlockMSE, TurboQuantLearnedBlockMSE))
                else bits
            )
            packed = pack_indices(quantized.indices, pack_bits)
            return PackedMSESegment(
                packed_indices=packed,
                norms=quantized.norms.reshape(original_shape[:-1]).detach(),
                shape=original_shape,
                bits=bits,
                dtype=states.dtype,
                block_size=(
                    quantizer.block_size
                    if isinstance(quantizer, (TurboQuantBlockMSE, TurboQuantLearnedBlockMSE))
                    else 1
                ),
                quantizer_kind=quantizer_kind,
            )
        if not isinstance(quantized, ProdQuantized):
            raise TypeError(f"unsupported quantized payload: {type(quantized)!r}")
        mse_bits = bits - 1
        packed_mse = pack_indices(quantized.mse.indices, mse_bits)
        qjl_indices = (quantized.qjl > 0).to(torch.int16)
        packed_qjl = pack_indices(qjl_indices, 1)
        return PackedProdSegment(
            packed_mse_indices=packed_mse,
            mse_norms=quantized.mse.norms.reshape(original_shape[:-1]).detach(),
            packed_qjl_signs=packed_qjl,
            residual_norms=quantized.residual_norms.reshape(original_shape[:-1]).detach(),
            shape=original_shape,
            bits=bits,
            dtype=states.dtype,
        )

    def _quantize_auto_mse_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: int,
        name: str,
        layer_idx: int,
    ) -> CacheSegment:
        candidate_kinds = (
            "mse",
            "unit_mse",
            "selected_gain_mse",
            "learned_unit_mse_block2",
        )
        best_segment: CacheSegment | None = None
        best_error: float | None = None
        states_f = states.detach().to(dtype=torch.float32)
        for candidate_kind in candidate_kinds:
            segment = self._quantize_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind=candidate_kind,
                layer_idx=layer_idx,
            )
            decoded = self._decode_segment(segment, name=name, layer_idx=layer_idx).to(dtype=torch.float32)
            error = float(torch.mean((states_f - decoded) ** 2).item())
            if best_error is None or error < best_error:
                best_error = error
                best_segment = segment
        if best_segment is None:
            raise RuntimeError("auto_mse did not produce a candidate segment")
        return best_segment

    def _token_protected_raw_count(self, seq_len: int, *, target_bits: float, quant_bits: int) -> int:
        if seq_len <= 0:
            return 0
        if self.config.token_protection_target_ratio is not None:
            target_equiv_bits = 16.0 * self.config.token_protection_target_ratio
            fraction = (target_equiv_bits - quant_bits) / (16.0 - quant_bits)
        elif target_bits > quant_bits:
            fraction = (target_bits - quant_bits) / (16.0 - quant_bits)
        else:
            return 0
        count = int(floor(seq_len * fraction))
        if count <= 0 and seq_len > 0:
            count = 1
        return max(0, min(seq_len, count))

    def _estimate_token_protected_nbytes(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        raw_count: int,
        quant_bits: int,
        quantizer_kind: str,
    ) -> int:
        if quantizer_kind != "mse":
            raise ValueError("sink_recent_budget currently supports the mse quantizer")
        seq_len = shape[-2]
        quant_count = seq_len - raw_count
        element_size = torch.tensor([], dtype=dtype).element_size()
        prefix = _numel(shape[:-2])
        dimension = shape[-1]
        raw_nbytes = prefix * raw_count * dimension * element_size
        quant_values = prefix * quant_count * dimension
        quant_packed_nbytes = (quant_values * quant_bits + 7) // 8
        quant_norm_nbytes = prefix * quant_count * element_size
        index_nbytes = seq_len * torch.tensor([], dtype=torch.int16).element_size()
        return raw_nbytes + quant_packed_nbytes + quant_norm_nbytes + index_nbytes

    def _budget_matched_raw_count(
        self,
        states: torch.Tensor,
        *,
        target_nbytes: int,
        quant_bits: int,
        quantizer_kind: str,
    ) -> int:
        seq_len = states.shape[-2]
        low = 0
        high = seq_len
        while low <= high:
            mid = (low + high) // 2
            nbytes = self._estimate_token_protected_nbytes(
                tuple(states.shape),
                dtype=states.dtype,
                raw_count=mid,
                quant_bits=quant_bits,
                quantizer_kind=quantizer_kind,
            )
            if nbytes <= target_nbytes:
                low = mid + 1
            else:
                high = mid - 1
        return max(0, high)

    def _select_protected_token_indices(
        self,
        states: torch.Tensor,
        raw_count: int,
        *,
        token_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = states.shape[-2]
        if raw_count <= 0:
            return torch.empty(0, dtype=torch.long)
        if self.config.token_protection_policy == "attention_error_budget" and token_scores is not None and raw_count > 1:
            start_count = min(self.config.protected_start_tokens, raw_count - 1, seq_len)
        else:
            start_count = min(self.config.protected_start_tokens, raw_count, seq_len)
        start = torch.arange(start_count, dtype=torch.long)
        remaining_count = raw_count - start_count
        if remaining_count <= 0:
            return start
        if self.config.token_protection_policy == "sink_recent_budget":
            recent_start = max(start_count, seq_len - remaining_count)
            recent = torch.arange(recent_start, seq_len, dtype=torch.long)
            return torch.unique(torch.cat([start, recent]), sorted=True)
        if self.config.token_protection_policy == "norm_aware_budget":
            token_scores = torch.linalg.vector_norm(states.detach().to(dtype=torch.float32), dim=-1)
            if token_scores.ndim > 1:
                token_scores = token_scores.mean(dim=tuple(range(token_scores.ndim - 1)))
            token_scores = token_scores.cpu()
            if start_count > 0:
                token_scores[:start_count] = -torch.inf
            top_count = min(remaining_count, seq_len - start_count)
            if top_count <= 0:
                return start
            top = torch.topk(token_scores, k=top_count, largest=True, sorted=False).indices.to(dtype=torch.long)
            return torch.unique(torch.cat([start, top]), sorted=True)
        if self.config.token_protection_policy == "attention_error_budget":
            if token_scores is None:
                recent_start = max(start_count, seq_len - remaining_count)
                recent = torch.arange(recent_start, seq_len, dtype=torch.long)
                return torch.unique(torch.cat([start, recent]), sorted=True)
            token_scores = token_scores.detach().to(dtype=torch.float32, device="cpu").flatten()
            if token_scores.numel() != seq_len:
                raise ValueError(f"token_scores length mismatch: expected {seq_len}, got {token_scores.numel()}")
            if start_count > 0:
                token_scores[:start_count] = -torch.inf
            top_count = min(remaining_count, seq_len - start_count)
            if top_count <= 0:
                return start
            top = torch.topk(token_scores, k=top_count, largest=True, sorted=False).indices.to(dtype=torch.long)
            return torch.unique(torch.cat([start, top]), sorted=True)
        raise ValueError(f"unsupported token protection policy: {self.config.token_protection_policy}")

    def _grouped_attention_scores(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        *,
        scaling: float,
    ) -> torch.Tensor:
        if query_states.ndim != 4 or key_states.ndim != 4:
            raise ValueError("attention token scores require 4D query and key tensors")
        if query_states.shape[0] != key_states.shape[0]:
            raise ValueError("query_states and key_states batch dimensions must match")
        if query_states.shape[1] % key_states.shape[1] != 0:
            raise ValueError("query head count must be divisible by key head count")
        query_tokens = min(self.config.attention_error_query_tokens, query_states.shape[-2])
        if query_tokens <= 0:
            return torch.empty(0, device=key_states.device, dtype=torch.float32)
        query_focus = query_states[..., -query_tokens:, :]
        bsz, num_query_heads, _, head_dim = query_focus.shape
        num_key_heads = key_states.shape[1]
        num_groups = num_query_heads // num_key_heads
        query_grouped = query_focus.reshape(bsz, num_key_heads, num_groups, query_tokens, head_dim)
        scores = torch.einsum("bhgqd,bhsd->bhgqs", query_grouped.to(torch.float32), key_states.to(torch.float32))
        return scores * float(scaling)

    def _causal_attention_probs(self, scores: torch.Tensor, *, key_len: int) -> torch.Tensor:
        query_tokens = scores.shape[-2]
        if query_tokens == 1:
            return torch.softmax(scores, dim=-1)
        query_positions = torch.arange(query_tokens, device=scores.device)
        key_positions = torch.arange(key_len, device=scores.device)
        causal_mask = key_positions.view(1, 1, 1, 1, key_len) <= (
            (key_len - query_tokens + query_positions).view(1, 1, 1, query_tokens, 1)
        )
        masked_scores = scores.masked_fill(~causal_mask, float("-inf"))
        return torch.softmax(masked_scores, dim=-1)

    def _attention_quantization_error_scores(
        self,
        states: torch.Tensor,
        *,
        name: str,
        quantizer_kind: str,
        layer_idx: int,
        quant_bits: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        quantized_segment = self._quantize_to_segment(
            states,
            bits=quant_bits,
            name=f"{name}_score_probe",
            quantizer_kind=quantizer_kind,
            layer_idx=layer_idx,
        )
        states_hat = self._decode_segment(quantized_segment, name=f"{name}_score_probe", layer_idx=layer_idx)
        key_scores = self._grouped_attention_scores(query_states, key_states, scaling=scaling)
        probs = self._causal_attention_probs(key_scores, key_len=key_states.shape[-2])
        if name == "key":
            err_scores = self._grouped_attention_scores(query_states, key_states - states_hat, scaling=scaling).abs()
            scores = (probs * err_scores).mean(dim=(0, 1, 2, 3))
        elif name == "value":
            value_err = torch.linalg.vector_norm((states - states_hat).detach().to(torch.float32), dim=-1)
            scores = probs.mean(dim=(2, 3)) * value_err
            scores = scores.mean(dim=(0, 1))
        else:
            raise ValueError(f"unsupported attention error score target: {name}")
        return scores.detach()

    def _quantize_token_protected_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        quantizer_kind: str,
        layer_idx: int,
        token_scores: torch.Tensor | None = None,
    ) -> CacheSegment:
        quant_bits = self.config.token_quant_bits or max(1, floor(bits))
        if quant_bits >= bits or bits >= 16:
            return self._quantize_effective_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind=quantizer_kind,
                layer_idx=layer_idx,
                allow_token_protection=False,
            )
        seq_len = states.shape[-2]
        if self.config.token_protection_target_ratio is None:
            baseline_segment = self._quantize_effective_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind=quantizer_kind,
                layer_idx=layer_idx,
                allow_token_protection=False,
            )
            raw_count = self._budget_matched_raw_count(
                states,
                target_nbytes=baseline_segment.nbytes,
                quant_bits=quant_bits,
                quantizer_kind=quantizer_kind,
            )
        else:
            raw_count = self._token_protected_raw_count(seq_len, target_bits=bits, quant_bits=quant_bits)
        if raw_count <= 0:
            return self._quantize_to_segment(
                states,
                bits=quant_bits,
                name=name,
                quantizer_kind=quantizer_kind,
                layer_idx=layer_idx,
            )
        raw_indices = self._select_protected_token_indices(states, raw_count, token_scores=token_scores).to(device=states.device)
        all_indices = torch.arange(seq_len, device=states.device, dtype=torch.long)
        quant_mask = torch.ones(seq_len, device=states.device, dtype=torch.bool)
        quant_mask[raw_indices] = False
        quantized_indices = all_indices[quant_mask]
        raw_tensor = states.index_select(-2, raw_indices).detach()
        quantized_states = states.index_select(-2, quantized_indices)
        quantized_segment = self._quantize_to_segment(
            quantized_states,
            bits=quant_bits,
            name=name,
            quantizer_kind=quantizer_kind,
            layer_idx=layer_idx,
        )
        return TokenProtectedSegment(
            raw_tensor=raw_tensor,
            quantized=quantized_segment,
            raw_indices=raw_indices.to(dtype=torch.int16),
            quantized_indices=quantized_indices.to(dtype=torch.int16),
            shape=tuple(states.shape),
            dtype=states.dtype,
            target_bits=float(bits),
            quant_bits=quant_bits,
            policy=self.config.token_protection_policy,
        )

    def _quantize_uniform_to_segment(self, states: torch.Tensor, *, bits: int, granularity: str) -> UniformAffineSegment:
        if bits <= 0 or bits > 8:
            raise ValueError("uniform quantization supports 1..8 bits")
        if granularity not in {"token", "channel"}:
            raise ValueError(f"unsupported uniform granularity: {granularity}")
        qmax = (1 << (bits - 1)) - 1
        if qmax < 1:
            qmax = 1
        reduce_dim = -2 if granularity == "channel" else -1
        max_abs = states.detach().to(dtype=torch.float32).abs().amax(dim=reduce_dim, keepdim=True)
        scales = (max_abs / qmax).clamp_min(torch.finfo(torch.float32).eps)
        quantized = torch.round(states.detach().to(dtype=torch.float32) / scales).clamp(-qmax, qmax).to(dtype=torch.int16)
        encoded = quantized + qmax
        return UniformAffineSegment(
            packed_values=pack_indices(encoded, bits),
            scales=scales.to(dtype=states.dtype).detach(),
            shape=tuple(states.shape),
            bits=bits,
            dtype=states.dtype,
            granularity=granularity,
        )

    def _resolve_effective_bit_allocation(self, bits: float, dimension: int) -> tuple[int, int, int]:
        """Return `(regular_bits, outlier_bits, outlier_count)` for one head dimension."""

        if bits >= 16:
            return 16, 16, 0
        lower_bits = max(1, floor(bits))
        upper_bits = ceil(bits)
        if lower_bits == upper_bits:
            return lower_bits, upper_bits, 0
        if upper_bits > 8:
            raise ValueError("effective bit allocation only supports packed bits up to 8")

        fraction = bits - lower_bits
        if self.config.effective_bit_allocation == "quarter_high2":
            outlier_bits = lower_bits + 2
            if outlier_bits > 8:
                raise ValueError("quarter_high2 allocation only supports packed bits up to 8")
            outlier_count = int(round(fraction * dimension / 2.0))
        else:
            outlier_bits = upper_bits
            outlier_count = int(round(fraction * dimension))
        outlier_count = max(0, min(dimension, outlier_count))
        return lower_bits, outlier_bits, outlier_count

    def _score_error_gain_channels(
        self,
        states: torch.Tensor,
        *,
        regular_bits: int,
        outlier_bits: int,
        name: str,
        quantizer_kind: str,
        layer_idx: int,
    ) -> torch.Tensor:
        regular_segment = self._quantize_to_segment(
            states,
            bits=regular_bits,
            name=f"{name}_regular",
            quantizer_kind=quantizer_kind,
            layer_idx=layer_idx,
        )
        outlier_segment = self._quantize_to_segment(
            states,
            bits=outlier_bits,
            name=f"{name}_outlier",
            quantizer_kind=quantizer_kind,
            layer_idx=layer_idx,
        )
        regular_hat = self._decode_segment(regular_segment, name=f"{name}_regular", layer_idx=layer_idx)
        outlier_hat = self._decode_segment(outlier_segment, name=f"{name}_outlier", layer_idx=layer_idx)
        states_f = states.detach().to(dtype=torch.float32)
        regular_error = (states_f - regular_hat.to(dtype=torch.float32)).square()
        outlier_error = (states_f - outlier_hat.to(dtype=torch.float32)).square()
        gain = regular_error - outlier_error
        return gain.mean(dim=tuple(range(gain.ndim - 1)))

    def _score_attention_error_gain_channels(
        self,
        states: torch.Tensor,
        *,
        regular_bits: int,
        outlier_bits: int,
        name: str,
        quantizer_kind: str,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states_for_attention: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        if states.ndim != 4 or query_states.ndim != 4 or key_states_for_attention.ndim != 4:
            return self._score_error_gain_channels(
                states,
                regular_bits=regular_bits,
                outlier_bits=outlier_bits,
                name=name,
                quantizer_kind=quantizer_kind,
                layer_idx=layer_idx,
            )
        regular_segment = self._quantize_to_segment(
            states,
            bits=regular_bits,
            name=f"{name}_regular_attention_probe",
            quantizer_kind=quantizer_kind,
            layer_idx=layer_idx,
        )
        outlier_segment = self._quantize_to_segment(
            states,
            bits=outlier_bits,
            name=f"{name}_outlier_attention_probe",
            quantizer_kind=quantizer_kind,
            layer_idx=layer_idx,
        )
        regular_hat = self._decode_segment(regular_segment, name=f"{name}_regular_attention_probe", layer_idx=layer_idx)
        outlier_hat = self._decode_segment(outlier_segment, name=f"{name}_outlier_attention_probe", layer_idx=layer_idx)
        states_f = states.detach().to(dtype=torch.float32)
        regular_error = (states_f - regular_hat.to(dtype=torch.float32)).square()
        outlier_error = (states_f - outlier_hat.to(dtype=torch.float32)).square()
        error_gain = (regular_error - outlier_error).clamp_min(0.0)

        key_scores = self._grouped_attention_scores(query_states, key_states_for_attention, scaling=scaling)
        probs = self._causal_attention_probs(key_scores, key_len=key_states_for_attention.shape[-2]).detach()
        if name == "key":
            query_tokens = min(self.config.attention_error_query_tokens, query_states.shape[-2])
            query_focus = query_states[..., -query_tokens:, :]
            bsz, num_query_heads, _, head_dim = query_focus.shape
            num_key_heads = key_states_for_attention.shape[1]
            num_groups = num_query_heads // num_key_heads
            query_grouped = query_focus.reshape(bsz, num_key_heads, num_groups, query_tokens, head_dim)
            query_sensitivity = query_grouped.detach().to(dtype=torch.float32).square().mean(dim=(2, 3))
            token_weight = probs.mean(dim=(2, 3))
            weighted_gain = (token_weight.unsqueeze(-1) * error_gain).mean(dim=-2)
            scores = (weighted_gain * query_sensitivity).mean(dim=tuple(range(weighted_gain.ndim - 1)))
        elif name == "value":
            token_weight = probs.square().mean(dim=(2, 3))
            scores = (token_weight.unsqueeze(-1) * error_gain).mean(dim=tuple(range(error_gain.ndim - 1)))
        else:
            raise ValueError(f"unsupported attention-error outlier target: {name}")
        return scores

    def _static_channel_scores(self, *, name: str, layer_idx: int, dimension: int, device: torch.device) -> torch.Tensor:
        schedules = self.config.layer_key_channel_scores if name == "key" else self.config.layer_value_channel_scores
        if schedules is None:
            raise ValueError(f"outlier_policy=static_score requires layer_{name}_channel_scores")
        if layer_idx >= len(schedules):
            raise ValueError(f"layer_{name}_channel_scores has {len(schedules)} entries, missing layer {layer_idx}")
        scores = torch.tensor(schedules[layer_idx], device=device, dtype=torch.float32)
        if scores.numel() != dimension:
            raise ValueError(
                f"layer {layer_idx} {name} channel score dimension mismatch: expected {dimension}, got {scores.numel()}"
            )
        return scores

    def _outlier_policy_for(self, name: str) -> str:
        if name == "key" and self.config.key_outlier_policy is not None:
            return self.config.key_outlier_policy
        if name == "value" and self.config.value_outlier_policy is not None:
            return self.config.value_outlier_policy
        return self.config.outlier_policy

    def _select_outlier_channels(
        self,
        states: torch.Tensor,
        outlier_count: int,
        *,
        regular_bits: int,
        outlier_bits: int,
        name: str,
        quantizer_kind: str,
        layer_idx: int,
        query_states: torch.Tensor | None = None,
        key_states_for_attention: torch.Tensor | None = None,
        scaling: float = 1.0,
    ) -> torch.Tensor:
        if outlier_count <= 0:
            return torch.empty(0, device=states.device, dtype=torch.long)
        policy = self._outlier_policy_for(name)
        if policy == "dynamic_absmean":
            scores = states.detach().abs().to(dtype=torch.float32).mean(dim=tuple(range(states.ndim - 1)))
        elif policy in {"error_gain", "joint_error_gain"}:
            scores = self._score_error_gain_channels(
                states,
                regular_bits=regular_bits,
                outlier_bits=outlier_bits,
                name=name,
                quantizer_kind=quantizer_kind,
                layer_idx=layer_idx,
            )
        elif policy in {"attention_error_gain", "joint_attention_error_gain"}:
            if query_states is None or key_states_for_attention is None:
                scores = self._score_error_gain_channels(
                    states,
                    regular_bits=regular_bits,
                    outlier_bits=outlier_bits,
                    name=name,
                    quantizer_kind=quantizer_kind,
                    layer_idx=layer_idx,
                )
            else:
                scores = self._score_attention_error_gain_channels(
                    states,
                    regular_bits=regular_bits,
                    outlier_bits=outlier_bits,
                    name=name,
                    quantizer_kind=quantizer_kind,
                    layer_idx=layer_idx,
                    query_states=query_states,
                    key_states_for_attention=key_states_for_attention,
                    scaling=scaling,
                )
        elif policy == "sensitivity_error_gain":
            error_gain = self._score_error_gain_channels(
                states,
                regular_bits=regular_bits,
                outlier_bits=outlier_bits,
                name=name,
                quantizer_kind=quantizer_kind,
                layer_idx=layer_idx,
            )
            sensitivity = self._static_channel_scores(
                name=name,
                layer_idx=layer_idx,
                dimension=states.shape[-1],
                device=states.device,
            )
            sensitivity = sensitivity.clamp_min(0)
            if self.config.sensitivity_score_power != 1.0:
                sensitivity = sensitivity.pow(float(self.config.sensitivity_score_power))
            scores = error_gain.clamp_min(0) * sensitivity
        elif policy == "static_score":
            scores = self._static_channel_scores(name=name, layer_idx=layer_idx, dimension=states.shape[-1], device=states.device)
        else:
            raise ValueError(f"unsupported outlier policy: {policy}")
        return torch.topk(scores, k=outlier_count, largest=True, sorted=True).indices.sort().values

    @staticmethod
    def _top_indices_from_scores(scores: torch.Tensor, count: int) -> torch.Tensor:
        if count <= 0:
            return torch.empty(0, device=scores.device, dtype=torch.long)
        count = min(count, int(scores.numel()))
        return torch.topk(scores, k=count, largest=True, sorted=True).indices.sort().values

    @staticmethod
    def _split_joint_channel_budget(
        combined_scores: torch.Tensor,
        *,
        dimension: int,
        total_budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_budget = max(0, min(total_budget, 2 * dimension))
        if total_budget <= 0:
            empty = torch.empty(0, device=combined_scores.device, dtype=torch.long)
            return empty, empty
        max_per_side = dimension if total_budget == 2 * dimension else max(1, dimension - 1)
        min_per_side = 1 if total_budget >= 2 and dimension > 1 else 0
        sorted_indices = torch.argsort(combined_scores, descending=True)
        selected: list[int] = []
        key_count = 0
        value_count = 0
        for raw_idx in sorted_indices.tolist():
            is_key = raw_idx < dimension
            if is_key and key_count >= max_per_side:
                continue
            if not is_key and value_count >= max_per_side:
                continue
            remaining_after = total_budget - len(selected) - 1
            if is_key:
                if value_count + remaining_after < min_per_side:
                    continue
                key_count += 1
            else:
                if key_count + remaining_after < min_per_side:
                    continue
                value_count += 1
            selected.append(raw_idx)
            if len(selected) == total_budget:
                break
        selected_tensor = torch.tensor(selected, device=combined_scores.device, dtype=torch.long)
        key_indices = selected_tensor[selected_tensor < dimension].sort().values
        value_indices = (selected_tensor[selected_tensor >= dimension] - dimension).sort().values
        return key_indices, value_indices

    def _build_outlier_segment_from_indices(
        self,
        states: torch.Tensor,
        outlier_indices: torch.Tensor,
        *,
        bits: float,
        name: str,
        quantizer_kind: str,
        layer_idx: int,
        regular_bits: int,
        outlier_bits: int,
        policy: str,
    ) -> CacheSegment:
        original_shape = tuple(states.shape)
        all_indices = torch.arange(states.shape[-1], device=states.device, dtype=torch.long)
        outlier_indices = outlier_indices.to(device=states.device, dtype=torch.long).sort().values
        regular_mask = torch.ones(states.shape[-1], device=states.device, dtype=torch.bool)
        regular_mask[outlier_indices] = False
        regular_indices = all_indices[regular_mask]

        regular_states = states.index_select(-1, regular_indices)
        outlier_states = states.index_select(-1, outlier_indices)
        regular_segment = self._quantize_to_segment(
            regular_states,
            bits=regular_bits,
            name=f"{name}_regular",
            quantizer_kind=quantizer_kind,
            layer_idx=layer_idx,
        )
        outlier_segment = self._quantize_to_segment(
            outlier_states,
            bits=outlier_bits,
            name=f"{name}_outlier",
            quantizer_kind=quantizer_kind,
            layer_idx=layer_idx,
        )
        if isinstance(regular_segment, RawTensorSegment) or isinstance(outlier_segment, RawTensorSegment):
            raise RuntimeError("outlier effective-bit segments must use packed subsegments")

        return OutlierMSESegment(
            regular=regular_segment,
            outlier=outlier_segment,
            regular_indices=regular_indices.to(dtype=torch.int16),
            outlier_indices=outlier_indices.to(dtype=torch.int16),
            shape=original_shape,
            dtype=states.dtype,
            requested_bits=float(bits),
            policy=policy,
        )

    def _score_joint_outlier_channels(
        self,
        states: torch.Tensor,
        *,
        regular_bits: int,
        outlier_bits: int,
        name: str,
        quantizer_kind: str,
        layer_idx: int,
        query_states: torch.Tensor | None,
        key_states_for_attention: torch.Tensor | None,
        scaling: float,
        policy: str,
    ) -> torch.Tensor:
        if policy == "joint_attention_error_gain":
            if query_states is not None and key_states_for_attention is not None:
                return self._score_attention_error_gain_channels(
                    states,
                    regular_bits=regular_bits,
                    outlier_bits=outlier_bits,
                    name=name,
                    quantizer_kind=quantizer_kind,
                    layer_idx=layer_idx,
                    query_states=query_states,
                    key_states_for_attention=key_states_for_attention,
                    scaling=scaling,
                )
        return self._score_error_gain_channels(
            states,
            regular_bits=regular_bits,
            outlier_bits=outlier_bits,
            name=name,
            quantizer_kind=quantizer_kind,
            layer_idx=layer_idx,
        )

    def _quantize_joint_effective_to_segments(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        key_bits: float,
        value_bits: float,
        key_quantizer: str,
        value_quantizer: str,
        layer_idx: int,
        query_states: torch.Tensor | None,
        scaling: float,
        policy: str,
    ) -> tuple[CacheSegment, CacheSegment]:
        key_regular_bits, key_outlier_bits, key_outlier_budget = self._resolve_effective_bit_allocation(
            key_bits, key_states.shape[-1]
        )
        value_regular_bits, value_outlier_bits, value_outlier_budget = self._resolve_effective_bit_allocation(
            value_bits, value_states.shape[-1]
        )
        if (
            key_outlier_budget == 0
            or value_outlier_budget == 0
            or key_regular_bits != value_regular_bits
            or key_outlier_bits != value_outlier_bits
            or key_bits != value_bits
            or key_states.shape[-1] != value_states.shape[-1]
        ):
            return (
                self._quantize_effective_to_segment(
                    key_states,
                    bits=key_bits,
                    name="key",
                    quantizer_kind=key_quantizer,
                    layer_idx=layer_idx,
                    query_states=query_states,
                    key_states_for_attention=key_states,
                    scaling=scaling,
                ),
                self._quantize_effective_to_segment(
                    value_states,
                    bits=value_bits,
                    name="value",
                    quantizer_kind=value_quantizer,
                    layer_idx=layer_idx,
                    query_states=query_states,
                    key_states_for_attention=key_states,
                    scaling=scaling,
                ),
            )

        key_scores = self._score_joint_outlier_channels(
            key_states,
            regular_bits=key_regular_bits,
            outlier_bits=key_outlier_bits,
            name="key",
            quantizer_kind=key_quantizer,
            layer_idx=layer_idx,
            query_states=query_states,
            key_states_for_attention=key_states,
            scaling=scaling,
            policy=policy,
        )
        value_scores = self._score_joint_outlier_channels(
            value_states,
            regular_bits=value_regular_bits,
            outlier_bits=value_outlier_bits,
            name="value",
            quantizer_kind=value_quantizer,
            layer_idx=layer_idx,
            query_states=query_states,
            key_states_for_attention=key_states,
            scaling=scaling,
            policy=policy,
        )
        key_scores = key_scores.clamp_min(0)
        value_scores = value_scores.clamp_min(0)
        key_scale = key_scores.mean().clamp_min(torch.finfo(torch.float32).eps)
        value_scale = value_scores.mean().clamp_min(torch.finfo(torch.float32).eps)
        combined = torch.cat([key_scores / key_scale, value_scores / value_scale])
        total_budget = key_outlier_budget + value_outlier_budget
        dimension = key_states.shape[-1]
        key_indices, value_indices = self._split_joint_channel_budget(
            combined,
            dimension=dimension,
            total_budget=total_budget,
        )

        key_segment = self._build_outlier_segment_from_indices(
            key_states,
            key_indices,
            bits=key_bits,
            name="key",
            quantizer_kind=key_quantizer,
            layer_idx=layer_idx,
            regular_bits=key_regular_bits,
            outlier_bits=key_outlier_bits,
            policy=policy,
        )
        value_segment = self._build_outlier_segment_from_indices(
            value_states,
            value_indices,
            bits=value_bits,
            name="value",
            quantizer_kind=value_quantizer,
            layer_idx=layer_idx,
            regular_bits=value_regular_bits,
            outlier_bits=value_outlier_bits,
            policy=policy,
        )
        return key_segment, value_segment

    def _quantize_head_adaptive_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        quantizer_kind: str,
        layer_idx: int,
        regular_bits: int,
        outlier_bits: int,
        outlier_count: int,
    ) -> CacheSegment:
        if states.ndim < 4:
            return self._quantize_to_segment(
                states,
                bits=regular_bits,
                name=name,
                quantizer_kind=quantizer_kind,
                layer_idx=layer_idx,
            )
        head_segments = []
        num_heads = states.shape[-3]
        for head_idx in range(num_heads):
            head_states = states.index_select(-3, torch.tensor([head_idx], device=states.device)).squeeze(-3)
            original_policy = self._outlier_policy_for(name)
            if original_policy == "head_dynamic_absmean":
                scores = head_states.detach().abs().to(dtype=torch.float32).mean(dim=tuple(range(head_states.ndim - 1)))
            elif original_policy == "head_error_gain":
                scores = self._score_error_gain_channels(
                    head_states,
                    regular_bits=regular_bits,
                    outlier_bits=outlier_bits,
                    name=name,
                    quantizer_kind=quantizer_kind,
                    layer_idx=layer_idx,
                )
            else:
                raise ValueError(f"unsupported head-adaptive outlier policy: {original_policy}")
            outlier_indices = torch.topk(scores, k=outlier_count, largest=True, sorted=True).indices.sort().values
            all_indices = torch.arange(head_states.shape[-1], device=states.device, dtype=torch.long)
            regular_mask = torch.ones(head_states.shape[-1], device=states.device, dtype=torch.bool)
            regular_mask[outlier_indices] = False
            regular_indices = all_indices[regular_mask]
            regular_states = head_states.index_select(-1, regular_indices)
            outlier_states = head_states.index_select(-1, outlier_indices)
            regular_segment = self._quantize_to_segment(
                regular_states,
                bits=regular_bits,
                name=f"{name}_regular",
                quantizer_kind=quantizer_kind,
                layer_idx=layer_idx,
            )
            outlier_segment = self._quantize_to_segment(
                outlier_states,
                bits=outlier_bits,
                name=f"{name}_outlier",
                quantizer_kind=quantizer_kind,
                layer_idx=layer_idx,
            )
            if isinstance(regular_segment, RawTensorSegment) or isinstance(outlier_segment, RawTensorSegment):
                raise RuntimeError("head-adaptive outlier segments must use packed subsegments")
            head_segments.append(
                OutlierMSESegment(
                    regular=regular_segment,
                    outlier=outlier_segment,
                    regular_indices=regular_indices.to(dtype=torch.int16),
                    outlier_indices=outlier_indices.to(dtype=torch.int16),
                    shape=tuple(head_states.shape),
                    dtype=states.dtype,
                    requested_bits=float(bits),
                    policy=self._outlier_policy_for(name),
                )
            )
        return HeadAdaptiveOutlierMSESegment(
            head_segments=tuple(head_segments),
            shape=tuple(states.shape),
            dtype=states.dtype,
            requested_bits=float(bits),
            policy=self._outlier_policy_for(name),
        )

    def _quantize_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        quantizer_kind: str,
        layer_idx: int,
        allow_token_protection: bool = True,
        token_scores: torch.Tensor | None = None,
        query_states: torch.Tensor | None = None,
        key_states_for_attention: torch.Tensor | None = None,
        scaling: float = 1.0,
    ) -> CacheSegment:
        if quantizer_kind == "rate_regime_mse":
            return self._quantize_effective_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind="gain_mse" if bits < 3.0 else "learned_unit_mse_block2",
                layer_idx=layer_idx,
                allow_token_protection=allow_token_protection,
                token_scores=token_scores,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "hadamard_rate_regime_mse":
            if bits < 3.0:
                target_quantizer = "margin_vector_outlier_hadamard_mse" if name == "value" else "mse"
            else:
                target_quantizer = "learned_unit_mse_block2"
            return self._quantize_effective_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind=target_quantizer,
                layer_idx=layer_idx,
                allow_token_protection=allow_token_protection,
                token_scores=token_scores,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "rate_hadamard_value_mse":
            if name == "value" and bits < 3.0:
                target_quantizer = "margin_vector_outlier_hadamard_mse"
            elif name == "value" and layer_idx >= 16:
                target_quantizer = "outlier_hadamard_mse"
            else:
                target_quantizer = "mse"
            return self._quantize_effective_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind=target_quantizer,
                layer_idx=layer_idx,
                allow_token_protection=allow_token_protection,
                token_scores=token_scores,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        target_enabled = self.config.token_protection_targets == "both" or self.config.token_protection_targets == name
        if (
            allow_token_protection
            and target_enabled
            and self.config.token_protection_policy in {"sink_recent_budget", "norm_aware_budget", "attention_error_budget"}
        ):
            return self._quantize_token_protected_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind=quantizer_kind,
                layer_idx=layer_idx,
                token_scores=token_scores,
            )
        if quantizer_kind == "attention_auto_mse":
            return self._quantize_attention_auto_mse_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "attention_rotation_bank_mse":
            if query_states is not None and key_states_for_attention is not None:
                return self._quantize_attention_rotation_bank_effective_to_segment(
                    states,
                    bits=bits,
                    name=name,
                    layer_idx=layer_idx,
                    query_states=query_states,
                    key_states_for_attention=key_states_for_attention,
                    scaling=scaling,
            )
            quantizer_kind = "rotation_bank_mse"
        if quantizer_kind == "segment_rotation_bank_mse":
            return self._quantize_segment_rotation_bank_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
            )
        if quantizer_kind == "head_hadamard_mse":
            return self._quantize_head_hadamard_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
            )
        if quantizer_kind == "attention_outlier_hadamard_mse":
            return self._quantize_attention_outlier_hadamard_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "attention_adaptive_outlier_hadamard_mse":
            return self._quantize_attention_adaptive_outlier_hadamard_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "bitwidth_attention_outlier_hadamard_mse":
            regular_bits, _, _ = self._resolve_effective_bit_allocation(bits, states.shape[-1])
            if regular_bits <= 2:
                return self._quantize_effective_to_segment(
                    states,
                    bits=bits,
                    name=name,
                    quantizer_kind="outlier_hadamard_mse",
                    layer_idx=layer_idx,
                    allow_token_protection=False,
                    query_states=query_states,
                    key_states_for_attention=key_states_for_attention,
                    scaling=scaling,
                )
            return self._quantize_attention_outlier_hadamard_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "entropy_guarded_outlier_hadamard_mse":
            return self._quantize_entropy_guarded_outlier_hadamard_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "hadamard_residual_mse":
            return self._quantize_hadamard_residual_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
            )
        if quantizer_kind == "attention_hadamard_residual_mse":
            return self._quantize_attention_hadamard_residual_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "attention_weighted_hadamard_residual_mse":
            return self._quantize_attention_weighted_hadamard_residual_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "attention_adaptive_rotated_outlier_mse":
            return self._quantize_attention_adaptive_rotated_outlier_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "attention_adaptive_paired_rotated_outlier_mse":
            return self._quantize_attention_adaptive_rotated_outlier_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
                segment_quantizer_kind="attention_adaptive_paired_rotated_outlier_mse",
                quantizer_name="key",
            )
        if quantizer_kind == "entropy_guarded_paired_rotated_outlier_mse":
            return self._quantize_entropy_guarded_paired_rotated_outlier_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind in {
            "rotated_outlier_mse",
            "attention_rotated_outlier_mse",
            "calibrated_rotated_outlier_mse",
            "paired_rotated_outlier_mse",
            "shared_rotated_outlier_mse",
        }:
            return self._quantize_rotated_outlier_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                base_quantizer_kind=(
                    "calibrated_rotation_mse" if quantizer_kind == "calibrated_rotated_outlier_mse" else "mse"
                ),
                segment_quantizer_kind=quantizer_kind,
                quantizer_name=(
                    "key"
                    if quantizer_kind in {"paired_rotated_outlier_mse", "shared_rotated_outlier_mse"}
                    else None
                ),
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
                attention_aware=quantizer_kind == "attention_rotated_outlier_mse",
            )
        if quantizer_kind == "regular_outlier_hadamard_mse":
            return self._quantize_regular_outlier_hadamard_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
            )
        if quantizer_kind == "outlier_only_hadamard_mse":
            return self._quantize_outlier_only_hadamard_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
            )
        if quantizer_kind == "regular_attention_scale_mse":
            return self._quantize_regular_attention_scale_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "adaptive_regular_gain_mse":
            return self._quantize_adaptive_regular_gain_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "attention_adaptive_regular_gain_mse":
            return self._quantize_attention_adaptive_regular_gain_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if quantizer_kind == "rms_regular_gain_mse":
            return self._quantize_rms_regular_gain_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
            )
        if quantizer_kind == "head_rotation_mse":
            return self._quantize_head_rotation_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
            )
        regular_bits, outlier_bits, outlier_count = self._resolve_effective_bit_allocation(bits, states.shape[-1])
        if outlier_count == 0:
            return self._quantize_to_segment(
                states,
                bits=regular_bits,
                name=name,
                quantizer_kind=(
                    "mse"
                    if quantizer_kind
                    in {"regular_gain_mse", "regular_half_gain_mse", "regular_selected_gain_mse", "regular_clipped_gain_mse"}
                    else quantizer_kind
                ),
                layer_idx=layer_idx,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        if self._outlier_policy_for(name) in {"head_dynamic_absmean", "head_error_gain"}:
            return self._quantize_head_adaptive_effective_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind=quantizer_kind,
                layer_idx=layer_idx,
                regular_bits=regular_bits,
                outlier_bits=outlier_bits,
                outlier_count=outlier_count,
            )

        original_shape = tuple(states.shape)
        all_indices = torch.arange(states.shape[-1], device=states.device, dtype=torch.long)
        outlier_indices = self._select_outlier_channels(
            states,
            outlier_count,
            regular_bits=regular_bits,
            outlier_bits=outlier_bits,
            name=name,
            quantizer_kind=quantizer_kind,
            layer_idx=layer_idx,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
        )
        regular_mask = torch.ones(states.shape[-1], device=states.device, dtype=torch.bool)
        regular_mask[outlier_indices] = False
        regular_indices = all_indices[regular_mask]

        regular_states = states.index_select(-1, regular_indices)
        outlier_states = states.index_select(-1, outlier_indices)
        subspace_quantizer_kind = "mse" if quantizer_kind == "calibrated_rotation_mse" else quantizer_kind
        regular_segment = self._quantize_to_segment(
            regular_states,
            bits=regular_bits,
            name=f"{name}_regular",
            quantizer_kind=(
                "gain_mse"
                if quantizer_kind == "regular_gain_mse"
                else "regular_half_gain_mse"
                if quantizer_kind == "regular_half_gain_mse"
                else "selected_gain_mse"
                if quantizer_kind == "regular_selected_gain_mse"
                else "clipped_gain_mse"
                if quantizer_kind == "regular_clipped_gain_mse"
                else subspace_quantizer_kind
            ),
            layer_idx=layer_idx,
        )
        outlier_segment = self._quantize_to_segment(
            outlier_states,
            bits=outlier_bits,
            name=f"{name}_outlier",
            quantizer_kind=(
                "mse"
                if quantizer_kind
                in {"regular_gain_mse", "regular_half_gain_mse", "regular_selected_gain_mse", "regular_clipped_gain_mse"}
                else subspace_quantizer_kind
            ),
            layer_idx=layer_idx,
        )
        if isinstance(regular_segment, RawTensorSegment) or isinstance(outlier_segment, RawTensorSegment):
            raise RuntimeError("outlier effective-bit segments must use packed subsegments")

        return OutlierMSESegment(
            regular=regular_segment,
            outlier=outlier_segment,
            regular_indices=regular_indices.to(dtype=torch.int16),
            outlier_indices=outlier_indices.to(dtype=torch.int16),
            shape=original_shape,
            dtype=states.dtype,
            requested_bits=float(bits),
            policy=self._outlier_policy_for(name),
        )

    def _attention_auto_mse_error(
        self,
        states: torch.Tensor,
        states_hat: torch.Tensor,
        *,
        name: str,
        query_states: torch.Tensor | None,
        key_states_for_attention: torch.Tensor | None,
        scaling: float,
    ) -> float:
        if (
            query_states is None
            or key_states_for_attention is None
            or states.ndim != 4
            or query_states.ndim != 4
            or key_states_for_attention.ndim != 4
        ):
            return float(torch.mean((states.detach().to(dtype=torch.float32) - states_hat.to(dtype=torch.float32)) ** 2).item())
        with torch.no_grad():
            if name == "key":
                original_scores = self._grouped_attention_scores(query_states, states, scaling=scaling)
                candidate_scores = self._grouped_attention_scores(query_states, states_hat, scaling=scaling)
                probs = self._causal_attention_probs(original_scores, key_len=states.shape[-2]).detach()
                score_error = (original_scores - candidate_scores).square()
                return float((probs * score_error).mean().item())
            if name == "value":
                key_scores = self._grouped_attention_scores(
                    query_states,
                    key_states_for_attention,
                    scaling=scaling,
                )
                probs = self._causal_attention_probs(
                    key_scores,
                    key_len=key_states_for_attention.shape[-2],
                ).detach()
                value_error = (states.detach().to(dtype=torch.float32) - states_hat.to(dtype=torch.float32)).square().sum(dim=-1)
                token_weight = probs.square().mean(dim=(2, 3))
                return float((token_weight * value_error).mean().item())
        return float(torch.mean((states.detach().to(dtype=torch.float32) - states_hat.to(dtype=torch.float32)) ** 2).item())

    def _attention_entropy_ratio(
        self,
        *,
        query_states: torch.Tensor | None,
        key_states: torch.Tensor | None,
        scaling: float,
        query_tokens: int = 8,
    ) -> float | None:
        if query_states is None or key_states is None or query_states.ndim != 4 or key_states.ndim != 4:
            return None
        if key_states.shape[-2] <= 1:
            return 0.0
        with torch.no_grad():
            requested = min(query_tokens, query_states.shape[-2])
            query_focus = query_states[..., -requested:, :]
            bsz, num_query_heads, _, head_dim = query_focus.shape
            num_key_heads = key_states.shape[1]
            if num_query_heads % num_key_heads != 0 or head_dim != key_states.shape[-1]:
                return None
            groups = num_query_heads // num_key_heads
            query_grouped = query_focus.reshape(bsz, num_key_heads, groups, requested, head_dim)
            scores = torch.einsum("bhgqd,bhsd->bhgqs", query_grouped.to(torch.float32), key_states.to(torch.float32))
            probs = self._causal_attention_probs(scores * float(scaling), key_len=key_states.shape[-2])
            entropy = -(probs.clamp_min(torch.finfo(torch.float32).tiny).log() * probs).sum(dim=-1)
            normalizer = torch.log(torch.tensor(float(key_states.shape[-2]), device=entropy.device, dtype=torch.float32))
            return float((entropy / normalizer.clamp_min(torch.finfo(torch.float32).eps)).mean().item())

    def _quantize_attention_auto_mse_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
        query_states: torch.Tensor | None,
        key_states_for_attention: torch.Tensor | None,
        scaling: float,
    ) -> CacheSegment:
        candidate_kinds = (
            "mse",
            "gain_mse",
            "unit_mse",
            "learned_unit_mse_block2",
        )
        best_segment: CacheSegment | None = None
        best_error: float | None = None
        for candidate_kind in candidate_kinds:
            segment = self._quantize_effective_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind=candidate_kind,
                layer_idx=layer_idx,
                allow_token_protection=False,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
            decoded = self._decode_segment(segment, name=name, layer_idx=layer_idx)
            error = self._attention_auto_mse_error(
                states,
                decoded,
                name=name,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
            if best_error is None or error < best_error:
                best_error = error
                best_segment = segment
        if best_segment is None:
            raise RuntimeError("attention_auto_mse did not produce a candidate segment")
        return best_segment

    def _decode_segment(self, segment: CacheSegment, *, name: str, layer_idx: int) -> torch.Tensor:
        if isinstance(segment, RawTensorSegment):
            return segment.tensor
        if isinstance(segment, UniformAffineSegment):
            qmax = (1 << (segment.bits - 1)) - 1
            if qmax < 1:
                qmax = 1
            encoded = unpack_indices(segment.packed_values, bits=segment.bits, shape=segment.shape)
            quantized = encoded.to(dtype=torch.int16) - qmax
            return quantized.to(device=segment.scales.device, dtype=segment.dtype) * segment.scales
        if isinstance(segment, CenteredSegment):
            residual = self._decode_segment(segment.residual, name=f"{name}_centered", layer_idx=layer_idx)
            return residual + segment.center.to(device=residual.device, dtype=segment.dtype)
        if isinstance(segment, OutlierHadamardSegment):
            dimension = segment.shape[-1]
            base_quantizer = self._get_quantizer(
                quantizer_kind="mse",
                name=name,
                dimension=dimension,
                bits=segment.bits,
                layer_idx=layer_idx,
                device=segment.packed_indices.device,
                dtype=segment.dtype,
            )
            if not isinstance(base_quantizer, TurboQuantMSE):
                raise TypeError("outlier_hadamard_mse expects TurboQuantMSE codebooks")
            code_indices = unpack_indices(segment.packed_indices, bits=segment.bits, shape=segment.shape).reshape(-1, dimension)
            transformed_hat = base_quantizer.centroids[code_indices.to(device=segment.packed_indices.device, dtype=torch.long)]
            signs = segment.signs.to(device=segment.packed_indices.device)
            signs = torch.where(signs > 0, 1.0, -1.0).to(dtype=segment.dtype)
            unpermuted_unit = self._inverse_outlier_hadamard_transform(
                transformed_hat.to(dtype=segment.dtype),
                permutation=segment.permutation.to(device=segment.packed_indices.device, dtype=torch.long),
                signs=signs,
                block_size=segment.block_size,
            )
            norms = segment.norms.reshape(-1).to(device=unpermuted_unit.device, dtype=segment.dtype)
            return (unpermuted_unit * norms.unsqueeze(-1)).reshape(segment.shape)
        if isinstance(segment, RmsScaledSegment):
            residual = self._decode_segment(segment.residual, name=f"{name}_rms", layer_idx=layer_idx)
            return residual * segment.scales.to(device=residual.device, dtype=segment.dtype)
        if isinstance(segment, VectorAdaptiveRotationSegment):
            baseline_hat = self._decode_segment(segment.baseline, name=f"{name}_vector_baseline", layer_idx=layer_idx)
            candidate_hat = self._decode_segment(segment.candidate, name=f"{name}_vector_candidate", layer_idx=layer_idx)
            mask = unpack_indices(
                segment.packed_candidate_mask,
                bits=1,
                shape=segment.selector_shape,
            ).reshape(-1).to(device=baseline_hat.device, dtype=torch.bool)
            dimension = segment.shape[-1]
            output = torch.empty((_numel(segment.shape[:-1]), dimension), device=baseline_hat.device, dtype=segment.dtype)
            output[~mask] = baseline_hat.reshape(-1, dimension)
            output[mask] = candidate_hat.reshape(-1, dimension)
            return output.reshape(segment.shape)
        if isinstance(segment, HeadRotationMSESegment):
            head_outputs = [
                self._decode_segment(head_segment, name=f"{name}_head{head_idx}", layer_idx=layer_idx).unsqueeze(-3)
                for head_idx, head_segment in enumerate(segment.head_segments)
            ]
            return torch.cat(head_outputs, dim=-3).reshape(segment.shape)
        if isinstance(segment, HeadHadamardSegment):
            transformed_hat = self._decode_segment(segment.residual, name=f"{name}_head_hadamard", layer_idx=layer_idx)
            return self._inverse_head_hadamard_transform(transformed_hat, name=name, layer_idx=layer_idx).reshape(segment.shape)
        if isinstance(segment, HadamardResidualSegment):
            base_hat = self._decode_segment(segment.base, name=f"{name}_hadamard_residual_base", layer_idx=layer_idx)
            dimension = segment.shape[-1]
            residual_count = int(segment.residual_indices.numel())
            if residual_count == 0:
                return base_hat.reshape(segment.shape)
            encoded = unpack_indices(
                segment.packed_residual_signs,
                bits=1,
                shape=(*segment.shape[:-1], residual_count),
            ).reshape(-1, residual_count)
            signs = torch.where(encoded.to(device=base_hat.device) > 0, 1.0, -1.0).to(dtype=segment.dtype)
            coeffs = torch.zeros(
                (_numel(segment.shape[:-1]), dimension),
                device=base_hat.device,
                dtype=segment.dtype,
            )
            scales = segment.residual_scales.reshape(-1, 1).to(device=base_hat.device, dtype=segment.dtype)
            coeffs.index_copy_(
                -1,
                segment.residual_indices.to(device=base_hat.device, dtype=torch.long),
                signs * scales,
            )
            transform = self._hadamard_residual_matrix(
                dimension,
                device=base_hat.device,
                dtype=segment.dtype,
                name=name,
                layer_idx=layer_idx,
            )
            residual_hat = coeffs @ transform
            return (base_hat.reshape(-1, dimension) + residual_hat).reshape(segment.shape)
        if isinstance(segment, OutlierMSESegment):
            regular = self._decode_segment(segment.regular, name=f"{name}_regular", layer_idx=layer_idx)
            outlier = self._decode_segment(segment.outlier, name=f"{name}_outlier", layer_idx=layer_idx)
            output = torch.empty(segment.shape, device=regular.device, dtype=segment.dtype)
            output.index_copy_(-1, segment.regular_indices.to(device=regular.device, dtype=torch.long), regular)
            output.index_copy_(-1, segment.outlier_indices.to(device=regular.device, dtype=torch.long), outlier)
            return output
        if isinstance(segment, RotatedOutlierMSESegment):
            dimension = segment.shape[-1]
            base_kind = (
                "calibrated_rotation_mse"
                if segment.quantizer_kind == "calibrated_rotated_outlier_mse"
                else "mse"
            )
            quantizer_name = (
                "key"
                if segment.quantizer_kind
                in {
                    "attention_adaptive_paired_rotated_outlier_mse",
                    "entropy_guarded_paired_rotated_outlier_mse",
                    "paired_rotated_outlier_mse",
                    "shared_rotated_outlier_mse",
                }
                else name
            )
            quantizer = self._get_quantizer(
                quantizer_kind=base_kind,
                name=quantizer_name,
                dimension=dimension,
                bits=segment.regular_bits,
                layer_idx=layer_idx,
                device=segment.packed_regular_indices.device,
                dtype=segment.dtype,
            )
            if not isinstance(quantizer, TurboQuantMSE):
                raise TypeError("rotated_outlier_mse expects TurboQuantMSE quantizers")
            outlier_quantizer = self._get_quantizer(
                quantizer_kind=base_kind,
                name=quantizer_name,
                dimension=dimension,
                bits=segment.outlier_bits,
                layer_idx=layer_idx,
                device=segment.packed_outlier_indices.device,
                dtype=segment.dtype,
            )
            if not isinstance(outlier_quantizer, TurboQuantMSE):
                raise TypeError("rotated_outlier_mse expects TurboQuantMSE quantizers")
            regular_indices = unpack_indices(
                segment.packed_regular_indices,
                bits=segment.regular_bits,
                shape=(*segment.shape[:-1], int(segment.regular_indices.numel())),
            ).reshape(-1, int(segment.regular_indices.numel()))
            outlier_indices = unpack_indices(
                segment.packed_outlier_indices,
                bits=segment.outlier_bits,
                shape=(*segment.shape[:-1], int(segment.outlier_indices.numel())),
            ).reshape(-1, int(segment.outlier_indices.numel()))
            rotated_hat = torch.empty(
                (_numel(segment.shape[:-1]), dimension),
                device=segment.packed_regular_indices.device,
                dtype=segment.dtype,
            )
            centroids = quantizer.centroids
            rotated_hat.index_copy_(
                -1,
                segment.regular_indices.to(device=rotated_hat.device, dtype=torch.long),
                centroids[regular_indices.to(device=rotated_hat.device, dtype=torch.long)].to(dtype=segment.dtype),
            )
            rotated_hat.index_copy_(
                -1,
                segment.outlier_indices.to(device=rotated_hat.device, dtype=torch.long),
                outlier_quantizer.centroids[outlier_indices.to(device=rotated_hat.device, dtype=torch.long)].to(
                    dtype=segment.dtype
                ),
            )
            x_hat_unit = rotated_hat @ quantizer.inverse_transform.T
            regular_scales = segment.regular_norms.reshape(-1).to(device=x_hat_unit.device, dtype=segment.dtype)
            outlier_scales = segment.outlier_norms.reshape(-1).to(device=x_hat_unit.device, dtype=segment.dtype)
            scales = 0.5 * (regular_scales + outlier_scales)
            return (x_hat_unit * scales.unsqueeze(-1)).reshape(segment.shape)
        if isinstance(segment, HeadAdaptiveOutlierMSESegment):
            head_outputs = [
                self._decode_segment(head_segment, name=name, layer_idx=layer_idx).unsqueeze(-3)
                for head_segment in segment.head_segments
            ]
            return torch.cat(head_outputs, dim=-3).reshape(segment.shape)
        if isinstance(segment, TokenProtectedSegment):
            quantized = self._decode_segment(segment.quantized, name=name, layer_idx=layer_idx)
            output = torch.empty(segment.shape, device=segment.raw_tensor.device, dtype=segment.dtype)
            output.index_copy_(-2, segment.raw_indices.to(device=output.device, dtype=torch.long), segment.raw_tensor)
            output.index_copy_(-2, segment.quantized_indices.to(device=output.device, dtype=torch.long), quantized)
            return output
        if isinstance(segment, PackedProdSegment):
            dimension = segment.shape[-1]
            quantizer = self._get_quantizer(
                quantizer_kind="prod",
                name=name,
                dimension=dimension,
                bits=segment.bits,
                layer_idx=layer_idx,
                device=segment.packed_mse_indices.device,
                dtype=segment.dtype,
            )
            if not isinstance(quantizer, TurboQuantProd):
                raise TypeError("expected TurboQuantProd")
            mse_bits = segment.bits - 1
            mse_indices = unpack_indices(segment.packed_mse_indices, bits=mse_bits, shape=segment.shape)
            qjl_bits = unpack_indices(segment.packed_qjl_signs, bits=1, shape=segment.shape)
            qjl = torch.where(qjl_bits > 0, 1, -1).to(dtype=torch.int8)
            quantized = ProdQuantized(
                mse=MSEQuantized(indices=mse_indices.reshape(-1, dimension), norms=segment.mse_norms.reshape(-1)),
                qjl=qjl.reshape(-1, dimension),
                residual_norms=segment.residual_norms.reshape(-1),
            )
            return quantizer.dequantize(quantized).reshape(segment.shape)

        dimension = segment.shape[-1]
        quantizer_kind = segment.quantizer_kind
        if quantizer_kind in {"rotation_bank_mse", "attention_rotation_bank_mse", "segment_rotation_bank_mse"}:
            if segment.packed_rotation_ids is None or segment.rotation_id_bits is None:
                raise ValueError("rotation_bank_mse segment is missing rotation ids")
            indices = unpack_indices(segment.packed_indices, bits=segment.bits, shape=segment.shape).reshape(-1, dimension)
            norms = segment.norms.reshape(-1)
            rotation_ids = unpack_indices(
                segment.packed_rotation_ids,
                bits=segment.rotation_id_bits,
                shape=segment.shape[:-1],
            ).reshape(-1).to(device=segment.packed_indices.device, dtype=torch.long)
            output = torch.empty((indices.shape[0], dimension), device=segment.packed_indices.device, dtype=segment.dtype)
            bank_size = int(segment.rotation_bank_size or self.config.rotation_bank_size)
            for rotation_idx in range(bank_size):
                mask = rotation_ids == rotation_idx
                if not bool(mask.any()):
                    continue
                quantizer = self._get_quantizer(
                    quantizer_kind=quantizer_kind,
                    name=name,
                    dimension=dimension,
                    bits=segment.bits,
                    layer_idx=layer_idx,
                    device=segment.packed_indices.device,
                    dtype=segment.dtype,
                    rotation_bank_index=rotation_idx,
                )
                if not isinstance(quantizer, TurboQuantMSE):
                    raise TypeError("rotation_bank_mse expects TurboQuantMSE quantizers")
                quantized = MSEQuantized(indices=indices[mask], norms=norms[mask])
                output[mask] = quantizer.dequantize(quantized)
            return output.reshape(segment.shape)

        quantizer = self._get_quantizer(
            quantizer_kind=quantizer_kind,
            name=name,
            dimension=dimension,
            bits=segment.bits,
            layer_idx=layer_idx,
            device=segment.packed_indices.device,
            dtype=segment.dtype,
        )
        index_shape = segment.index_shape
        indices = unpack_indices(segment.packed_indices, bits=segment.bits * segment.block_size, shape=index_shape)
        if isinstance(quantizer, (TurboQuantBlockMSE, TurboQuantLearnedBlockMSE)):
            quantized = BlockMSEQuantized(indices=indices.reshape(-1, dimension // segment.block_size), norms=segment.norms.reshape(-1))
        else:
            quantized = MSEQuantized(indices=indices.reshape(-1, dimension), norms=segment.norms.reshape(-1))
        return quantizer.dequantize(quantized).reshape(segment.shape)

    def _outlier_hadamard_permutation(self, states: torch.Tensor, *, block_size: int) -> torch.Tensor:
        dimension = states.shape[-1]
        scores = states.detach().abs().to(dtype=torch.float32).mean(dim=tuple(range(states.ndim - 1)))
        order = torch.argsort(scores, descending=True)
        slots: list[int] = []
        block_count = max(1, ceil(dimension / block_size))
        for offset in range(block_size):
            for block_idx in range(block_count):
                slot = block_idx * block_size + offset
                if slot < dimension:
                    slots.append(slot)
        permutation = torch.empty(dimension, device=states.device, dtype=torch.long)
        permutation[torch.tensor(slots, device=states.device, dtype=torch.long)] = order
        return permutation

    def _outlier_hadamard_signs(self, *, dimension: int, device: torch.device, dtype: torch.dtype, seed: int) -> torch.Tensor:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        signs = torch.randint(0, 2, (dimension,), generator=gen, device=device, dtype=torch.int8)
        return signs.to(dtype=dtype).mul_(2).sub_(1)

    def _block_hadamard_transform(self, x: torch.Tensor, *, block_size: int, inverse: bool = False) -> torch.Tensor:
        dimension = x.shape[-1]
        pieces = []
        for start in range(0, dimension, block_size):
            end = min(start + block_size, dimension)
            current = end - start
            if current == 1:
                pieces.append(x[..., start:end])
                continue
            if current & (current - 1):
                transform = hadamard_orthogonal(1 << (current - 1).bit_length(), device=x.device, dtype=x.dtype, seed=None)
                transform = transform[:current, :current]
                q, _ = torch.linalg.qr(transform.to(dtype=torch.float32))
                transform = q.to(device=x.device, dtype=x.dtype)
            else:
                transform = hadamard_orthogonal(current, device=x.device, dtype=x.dtype, seed=None)
            pieces.append(x[..., start:end] @ (transform if inverse else transform.T))
        return torch.cat(pieces, dim=-1)

    def _apply_outlier_hadamard_transform(
        self,
        x: torch.Tensor,
        *,
        permutation: torch.Tensor,
        signs: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        permuted = x.index_select(-1, permutation) * signs
        return self._block_hadamard_transform(permuted, block_size=block_size, inverse=False)

    def _inverse_outlier_hadamard_transform(
        self,
        transformed: torch.Tensor,
        *,
        permutation: torch.Tensor,
        signs: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        permuted_hat = self._block_hadamard_transform(transformed, block_size=block_size, inverse=True) * signs
        output = torch.empty_like(permuted_hat)
        output.index_copy_(-1, permutation, permuted_hat)
        return output

    def _quantize_outlier_hadamard_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: int,
        name: str,
        layer_idx: int,
        permutation: torch.Tensor | None = None,
        signs: torch.Tensor | None = None,
    ) -> OutlierHadamardSegment:
        if bits >= 16:
            raise ValueError("outlier_hadamard_mse is only used for packed low-bit segments")
        original_shape = tuple(states.shape)
        dimension = original_shape[-1]
        block_size = min(self.config.outlier_hadamard_block_size, dimension)
        flat = states.reshape(-1, dimension)
        norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True).clamp_min(torch.finfo(states.dtype).eps)
        unit = flat / norms
        if permutation is None:
            permutation = self._outlier_hadamard_permutation(states, block_size=block_size)
        else:
            permutation = permutation.to(device=states.device, dtype=torch.long)
        if signs is None:
            signs = self._outlier_hadamard_signs(
                dimension=dimension,
                device=states.device,
                dtype=states.dtype,
                seed=self.config.seed + 1_000 * layer_idx + (0 if name == "key" else 500) + 17_171,
            )
        else:
            signs = signs.to(device=states.device, dtype=states.dtype)
        transformed = self._apply_outlier_hadamard_transform(unit, permutation=permutation, signs=signs, block_size=block_size)
        base_quantizer = self._get_quantizer(
            quantizer_kind="mse",
            name=name,
            dimension=dimension,
            bits=bits,
            layer_idx=layer_idx,
            device=states.device,
            dtype=states.dtype,
        )
        if not isinstance(base_quantizer, TurboQuantMSE):
            raise TypeError("outlier_hadamard_mse expects TurboQuantMSE codebooks")
        distances = torch.abs(transformed.unsqueeze(-1) - base_quantizer.centroids)
        indices = torch.argmin(distances, dim=-1).to(torch.int16)
        return OutlierHadamardSegment(
            packed_indices=pack_indices(indices, bits),
            norms=norms.squeeze(-1).reshape(original_shape[:-1]).detach(),
            permutation=permutation.to(dtype=torch.int16).detach(),
            signs=signs.to(dtype=torch.int8).detach(),
            shape=original_shape,
            bits=bits,
            dtype=states.dtype,
            block_size=block_size,
        )

    def _quantize_adaptive_outlier_hadamard_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: int,
        name: str,
        layer_idx: int,
    ) -> CacheSegment:
        baseline_segment = self._quantize_to_segment(
            states,
            bits=bits,
            name=name,
            quantizer_kind="mse",
            layer_idx=layer_idx,
        )
        hadamard_segment = self._quantize_outlier_hadamard_to_segment(
            states,
            bits=bits,
            name=name,
            layer_idx=layer_idx,
        )
        states_f = states.detach().to(dtype=torch.float32)
        baseline_hat = self._decode_segment(baseline_segment, name=name, layer_idx=layer_idx).to(dtype=torch.float32)
        hadamard_hat = self._decode_segment(hadamard_segment, name=name, layer_idx=layer_idx).to(dtype=torch.float32)
        baseline_error = torch.mean((states_f - baseline_hat) ** 2)
        hadamard_error = torch.mean((states_f - hadamard_hat) ** 2)
        if bool(hadamard_error < baseline_error):
            return hadamard_segment
        return baseline_segment

    def _quantize_vector_adaptive_outlier_hadamard_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: int,
        name: str,
        layer_idx: int,
        activation_margin: float = 1.0,
    ) -> CacheSegment:
        if bits >= 16:
            return RawTensorSegment(tensor=states.detach())
        original_shape = tuple(states.shape)
        dimension = original_shape[-1]
        flat = states.reshape(-1, dimension)
        if flat.shape[0] <= 1:
            return self._quantize_to_segment(
                states,
                bits=bits,
                name=f"{name}_vector_baseline",
                quantizer_kind="mse",
                layer_idx=layer_idx,
            )

        block_size = min(self.config.outlier_hadamard_block_size, dimension)
        permutation = self._outlier_hadamard_permutation(states, block_size=block_size)
        signs = self._outlier_hadamard_signs(
            dimension=dimension,
            device=states.device,
            dtype=states.dtype,
            seed=self.config.seed + 1_000 * layer_idx + (0 if name == "key" else 500) + 17_171,
        )

        baseline_probe = self._quantize_to_segment(
            states,
            bits=bits,
            name=f"{name}_vector_baseline_probe",
            quantizer_kind="mse",
            layer_idx=layer_idx,
        )
        candidate_probe = self._quantize_outlier_hadamard_to_segment(
            states,
            bits=bits,
            name=f"{name}_vector_candidate_probe",
            layer_idx=layer_idx,
            permutation=permutation,
            signs=signs,
        )
        states_f = flat.detach().to(dtype=torch.float32)
        baseline_hat = self._decode_segment(
            baseline_probe,
            name=f"{name}_vector_baseline_probe",
            layer_idx=layer_idx,
        ).reshape(-1, dimension).to(dtype=torch.float32)
        candidate_hat = self._decode_segment(
            candidate_probe,
            name=f"{name}_vector_candidate_probe",
            layer_idx=layer_idx,
        ).reshape(-1, dimension).to(dtype=torch.float32)
        baseline_error = (states_f - baseline_hat).square().mean(dim=-1)
        candidate_error = (states_f - candidate_hat).square().mean(dim=-1)
        candidate_mask = candidate_error < baseline_error * float(activation_margin)
        candidate_count = int(candidate_mask.sum().item())
        if candidate_count == 0:
            return baseline_probe
        if candidate_count == flat.shape[0]:
            return candidate_probe

        baseline_states = flat[~candidate_mask]
        candidate_states = flat[candidate_mask]
        baseline_segment = self._quantize_to_segment(
            baseline_states,
            bits=bits,
            name=f"{name}_vector_baseline",
            quantizer_kind="mse",
            layer_idx=layer_idx,
        )
        candidate_segment = self._quantize_outlier_hadamard_to_segment(
            candidate_states,
            bits=bits,
            name=f"{name}_vector_candidate",
            layer_idx=layer_idx,
            permutation=permutation,
            signs=signs,
        )
        return VectorAdaptiveRotationSegment(
            baseline=baseline_segment,
            candidate=candidate_segment,
            packed_candidate_mask=pack_indices(candidate_mask.reshape(original_shape[:-1]).to(dtype=torch.int16), 1).detach(),
            selector_shape=original_shape[:-1],
            shape=original_shape,
            dtype=states.dtype,
            bits=bits,
            candidate_kind="outlier_hadamard_mse",
        )

    def _quantize_attention_adaptive_outlier_hadamard_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
        query_states: torch.Tensor | None,
        key_states_for_attention: torch.Tensor | None,
        scaling: float,
    ) -> CacheSegment:
        baseline_segment = self._quantize_effective_to_segment(
            states,
            bits=bits,
            name=name,
            quantizer_kind="mse",
            layer_idx=layer_idx,
            allow_token_protection=False,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
        )
        candidate_segment = self._quantize_effective_to_segment(
            states,
            bits=bits,
            name=name,
            quantizer_kind="outlier_hadamard_mse",
            layer_idx=layer_idx,
            allow_token_protection=False,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
        )
        baseline_hat = self._decode_segment(baseline_segment, name=name, layer_idx=layer_idx)
        candidate_hat = self._decode_segment(candidate_segment, name=name, layer_idx=layer_idx)
        baseline_error = self._attention_auto_mse_error(
            states,
            baseline_hat,
            name=name,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
        )
        candidate_error = self._attention_auto_mse_error(
            states,
            candidate_hat,
            name=name,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
        )
        if candidate_error < baseline_error:
            return candidate_segment
        return baseline_segment

    def _quantize_attention_scale_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: int,
        name: str,
        layer_idx: int,
        query_states: torch.Tensor | None,
        scaling: float,
    ) -> CacheSegment:
        if name != "key" or query_states is None or states.ndim != 4 or query_states.ndim != 4:
            return self._quantize_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind="mse",
                layer_idx=layer_idx,
            )
        original_shape = tuple(states.shape)
        dimension = original_shape[-1]
        flat = states.reshape(-1, dimension)
        quantizer = self._get_quantizer(
            quantizer_kind="mse",
            name=name,
            dimension=dimension,
            bits=bits,
            layer_idx=layer_idx,
            device=states.device,
            dtype=states.dtype,
        )
        if not isinstance(quantizer, TurboQuantMSE):
            raise TypeError("attention_scale_mse expects TurboQuantMSE")
        quantized = quantizer.quantize(flat)
        unit_hat = quantizer.dequantize(
            MSEQuantized(
                indices=quantized.indices,
                norms=torch.ones_like(quantized.norms),
            )
        ).reshape(original_shape)

        query_tokens = min(self.config.attention_error_query_tokens, query_states.shape[-2])
        query_focus = query_states[..., -query_tokens:, :]
        bsz, num_query_heads, _, head_dim = query_focus.shape
        num_key_heads = states.shape[1]
        if num_query_heads % num_key_heads != 0 or head_dim != dimension:
            return PackedMSESegment(
                packed_indices=pack_indices(quantized.indices, bits),
                norms=quantized.norms.reshape(original_shape[:-1]).detach(),
                shape=original_shape,
                bits=bits,
                dtype=states.dtype,
                quantizer_kind="mse",
            )
        num_groups = num_query_heads // num_key_heads
        query_grouped = query_focus.reshape(bsz, num_key_heads, num_groups, query_tokens, head_dim).to(torch.float32)
        original_scores = torch.einsum("bhgqd,bhsd->bhgqs", query_grouped, states.to(torch.float32)) * float(scaling)
        unit_scores = torch.einsum("bhgqd,bhsd->bhgqs", query_grouped, unit_hat.to(torch.float32)) * float(scaling)
        numerator = (original_scores * unit_scores).sum(dim=(2, 3))
        denominator = unit_scores.square().sum(dim=(2, 3)).clamp_min(torch.finfo(torch.float32).eps)
        scales = (numerator / denominator).clamp_min(0.0).to(dtype=states.dtype)
        return PackedMSESegment(
            packed_indices=pack_indices(quantized.indices, bits),
            norms=scales.reshape(original_shape[:-1]).detach(),
            shape=original_shape,
            bits=bits,
            dtype=states.dtype,
            block_size=1,
            quantizer_kind="mse",
        )

    def _quantize_rotation_bank_mse_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: int,
        name: str,
        layer_idx: int,
        attention_aware: bool = False,
        query_states: torch.Tensor | None = None,
        key_states_for_attention: torch.Tensor | None = None,
        scaling: float = 1.0,
    ) -> PackedMSESegment:
        if bits >= 16:
            raise ValueError("rotation_bank_mse is only used for packed low-bit segments")
        original_shape = tuple(states.shape)
        dimension = original_shape[-1]
        flat = states.reshape(-1, dimension)
        flat_f = flat.detach().to(dtype=torch.float32)
        best_error: torch.Tensor | None = None
        best_global_error: float | None = None
        best_indices: torch.Tensor | None = None
        best_norms: torch.Tensor | None = None
        best_rotation_ids: torch.Tensor | None = None

        for rotation_idx in range(self.config.rotation_bank_size):
            quantizer = self._get_quantizer(
                quantizer_kind="rotation_bank_mse",
                name=name,
                dimension=dimension,
                bits=bits,
                layer_idx=layer_idx,
                device=states.device,
                dtype=states.dtype,
                rotation_bank_index=rotation_idx,
            )
            if not isinstance(quantizer, TurboQuantMSE):
                raise TypeError("rotation_bank_mse expects TurboQuantMSE quantizers")
            quantized = quantizer.quantize(flat)
            decoded = quantizer.dequantize(quantized).to(dtype=torch.float32)
            candidate_ids = torch.full(
                (flat.shape[0],),
                rotation_idx,
                device=states.device,
                dtype=torch.int16,
            )
            if attention_aware and query_states is not None and key_states_for_attention is not None:
                decoded_states = decoded.reshape(original_shape).to(dtype=states.dtype)
                global_error = self._attention_auto_mse_error(
                    states,
                    decoded_states,
                    name=name,
                    query_states=query_states,
                    key_states_for_attention=key_states_for_attention,
                    scaling=scaling,
                )
                if best_global_error is None or global_error < best_global_error:
                    best_global_error = global_error
                    best_indices = quantized.indices
                    best_norms = quantized.norms
                    best_rotation_ids = candidate_ids
                continue
            error = (flat_f - decoded).square().sum(dim=-1)
            if best_error is None:
                best_error = error
                best_indices = quantized.indices
                best_norms = quantized.norms
                best_rotation_ids = candidate_ids
                continue
            selected = error < best_error
            best_error = torch.where(selected, error, best_error)
            best_indices = torch.where(selected.unsqueeze(-1), quantized.indices, best_indices)
            best_norms = torch.where(selected, quantized.norms, best_norms)
            best_rotation_ids = torch.where(selected, candidate_ids, best_rotation_ids)

        if best_indices is None or best_norms is None or best_rotation_ids is None:
            raise RuntimeError("rotation_bank_mse did not produce a candidate segment")
        packed = pack_indices(best_indices, bits)
        rotation_id_bits = max(1, int((self.config.rotation_bank_size - 1).bit_length()))
        packed_rotation_ids = pack_indices(best_rotation_ids, rotation_id_bits)
        return PackedMSESegment(
            packed_indices=packed,
            norms=best_norms.reshape(original_shape[:-1]).detach(),
            shape=original_shape,
            bits=bits,
            dtype=states.dtype,
            block_size=1,
            quantizer_kind="attention_rotation_bank_mse" if attention_aware else "rotation_bank_mse",
            packed_rotation_ids=packed_rotation_ids.detach(),
            rotation_id_bits=rotation_id_bits,
            rotation_bank_size=self.config.rotation_bank_size,
        )

    def _quantize_fixed_rotation_bank_mse_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: int,
        name: str,
        layer_idx: int,
        rotation_bank_index: int,
        quantizer_kind: str,
    ) -> PackedMSESegment:
        original_shape = tuple(states.shape)
        dimension = original_shape[-1]
        flat = states.reshape(-1, dimension)
        quantizer = self._get_quantizer(
            quantizer_kind=quantizer_kind,
            name=name,
            dimension=dimension,
            bits=bits,
            layer_idx=layer_idx,
            device=states.device,
            dtype=states.dtype,
            rotation_bank_index=rotation_bank_index,
        )
        if not isinstance(quantizer, TurboQuantMSE):
            raise TypeError("rotation-bank quantizers expect TurboQuantMSE")
        quantized = quantizer.quantize(flat)
        rotation_ids = torch.full(
            original_shape[:-1],
            rotation_bank_index,
            device=states.device,
            dtype=torch.int16,
        )
        rotation_id_bits = max(1, int((self.config.rotation_bank_size - 1).bit_length()))
        return PackedMSESegment(
            packed_indices=pack_indices(quantized.indices, bits),
            norms=quantized.norms.reshape(original_shape[:-1]).detach(),
            shape=original_shape,
            bits=bits,
            dtype=states.dtype,
            block_size=1,
            quantizer_kind=quantizer_kind,
            packed_rotation_ids=pack_indices(rotation_ids, rotation_id_bits).detach(),
            rotation_id_bits=rotation_id_bits,
            rotation_bank_size=self.config.rotation_bank_size,
        )

    def _quantize_fixed_rotation_bank_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
        rotation_bank_index: int,
        quantizer_kind: str,
    ) -> CacheSegment:
        regular_bits, outlier_bits, outlier_count = self._resolve_effective_bit_allocation(bits, states.shape[-1])
        if outlier_count == 0:
            return self._quantize_fixed_rotation_bank_mse_to_segment(
                states,
                bits=regular_bits,
                name=name,
                layer_idx=layer_idx,
                rotation_bank_index=rotation_bank_index,
                quantizer_kind=quantizer_kind,
            )

        outlier_indices = self._select_outlier_channels(
            states,
            outlier_count,
            regular_bits=regular_bits,
            outlier_bits=outlier_bits,
            name=name,
            quantizer_kind="mse",
            layer_idx=layer_idx,
        )
        original_shape = tuple(states.shape)
        all_indices = torch.arange(states.shape[-1], device=states.device, dtype=torch.long)
        regular_mask = torch.ones(states.shape[-1], device=states.device, dtype=torch.bool)
        regular_mask[outlier_indices] = False
        regular_indices = all_indices[regular_mask]
        regular_segment = self._quantize_fixed_rotation_bank_mse_to_segment(
            states.index_select(-1, regular_indices),
            bits=regular_bits,
            name=f"{name}_regular",
            layer_idx=layer_idx,
            rotation_bank_index=rotation_bank_index,
            quantizer_kind=quantizer_kind,
        )
        outlier_segment = self._quantize_fixed_rotation_bank_mse_to_segment(
            states.index_select(-1, outlier_indices),
            bits=outlier_bits,
            name=f"{name}_outlier",
            layer_idx=layer_idx,
            rotation_bank_index=rotation_bank_index,
            quantizer_kind=quantizer_kind,
        )
        return OutlierMSESegment(
            regular=regular_segment,
            outlier=outlier_segment,
            regular_indices=regular_indices.to(dtype=torch.int16),
            outlier_indices=outlier_indices.to(dtype=torch.int16),
            shape=original_shape,
            dtype=states.dtype,
            requested_bits=float(bits),
            policy=self._outlier_policy_for(name),
        )

    def _quantize_attention_rotation_bank_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states_for_attention: torch.Tensor,
        scaling: float,
    ) -> CacheSegment:
        best_segment: CacheSegment | None = None
        best_error: float | None = None
        for rotation_idx in range(self.config.rotation_bank_size):
            segment = self._quantize_fixed_rotation_bank_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                rotation_bank_index=rotation_idx,
                quantizer_kind="attention_rotation_bank_mse",
            )
            decoded = self._decode_segment(segment, name=name, layer_idx=layer_idx)
            error = self._attention_auto_mse_error(
                states,
                decoded,
                name=name,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
            if best_error is None or error < best_error:
                best_error = error
                best_segment = segment
        if best_segment is None:
            raise RuntimeError("attention_rotation_bank_mse did not produce a candidate segment")
        return best_segment

    def _quantize_segment_rotation_bank_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
    ) -> CacheSegment:
        best_segment: CacheSegment | None = None
        best_error: float | None = None
        states_f = states.detach().to(dtype=torch.float32)
        for rotation_idx in range(self.config.rotation_bank_size):
            segment = self._quantize_fixed_rotation_bank_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
                rotation_bank_index=rotation_idx,
                quantizer_kind="segment_rotation_bank_mse",
            )
            decoded = self._decode_segment(segment, name=name, layer_idx=layer_idx).to(dtype=torch.float32)
            error = float(torch.mean((states_f - decoded) ** 2).item())
            if best_error is None or error < best_error:
                best_error = error
                best_segment = segment
        if best_segment is None:
            raise RuntimeError("segment_rotation_bank_mse did not produce a candidate segment")
        return best_segment

    def _score_rotated_outlier_gain(
        self,
        gain: torch.Tensor,
        *,
        norms: torch.Tensor,
        quantizer: TurboQuantMSE,
        original_shape: tuple[int, ...],
        name: str,
        query_states: torch.Tensor | None,
        key_states_for_attention: torch.Tensor | None,
        scaling: float,
        attention_aware: bool,
    ) -> torch.Tensor:
        gain = gain.clamp_min(0.0).reshape(original_shape).to(dtype=torch.float32)
        if (
            not attention_aware
            or query_states is None
            or key_states_for_attention is None
            or len(original_shape) != 4
            or query_states.ndim != 4
            or key_states_for_attention.ndim != 4
            or key_states_for_attention.shape[-2] != original_shape[-2]
        ):
            return gain.mean(dim=tuple(range(gain.ndim - 1)))

        key_scores = self._grouped_attention_scores(query_states, key_states_for_attention, scaling=scaling)
        probs = self._causal_attention_probs(key_scores, key_len=key_states_for_attention.shape[-2]).detach()
        norm_weight = norms.reshape(original_shape[:-1]).to(dtype=torch.float32).square()

        if name == "key":
            query_tokens = min(self.config.attention_error_query_tokens, query_states.shape[-2])
            query_focus = query_states[..., -query_tokens:, :]
            bsz, num_query_heads, _, head_dim = query_focus.shape
            num_key_heads = original_shape[1]
            if num_query_heads % num_key_heads != 0 or head_dim != original_shape[-1]:
                return gain.mean(dim=tuple(range(gain.ndim - 1)))
            num_groups = num_query_heads // num_key_heads
            query_grouped = query_focus.reshape(bsz, num_key_heads, num_groups, query_tokens, head_dim)
            query_rotated = query_grouped.to(dtype=quantizer.transform.dtype) @ quantizer.transform.T
            query_sensitivity = query_rotated.to(dtype=torch.float32).square().mean(dim=(2, 3))
            token_weight = probs.mean(dim=(2, 3))
            weighted = gain * norm_weight.unsqueeze(-1) * token_weight.unsqueeze(-1) * query_sensitivity.unsqueeze(-2)
            return weighted.mean(dim=(0, 1, 2))

        if name == "value":
            token_weight = probs.square().mean(dim=(2, 3))
            weighted = gain * norm_weight.unsqueeze(-1) * token_weight.unsqueeze(-1)
            return weighted.mean(dim=(0, 1, 2))

        return gain.mean(dim=tuple(range(gain.ndim - 1)))

    def _quantize_rotated_outlier_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
        base_quantizer_kind: str = "mse",
        segment_quantizer_kind: str = "rotated_outlier_mse",
        quantizer_name: str | None = None,
        query_states: torch.Tensor | None = None,
        key_states_for_attention: torch.Tensor | None = None,
        scaling: float = 1.0,
        attention_aware: bool = False,
    ) -> CacheSegment:
        regular_bits, outlier_bits, outlier_count = self._resolve_effective_bit_allocation(bits, states.shape[-1])
        if outlier_count == 0:
            return self._quantize_to_segment(
                states,
                bits=regular_bits,
                name=name,
                quantizer_kind=base_quantizer_kind,
                layer_idx=layer_idx,
            )

        original_shape = tuple(states.shape)
        dimension = original_shape[-1]
        flat = states.reshape(-1, dimension)
        flat_f = flat.detach().to(dtype=torch.float32)
        rotation_name = name if quantizer_name is None else quantizer_name

        regular_quantizer = self._get_quantizer(
            quantizer_kind=base_quantizer_kind,
            name=rotation_name,
            dimension=dimension,
            bits=regular_bits,
            layer_idx=layer_idx,
            device=states.device,
            dtype=states.dtype,
        )
        outlier_quantizer = self._get_quantizer(
            quantizer_kind=base_quantizer_kind,
            name=rotation_name,
            dimension=dimension,
            bits=outlier_bits,
            layer_idx=layer_idx,
            device=states.device,
            dtype=states.dtype,
        )
        if not isinstance(regular_quantizer, TurboQuantMSE) or not isinstance(outlier_quantizer, TurboQuantMSE):
            raise TypeError("rotated_outlier_mse expects TurboQuantMSE quantizers")

        norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True).clamp_min(torch.finfo(states.dtype).eps)
        unit = flat / norms
        rotated = unit @ regular_quantizer.transform.T
        regular_distances = torch.abs(rotated.unsqueeze(-1) - regular_quantizer.centroids)
        regular_indices_all = torch.argmin(regular_distances, dim=-1).to(torch.int16)
        outlier_distances = torch.abs(rotated.unsqueeze(-1) - outlier_quantizer.centroids)
        outlier_indices_all = torch.argmin(outlier_distances, dim=-1).to(torch.int16)

        regular_hat_all = regular_quantizer.centroids[regular_indices_all.to(dtype=torch.long)]
        outlier_hat_all = outlier_quantizer.centroids[outlier_indices_all.to(dtype=torch.long)]
        gain = (rotated.to(dtype=torch.float32) - regular_hat_all.to(dtype=torch.float32)).square() - (
            rotated.to(dtype=torch.float32) - outlier_hat_all.to(dtype=torch.float32)
        ).square()
        scores = self._score_rotated_outlier_gain(
            gain,
            norms=norms,
            quantizer=regular_quantizer,
            original_shape=original_shape,
            name=name,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
            attention_aware=attention_aware,
        )
        outlier_indices = torch.topk(scores, k=outlier_count, largest=True, sorted=False).indices.sort().values
        all_indices = torch.arange(dimension, device=states.device, dtype=torch.long)
        regular_mask = torch.ones(dimension, device=states.device, dtype=torch.bool)
        regular_mask[outlier_indices] = False
        regular_indices = all_indices[regular_mask]

        regular_code_indices = regular_indices_all.index_select(-1, regular_indices)
        outlier_code_indices = outlier_indices_all.index_select(-1, outlier_indices)

        rotated_hat = regular_hat_all.clone()
        rotated_hat.index_copy_(-1, outlier_indices, outlier_hat_all.index_select(-1, outlier_indices))
        x_hat_unit = rotated_hat @ regular_quantizer.inverse_transform.T
        numerator = (flat_f * x_hat_unit.to(dtype=torch.float32)).sum(dim=-1)
        denominator = x_hat_unit.to(dtype=torch.float32).square().sum(dim=-1).clamp_min(torch.finfo(torch.float32).eps)
        scales = (numerator / denominator).to(dtype=states.dtype)

        regular_norms = scales.reshape(original_shape[:-1]).detach()
        outlier_norms = regular_norms
        return RotatedOutlierMSESegment(
            packed_regular_indices=pack_indices(regular_code_indices, regular_bits),
            regular_norms=regular_norms,
            packed_outlier_indices=pack_indices(outlier_code_indices, outlier_bits),
            outlier_norms=outlier_norms,
            regular_indices=regular_indices.to(dtype=torch.int16),
            outlier_indices=outlier_indices.to(dtype=torch.int16),
            shape=original_shape,
            dtype=states.dtype,
            regular_bits=regular_bits,
            outlier_bits=outlier_bits,
            requested_bits=float(bits),
            quantizer_kind=segment_quantizer_kind,
        )

    def _build_shared_rotated_outlier_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
        regular_bits: int,
        outlier_bits: int,
        outlier_indices: torch.Tensor,
        regular_quantizer: TurboQuantMSE,
        outlier_quantizer: TurboQuantMSE,
    ) -> RotatedOutlierMSESegment:
        original_shape = tuple(states.shape)
        dimension = original_shape[-1]
        flat = states.reshape(-1, dimension)
        flat_f = flat.detach().to(dtype=torch.float32)
        norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True).clamp_min(torch.finfo(states.dtype).eps)
        unit = flat / norms
        rotated = unit @ regular_quantizer.transform.T
        regular_distances = torch.abs(rotated.unsqueeze(-1) - regular_quantizer.centroids)
        regular_indices_all = torch.argmin(regular_distances, dim=-1).to(torch.int16)
        outlier_distances = torch.abs(rotated.unsqueeze(-1) - outlier_quantizer.centroids)
        outlier_indices_all = torch.argmin(outlier_distances, dim=-1).to(torch.int16)

        regular_hat_all = regular_quantizer.centroids[regular_indices_all.to(dtype=torch.long)]
        outlier_hat_all = outlier_quantizer.centroids[outlier_indices_all.to(dtype=torch.long)]
        all_indices = torch.arange(dimension, device=states.device, dtype=torch.long)
        outlier_indices = outlier_indices.to(device=states.device, dtype=torch.long).sort().values
        regular_mask = torch.ones(dimension, device=states.device, dtype=torch.bool)
        regular_mask[outlier_indices] = False
        regular_indices = all_indices[regular_mask]

        regular_code_indices = regular_indices_all.index_select(-1, regular_indices)
        outlier_code_indices = outlier_indices_all.index_select(-1, outlier_indices)

        rotated_hat = regular_hat_all.clone()
        rotated_hat.index_copy_(-1, outlier_indices, outlier_hat_all.index_select(-1, outlier_indices))
        x_hat_unit = rotated_hat @ regular_quantizer.inverse_transform.T
        numerator = (flat_f * x_hat_unit.to(dtype=torch.float32)).sum(dim=-1)
        denominator = x_hat_unit.to(dtype=torch.float32).square().sum(dim=-1).clamp_min(torch.finfo(torch.float32).eps)
        scales = (numerator / denominator).to(dtype=states.dtype)

        norms_out = scales.reshape(original_shape[:-1]).detach()
        return RotatedOutlierMSESegment(
            packed_regular_indices=pack_indices(regular_code_indices, regular_bits),
            regular_norms=norms_out,
            packed_outlier_indices=pack_indices(outlier_code_indices, outlier_bits),
            outlier_norms=norms_out,
            regular_indices=regular_indices.to(dtype=torch.int16),
            outlier_indices=outlier_indices.to(dtype=torch.int16),
            shape=original_shape,
            dtype=states.dtype,
            regular_bits=regular_bits,
            outlier_bits=outlier_bits,
            requested_bits=float(bits),
            quantizer_kind="shared_rotated_outlier_mse",
        )

    def _shared_rotated_outlier_scores(
        self,
        states: torch.Tensor,
        *,
        regular_quantizer: TurboQuantMSE,
        outlier_quantizer: TurboQuantMSE,
    ) -> torch.Tensor:
        dimension = states.shape[-1]
        flat = states.reshape(-1, dimension)
        norms = torch.linalg.vector_norm(flat, dim=-1, keepdim=True).clamp_min(torch.finfo(states.dtype).eps)
        unit = flat / norms
        rotated = unit @ regular_quantizer.transform.T
        regular_distances = torch.abs(rotated.unsqueeze(-1) - regular_quantizer.centroids)
        regular_indices = torch.argmin(regular_distances, dim=-1)
        outlier_distances = torch.abs(rotated.unsqueeze(-1) - outlier_quantizer.centroids)
        outlier_indices = torch.argmin(outlier_distances, dim=-1)
        regular_hat = regular_quantizer.centroids[regular_indices]
        outlier_hat = outlier_quantizer.centroids[outlier_indices]
        gain = (rotated.to(dtype=torch.float32) - regular_hat.to(dtype=torch.float32)).square() - (
            rotated.to(dtype=torch.float32) - outlier_hat.to(dtype=torch.float32)
        ).square()
        return gain.clamp_min(0).reshape(states.shape).mean(dim=tuple(range(states.ndim - 1)))

    def _quantize_shared_rotated_outlier_effective_to_segments(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *,
        key_bits: float,
        value_bits: float,
        layer_idx: int,
    ) -> tuple[CacheSegment, CacheSegment]:
        key_regular_bits, key_outlier_bits, key_outlier_count = self._resolve_effective_bit_allocation(
            key_bits, key_states.shape[-1]
        )
        value_regular_bits, value_outlier_bits, value_outlier_count = self._resolve_effective_bit_allocation(
            value_bits, value_states.shape[-1]
        )
        if (
            key_outlier_count == 0
            or value_outlier_count == 0
            or key_regular_bits != value_regular_bits
            or key_outlier_bits != value_outlier_bits
            or key_outlier_count != value_outlier_count
            or key_bits != value_bits
            or key_states.shape[-1] != value_states.shape[-1]
        ):
            return (
                self._quantize_effective_to_segment(
                    key_states,
                    bits=key_bits,
                    name="key",
                    quantizer_kind="rotated_outlier_mse",
                    layer_idx=layer_idx,
                ),
                self._quantize_effective_to_segment(
                    value_states,
                    bits=value_bits,
                    name="value",
                    quantizer_kind="rotated_outlier_mse",
                    layer_idx=layer_idx,
                ),
            )

        dimension = key_states.shape[-1]
        regular_quantizer = self._get_quantizer(
            quantizer_kind="mse",
            name="key",
            dimension=dimension,
            bits=key_regular_bits,
            layer_idx=layer_idx,
            device=key_states.device,
            dtype=key_states.dtype,
        )
        outlier_quantizer = self._get_quantizer(
            quantizer_kind="mse",
            name="key",
            dimension=dimension,
            bits=key_outlier_bits,
            layer_idx=layer_idx,
            device=key_states.device,
            dtype=key_states.dtype,
        )
        if not isinstance(regular_quantizer, TurboQuantMSE) or not isinstance(outlier_quantizer, TurboQuantMSE):
            raise TypeError("shared_rotated_outlier_mse expects TurboQuantMSE quantizers")

        key_scores = self._shared_rotated_outlier_scores(
            key_states,
            regular_quantizer=regular_quantizer,
            outlier_quantizer=outlier_quantizer,
        )
        value_scores = self._shared_rotated_outlier_scores(
            value_states,
            regular_quantizer=regular_quantizer,
            outlier_quantizer=outlier_quantizer,
        )
        eps = torch.finfo(torch.float32).eps
        key_scores = key_scores.clamp_min(0)
        value_scores = value_scores.clamp_min(0)
        key_scale = key_scores.mean().clamp_min(eps)
        value_scale = value_scores.mean().clamp_min(eps)
        shared_scores = key_scores / key_scale + value_scores / value_scale
        shared_indices = torch.topk(shared_scores, k=key_outlier_count, largest=True, sorted=False).indices.sort().values

        key_segment = self._build_shared_rotated_outlier_segment(
            key_states,
            bits=key_bits,
            name="key",
            layer_idx=layer_idx,
            regular_bits=key_regular_bits,
            outlier_bits=key_outlier_bits,
            outlier_indices=shared_indices,
            regular_quantizer=regular_quantizer,
            outlier_quantizer=outlier_quantizer,
        )
        value_segment = self._build_shared_rotated_outlier_segment(
            value_states,
            bits=value_bits,
            name="value",
            layer_idx=layer_idx,
            regular_bits=value_regular_bits,
            outlier_bits=value_outlier_bits,
            outlier_indices=shared_indices,
            regular_quantizer=regular_quantizer,
            outlier_quantizer=outlier_quantizer,
        )
        return key_segment, value_segment

    def _quantize_attention_adaptive_rotated_outlier_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
        query_states: torch.Tensor | None,
        key_states_for_attention: torch.Tensor | None,
        scaling: float,
        segment_quantizer_kind: str = "attention_adaptive_rotated_outlier_mse",
        quantizer_name: str | None = None,
    ) -> CacheSegment:
        baseline_segment = self._quantize_effective_to_segment(
            states,
            bits=bits,
            name=name,
            quantizer_kind="mse",
            layer_idx=layer_idx,
            allow_token_protection=False,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
        )
        candidate_segment = self._quantize_rotated_outlier_effective_to_segment(
            states,
            bits=bits,
            name=name,
            layer_idx=layer_idx,
            base_quantizer_kind="mse",
            segment_quantizer_kind=segment_quantizer_kind,
            quantizer_name=quantizer_name,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
            attention_aware=True,
        )
        baseline_hat = self._decode_segment(baseline_segment, name=name, layer_idx=layer_idx)
        candidate_hat = self._decode_segment(candidate_segment, name=name, layer_idx=layer_idx)
        baseline_error = self._attention_auto_mse_error(
            states,
            baseline_hat,
            name=name,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
        )
        candidate_error = self._attention_auto_mse_error(
            states,
            candidate_hat,
            name=name,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
        )
        if candidate_error < baseline_error:
            return candidate_segment
        return baseline_segment

    def _quantize_entropy_guarded_paired_rotated_outlier_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
        query_states: torch.Tensor | None,
        key_states_for_attention: torch.Tensor | None,
        scaling: float,
    ) -> CacheSegment:
        entropy_ratio = self._attention_entropy_ratio(
            query_states=query_states,
            key_states=key_states_for_attention,
            scaling=scaling,
            query_tokens=self.config.attention_error_query_tokens,
        )
        if entropy_ratio is None or entropy_ratio > self.config.attention_entropy_threshold:
            return self._quantize_effective_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind="mse",
                layer_idx=layer_idx,
                allow_token_protection=False,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        return self._quantize_rotated_outlier_effective_to_segment(
            states,
            bits=bits,
            name=name,
            layer_idx=layer_idx,
            base_quantizer_kind="mse",
            segment_quantizer_kind="entropy_guarded_paired_rotated_outlier_mse",
            quantizer_name="key",
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
            attention_aware=False,
        )

    def _quantize_regular_outlier_hadamard_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
    ) -> CacheSegment:
        regular_bits, outlier_bits, outlier_count = self._resolve_effective_bit_allocation(bits, states.shape[-1])
        if outlier_count == 0:
            return self._quantize_to_segment(
                states,
                bits=regular_bits,
                name=name,
                quantizer_kind="outlier_hadamard_mse",
                layer_idx=layer_idx,
            )
        outlier_indices = self._select_outlier_channels(
            states,
            outlier_count,
            regular_bits=regular_bits,
            outlier_bits=outlier_bits,
            name=name,
            quantizer_kind="mse",
            layer_idx=layer_idx,
        )
        original_shape = tuple(states.shape)
        all_indices = torch.arange(states.shape[-1], device=states.device, dtype=torch.long)
        regular_mask = torch.ones(states.shape[-1], device=states.device, dtype=torch.bool)
        regular_mask[outlier_indices] = False
        regular_indices = all_indices[regular_mask]
        regular_segment = self._quantize_to_segment(
            states.index_select(-1, regular_indices),
            bits=regular_bits,
            name=f"{name}_regular",
            quantizer_kind="outlier_hadamard_mse",
            layer_idx=layer_idx,
        )
        outlier_segment = self._quantize_to_segment(
            states.index_select(-1, outlier_indices),
            bits=outlier_bits,
            name=f"{name}_outlier",
            quantizer_kind="mse",
            layer_idx=layer_idx,
        )
        return OutlierMSESegment(
            regular=regular_segment,
            outlier=outlier_segment,
            regular_indices=regular_indices.to(dtype=torch.int16),
            outlier_indices=outlier_indices.to(dtype=torch.int16),
            shape=original_shape,
            dtype=states.dtype,
            requested_bits=float(bits),
            policy=self._outlier_policy_for(name),
        )

    def _quantize_outlier_only_hadamard_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
    ) -> CacheSegment:
        regular_bits, outlier_bits, outlier_count = self._resolve_effective_bit_allocation(bits, states.shape[-1])
        if outlier_count == 0:
            return self._quantize_to_segment(
                states,
                bits=regular_bits,
                name=name,
                quantizer_kind="mse",
                layer_idx=layer_idx,
            )
        outlier_indices = self._select_outlier_channels(
            states,
            outlier_count,
            regular_bits=regular_bits,
            outlier_bits=outlier_bits,
            name=name,
            quantizer_kind="mse",
            layer_idx=layer_idx,
        )
        original_shape = tuple(states.shape)
        all_indices = torch.arange(states.shape[-1], device=states.device, dtype=torch.long)
        regular_mask = torch.ones(states.shape[-1], device=states.device, dtype=torch.bool)
        regular_mask[outlier_indices] = False
        regular_indices = all_indices[regular_mask]
        regular_segment = self._quantize_to_segment(
            states.index_select(-1, regular_indices),
            bits=regular_bits,
            name=f"{name}_regular",
            quantizer_kind="mse",
            layer_idx=layer_idx,
        )
        outlier_segment = self._quantize_to_segment(
            states.index_select(-1, outlier_indices),
            bits=outlier_bits,
            name=f"{name}_outlier",
            quantizer_kind="outlier_hadamard_mse",
            layer_idx=layer_idx,
        )
        return OutlierMSESegment(
            regular=regular_segment,
            outlier=outlier_segment,
            regular_indices=regular_indices.to(dtype=torch.int16),
            outlier_indices=outlier_indices.to(dtype=torch.int16),
            shape=original_shape,
            dtype=states.dtype,
            requested_bits=float(bits),
            policy=self._outlier_policy_for(name),
        )

    def _quantize_regular_attention_scale_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
        query_states: torch.Tensor | None,
        key_states_for_attention: torch.Tensor | None,
        scaling: float,
    ) -> CacheSegment:
        regular_bits, outlier_bits, outlier_count = self._resolve_effective_bit_allocation(bits, states.shape[-1])
        if outlier_count == 0:
            return self._quantize_attention_scale_to_segment(
                states,
                bits=regular_bits,
                name=name,
                layer_idx=layer_idx,
                query_states=query_states,
                scaling=scaling,
            )
        outlier_indices = self._select_outlier_channels(
            states,
            outlier_count,
            regular_bits=regular_bits,
            outlier_bits=outlier_bits,
            name=name,
            quantizer_kind="mse",
            layer_idx=layer_idx,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
        )
        original_shape = tuple(states.shape)
        all_indices = torch.arange(states.shape[-1], device=states.device, dtype=torch.long)
        regular_mask = torch.ones(states.shape[-1], device=states.device, dtype=torch.bool)
        regular_mask[outlier_indices] = False
        regular_indices = all_indices[regular_mask]
        regular_segment = self._quantize_attention_scale_to_segment(
            states.index_select(-1, regular_indices),
            bits=regular_bits,
            name=f"{name}_regular",
            layer_idx=layer_idx,
            query_states=query_states.index_select(-1, regular_indices) if query_states is not None else None,
            scaling=scaling,
        )
        outlier_segment = self._quantize_to_segment(
            states.index_select(-1, outlier_indices),
            bits=outlier_bits,
            name=f"{name}_outlier",
            quantizer_kind="mse",
            layer_idx=layer_idx,
        )
        return OutlierMSESegment(
            regular=regular_segment,
            outlier=outlier_segment,
            regular_indices=regular_indices.to(dtype=torch.int16),
            outlier_indices=outlier_indices.to(dtype=torch.int16),
            shape=original_shape,
            dtype=states.dtype,
            requested_bits=float(bits),
            policy=self._outlier_policy_for(name),
        )

    def _quantize_adaptive_regular_gain_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
        query_states: torch.Tensor | None,
        key_states_for_attention: torch.Tensor | None,
        scaling: float,
    ) -> CacheSegment:
        baseline_segment = self._quantize_effective_to_segment(
            states,
            bits=bits,
            name=name,
            quantizer_kind="mse",
            layer_idx=layer_idx,
            allow_token_protection=False,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
        )
        gain_segment = self._quantize_effective_to_segment(
            states,
            bits=bits,
            name=name,
            quantizer_kind="regular_gain_mse",
            layer_idx=layer_idx,
            allow_token_protection=False,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
        )
        states_f = states.detach().to(dtype=torch.float32)
        baseline_hat = self._decode_segment(baseline_segment, name=name, layer_idx=layer_idx).to(dtype=torch.float32)
        gain_hat = self._decode_segment(gain_segment, name=name, layer_idx=layer_idx).to(dtype=torch.float32)
        baseline_error = torch.mean((states_f - baseline_hat) ** 2)
        gain_error = torch.mean((states_f - gain_hat) ** 2)
        if bool(gain_error < baseline_error):
            return gain_segment
        return baseline_segment

    def _quantize_attention_adaptive_regular_gain_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
        query_states: torch.Tensor | None,
        key_states_for_attention: torch.Tensor | None,
        scaling: float,
    ) -> CacheSegment:
        baseline_segment = self._quantize_effective_to_segment(
            states,
            bits=bits,
            name=name,
            quantizer_kind="mse",
            layer_idx=layer_idx,
            allow_token_protection=False,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
        )
        gain_segment = self._quantize_effective_to_segment(
            states,
            bits=bits,
            name=name,
            quantizer_kind="regular_gain_mse",
            layer_idx=layer_idx,
            allow_token_protection=False,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
        )
        baseline_hat = self._decode_segment(baseline_segment, name=name, layer_idx=layer_idx)
        gain_hat = self._decode_segment(gain_segment, name=name, layer_idx=layer_idx)
        baseline_error = self._attention_auto_mse_error(
            states,
            baseline_hat,
            name=name,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
        )
        gain_error = self._attention_auto_mse_error(
            states,
            gain_hat,
            name=name,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
        )
        if gain_error < baseline_error:
            return gain_segment
        return baseline_segment

    def _quantize_rms_preconditioned_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: int,
        name: str,
        layer_idx: int,
        inner_quantizer_kind: str,
    ) -> CacheSegment:
        if states.shape[-2] <= 1:
            return self._quantize_to_segment(
                states,
                bits=bits,
                name=f"{name}_rms_base",
                quantizer_kind=inner_quantizer_kind,
                layer_idx=layer_idx,
            )
        rms = states.detach().to(dtype=torch.float32).square().mean(dim=tuple(range(states.ndim - 1))).sqrt()
        rms = rms.clamp_min(torch.finfo(torch.float32).eps)
        geo = torch.exp(torch.log(rms).mean()).clamp_min(torch.finfo(torch.float32).eps)
        scales = (rms / geo).to(device=states.device, dtype=states.dtype)
        residual_segment = self._quantize_to_segment(
            states / scales,
            bits=bits,
            name=f"{name}_rms",
            quantizer_kind=inner_quantizer_kind,
            layer_idx=layer_idx,
        )
        return RmsScaledSegment(
            scales=scales.detach(),
            residual=residual_segment,
            shape=tuple(states.shape),
            dtype=states.dtype,
            bits=bits,
        )

    @staticmethod
    def _head_rotation_index(name: str) -> int:
        marker = "_head"
        if marker not in name:
            return 0
        tail = name.split(marker, 1)[1]
        digits = []
        for char in tail:
            if not char.isdigit():
                break
            digits.append(char)
        if not digits:
            return 0
        return int("".join(digits)) + 1

    def _quantize_head_rotation_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
    ) -> CacheSegment:
        if states.ndim != 4 or states.shape[-3] <= 1:
            return self._quantize_effective_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind="mse",
                layer_idx=layer_idx,
                allow_token_protection=False,
            )

        head_segments: list[CacheSegment] = []
        num_heads = states.shape[-3]
        for head_idx in range(num_heads):
            head_states = states.index_select(-3, torch.tensor([head_idx], device=states.device)).squeeze(-3)
            head_segments.append(
                self._quantize_effective_to_segment(
                    head_states,
                    bits=bits,
                    name=f"{name}_head{head_idx}",
                    quantizer_kind="mse",
                    layer_idx=layer_idx,
                    allow_token_protection=False,
                )
            )
        return HeadRotationMSESegment(
            head_segments=tuple(head_segments),
            shape=tuple(states.shape),
            dtype=states.dtype,
            bits=float(bits),
        )

    def _head_hadamard_matrix(
        self,
        states: torch.Tensor,
        *,
        name: str,
        layer_idx: int,
    ) -> torch.Tensor | None:
        if states.ndim != 4:
            return None
        num_heads = states.shape[-3]
        if num_heads <= 1 or num_heads & (num_heads - 1):
            return None
        seed = self.config.seed + 1_000 * layer_idx + (0 if name == "key" else 500) + 31_337
        return hadamard_orthogonal(num_heads, device=states.device, dtype=states.dtype, seed=seed)

    def _apply_head_hadamard_transform(self, states: torch.Tensor, *, name: str, layer_idx: int) -> torch.Tensor:
        transform = self._head_hadamard_matrix(states, name=name, layer_idx=layer_idx)
        if transform is None:
            return states
        return torch.einsum("ij,bjsd->bisd", transform, states)

    def _inverse_head_hadamard_transform(self, states: torch.Tensor, *, name: str, layer_idx: int) -> torch.Tensor:
        transform = self._head_hadamard_matrix(states, name=name, layer_idx=layer_idx)
        if transform is None:
            return states
        return torch.einsum("ji,bjsd->bisd", transform, states)

    def _quantize_head_hadamard_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
    ) -> CacheSegment:
        if self._head_hadamard_matrix(states, name=name, layer_idx=layer_idx) is None:
            return self._quantize_effective_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind="mse",
                layer_idx=layer_idx,
                allow_token_protection=False,
            )
        transformed = self._apply_head_hadamard_transform(states, name=name, layer_idx=layer_idx)
        residual = self._quantize_effective_to_segment(
            transformed,
            bits=bits,
            name=f"{name}_head_hadamard",
            quantizer_kind="mse",
            layer_idx=layer_idx,
            allow_token_protection=False,
        )
        return HeadHadamardSegment(
            residual=residual,
            shape=tuple(states.shape),
            dtype=states.dtype,
            bits=float(bits),
        )

    def _hadamard_residual_matrix(
        self,
        dimension: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
        layer_idx: int,
    ) -> torch.Tensor:
        if dimension & (dimension - 1):
            raise ValueError("hadamard_residual_mse requires a power-of-two head dimension")
        seed = self.config.seed + 1_000 * layer_idx + (0 if name == "key" else 500) + 41_771
        return hadamard_orthogonal(dimension, device=device, dtype=dtype, seed=seed)

    def _quantize_hadamard_residual_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
    ) -> CacheSegment:
        base_bits, _, residual_count = self._resolve_effective_bit_allocation(bits, states.shape[-1])
        if residual_count == 0:
            return self._quantize_to_segment(
                states,
                bits=base_bits,
                name=name,
                quantizer_kind="mse",
                layer_idx=layer_idx,
            )
        dimension = states.shape[-1]
        if dimension & (dimension - 1):
            return self._quantize_effective_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind="mse",
                layer_idx=layer_idx,
                allow_token_protection=False,
            )
        base_segment = self._quantize_to_segment(
            states,
            bits=base_bits,
            name=f"{name}_hadamard_residual_base",
            quantizer_kind="mse",
            layer_idx=layer_idx,
        )
        base_hat = self._decode_segment(base_segment, name=f"{name}_hadamard_residual_base", layer_idx=layer_idx)
        residual = states.detach().to(dtype=torch.float32) - base_hat.detach().to(dtype=torch.float32)
        flat_residual = residual.reshape(-1, dimension)
        residual_norms = torch.linalg.vector_norm(flat_residual, dim=-1, keepdim=True)
        safe_norms = residual_norms.clamp_min(torch.finfo(torch.float32).eps)
        residual_unit = flat_residual / safe_norms
        transform = self._hadamard_residual_matrix(
            dimension,
            device=states.device,
            dtype=states.dtype,
            name=name,
            layer_idx=layer_idx,
        )
        coeffs = residual_unit.to(device=states.device, dtype=states.dtype) @ transform.T
        scores = coeffs.detach().abs().to(dtype=torch.float32).mean(dim=0)
        residual_indices = torch.topk(scores, k=residual_count, largest=True, sorted=False).indices
        selected = coeffs.index_select(-1, residual_indices)
        signs = (selected >= 0).to(dtype=torch.int16)
        scales = selected.detach().abs().to(dtype=torch.float32).mean(dim=-1)
        scales = (scales * residual_norms.squeeze(-1)).reshape(states.shape[:-1]).to(device=states.device, dtype=states.dtype)
        return HadamardResidualSegment(
            base=base_segment,
            packed_residual_signs=pack_indices(signs.reshape(*states.shape[:-1], residual_count), 1),
            residual_scales=scales.detach(),
            residual_indices=residual_indices.to(dtype=torch.int16).detach(),
            shape=tuple(states.shape),
            dtype=states.dtype,
            base_bits=base_bits,
            requested_bits=float(bits),
        )

    def _quantize_attention_hadamard_residual_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
        query_states: torch.Tensor | None,
        key_states_for_attention: torch.Tensor | None,
        scaling: float,
    ) -> CacheSegment:
        candidate_kinds = ("mse", "hadamard_residual_mse")
        best_segment: CacheSegment | None = None
        best_error: float | None = None
        for candidate_kind in candidate_kinds:
            segment = self._quantize_effective_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind=candidate_kind,
                layer_idx=layer_idx,
                allow_token_protection=False,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
            decoded = self._decode_segment(segment, name=name, layer_idx=layer_idx)
            error = self._attention_auto_mse_error(
                states,
                decoded,
                name=name,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
            if best_error is None or error < best_error:
                best_error = error
                best_segment = segment
        if best_segment is None:
            raise RuntimeError("attention_hadamard_residual_mse did not produce a candidate segment")
        return best_segment

    def _quantize_attention_weighted_hadamard_residual_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
        query_states: torch.Tensor | None,
        key_states_for_attention: torch.Tensor | None,
        scaling: float,
    ) -> CacheSegment:
        base_bits, _, residual_count = self._resolve_effective_bit_allocation(bits, states.shape[-1])
        if residual_count == 0:
            return self._quantize_to_segment(
                states,
                bits=base_bits,
                name=name,
                quantizer_kind="mse",
                layer_idx=layer_idx,
            )
        dimension = states.shape[-1]
        if (
            name != "key"
            or query_states is None
            or states.ndim != 4
            or query_states.ndim != 4
            or dimension & (dimension - 1)
            or query_states.shape[0] != states.shape[0]
            or query_states.shape[1] % states.shape[1] != 0
            or query_states.shape[-1] != dimension
        ):
            return self._quantize_hadamard_residual_effective_to_segment(
                states,
                bits=bits,
                name=name,
                layer_idx=layer_idx,
            )

        base_segment = self._quantize_to_segment(
            states,
            bits=base_bits,
            name=f"{name}_attention_weighted_hadamard_residual_base",
            quantizer_kind="mse",
            layer_idx=layer_idx,
        )
        base_hat = self._decode_segment(
            base_segment,
            name=f"{name}_attention_weighted_hadamard_residual_base",
            layer_idx=layer_idx,
        )
        residual = states.detach().to(dtype=torch.float32) - base_hat.detach().to(dtype=torch.float32)
        flat_residual = residual.reshape(-1, dimension)
        residual_norms = torch.linalg.vector_norm(flat_residual, dim=-1, keepdim=True)
        safe_norms = residual_norms.clamp_min(torch.finfo(torch.float32).eps)
        residual_unit = flat_residual / safe_norms
        transform = self._hadamard_residual_matrix(
            dimension,
            device=states.device,
            dtype=states.dtype,
            name=name,
            layer_idx=layer_idx,
        )
        coeffs = residual_unit.to(device=states.device, dtype=states.dtype) @ transform.T
        coeff_abs = coeffs.reshape(states.shape).detach().abs().to(dtype=torch.float32)

        query_tokens = min(self.config.attention_error_query_tokens, query_states.shape[-2])
        query_focus = query_states[..., -query_tokens:, :]
        bsz, num_query_heads, _, head_dim = query_focus.shape
        num_key_heads = states.shape[1]
        num_groups = num_query_heads // num_key_heads
        query_grouped = query_focus.reshape(bsz, num_key_heads, num_groups, query_tokens, head_dim)
        query_coeff_abs = (query_grouped.to(dtype=states.dtype) @ transform.T).detach().abs().to(dtype=torch.float32)

        key_reference = key_states_for_attention if key_states_for_attention is not None else states
        if key_reference.shape != states.shape:
            key_reference = states
        key_scores = self._grouped_attention_scores(query_states, key_reference, scaling=scaling)
        probs = self._causal_attention_probs(key_scores, key_len=states.shape[-2]).detach().to(dtype=torch.float32)
        scores = torch.einsum("bhgqs,bhgqd,bhsd->d", probs, query_coeff_abs, coeff_abs)
        residual_indices = torch.topk(scores, k=residual_count, largest=True, sorted=False).indices.sort().values
        selected = coeffs.index_select(-1, residual_indices)
        signs = (selected >= 0).to(dtype=torch.int16)
        scales = selected.detach().abs().to(dtype=torch.float32).mean(dim=-1)
        scales = (scales * residual_norms.squeeze(-1)).reshape(states.shape[:-1]).to(device=states.device, dtype=states.dtype)
        return HadamardResidualSegment(
            base=base_segment,
            packed_residual_signs=pack_indices(signs.reshape(*states.shape[:-1], residual_count), 1),
            residual_scales=scales.detach(),
            residual_indices=residual_indices.to(dtype=torch.int16).detach(),
            shape=tuple(states.shape),
            dtype=states.dtype,
            base_bits=base_bits,
            requested_bits=float(bits),
        )

    def _quantize_attention_outlier_hadamard_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
        query_states: torch.Tensor | None,
        key_states_for_attention: torch.Tensor | None,
        scaling: float,
    ) -> CacheSegment:
        candidate_kinds = ("mse", "outlier_hadamard_mse")
        best_segment: CacheSegment | None = None
        best_error: float | None = None
        for candidate_kind in candidate_kinds:
            segment = self._quantize_effective_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind=candidate_kind,
                layer_idx=layer_idx,
                allow_token_protection=False,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
            decoded = self._decode_segment(segment, name=name, layer_idx=layer_idx)
            error = self._attention_auto_mse_error(
                states,
                decoded,
                name=name,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
            if best_error is None or error < best_error:
                best_error = error
                best_segment = segment
        if best_segment is None:
            raise RuntimeError("attention_outlier_hadamard_mse did not produce a candidate segment")
        return best_segment

    def _quantize_entropy_guarded_outlier_hadamard_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
        query_states: torch.Tensor | None,
        key_states_for_attention: torch.Tensor | None,
        scaling: float,
    ) -> CacheSegment:
        if name != "value":
            return self._quantize_effective_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind="mse",
                layer_idx=layer_idx,
                allow_token_protection=False,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )

        entropy_ratio = self._attention_entropy_ratio(
            query_states=query_states,
            key_states=key_states_for_attention,
            scaling=scaling,
            query_tokens=self.config.attention_error_query_tokens,
        )
        if entropy_ratio is None or entropy_ratio > self.config.attention_entropy_threshold:
            return self._quantize_effective_to_segment(
                states,
                bits=bits,
                name=name,
                quantizer_kind="mse",
                layer_idx=layer_idx,
                allow_token_protection=False,
                query_states=query_states,
                key_states_for_attention=key_states_for_attention,
                scaling=scaling,
            )
        return self._quantize_effective_to_segment(
            states,
            bits=bits,
            name=name,
            quantizer_kind="outlier_hadamard_mse",
            layer_idx=layer_idx,
            allow_token_protection=False,
            query_states=query_states,
            key_states_for_attention=key_states_for_attention,
            scaling=scaling,
        )

    def _quantize_rms_regular_gain_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        layer_idx: int,
    ) -> CacheSegment:
        regular_bits, outlier_bits, outlier_count = self._resolve_effective_bit_allocation(bits, states.shape[-1])
        if outlier_count == 0:
            return self._quantize_rms_preconditioned_to_segment(
                states,
                bits=regular_bits,
                name=name,
                layer_idx=layer_idx,
                inner_quantizer_kind="mse",
            )
        outlier_indices = self._select_outlier_channels(
            states,
            outlier_count,
            regular_bits=regular_bits,
            outlier_bits=outlier_bits,
            name=name,
            quantizer_kind="mse",
            layer_idx=layer_idx,
        )
        original_shape = tuple(states.shape)
        all_indices = torch.arange(states.shape[-1], device=states.device, dtype=torch.long)
        regular_mask = torch.ones(states.shape[-1], device=states.device, dtype=torch.bool)
        regular_mask[outlier_indices] = False
        regular_indices = all_indices[regular_mask]
        regular_segment = self._quantize_rms_preconditioned_to_segment(
            states.index_select(-1, regular_indices),
            bits=regular_bits,
            name=f"{name}_regular",
            layer_idx=layer_idx,
            inner_quantizer_kind="gain_mse",
        )
        outlier_segment = self._quantize_rms_preconditioned_to_segment(
            states.index_select(-1, outlier_indices),
            bits=outlier_bits,
            name=f"{name}_outlier",
            layer_idx=layer_idx,
            inner_quantizer_kind="mse",
        )
        return OutlierMSESegment(
            regular=regular_segment,
            outlier=outlier_segment,
            regular_indices=regular_indices.to(dtype=torch.int16),
            outlier_indices=outlier_indices.to(dtype=torch.int16),
            shape=original_shape,
            dtype=states.dtype,
            requested_bits=float(bits),
            policy=self._outlier_policy_for(name),
        )

    def _configured_key_bits(self, layer_idx: int) -> float:
        return self._configured_bits(self.config.key_bits, self.config.layer_key_bits, layer_idx, "key")

    def _configured_value_bits(self, layer_idx: int) -> float:
        return self._configured_bits(self.config.value_bits, self.config.layer_value_bits, layer_idx, "value")

    def _configured_key_quantizer(self, layer_idx: int) -> str:
        return self._configured_quantizer(self.config.key_quantizer, self.config.layer_key_quantizers, layer_idx, "key")

    def _configured_value_quantizer(self, layer_idx: int) -> str:
        return self._configured_quantizer(self.config.value_quantizer, self.config.layer_value_quantizers, layer_idx, "value")

    def _configured_rotation_index(self, layer_idx: int, name: str) -> int:
        if name == "key" or name.startswith("key_"):
            schedule = self.config.layer_key_rotation_indices
            schedule_name = "key"
        elif name == "value" or name.startswith("value_"):
            schedule = self.config.layer_value_rotation_indices
            schedule_name = "value"
        else:
            schedule = None
            schedule_name = name
        if schedule is None:
            return 0
        if layer_idx >= len(schedule):
            raise ValueError(f"{schedule_name} layer rotation schedule has {len(schedule)} entries, missing layer {layer_idx}")
        return int(schedule[layer_idx])

    def _calibrated_rotation_matrix(
        self,
        layer_idx: int,
        name: str,
        dimension: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if name == "key" or name.startswith("key_"):
            schedule = self.config.layer_key_rotation_matrices
            schedule_name = "key"
        elif name == "value" or name.startswith("value_"):
            schedule = self.config.layer_value_rotation_matrices
            schedule_name = "value"
        else:
            schedule = None
            schedule_name = name
        if schedule is None:
            seed = self.config.seed + 1_000 * layer_idx + (0 if schedule_name == "key" else 500)
            return random_orthogonal(dimension, device=device, dtype=dtype, seed=seed)
        if layer_idx >= len(schedule):
            raise ValueError(f"{schedule_name} calibrated rotation schedule has {len(schedule)} entries, missing layer {layer_idx}")
        matrix = torch.tensor(schedule[layer_idx], device=device, dtype=dtype)
        if tuple(matrix.shape) != (dimension, dimension):
            raise ValueError(
                f"{schedule_name} calibrated rotation for layer {layer_idx} must have shape {(dimension, dimension)}, "
                f"got {tuple(matrix.shape)}"
            )
        return matrix

    @staticmethod
    def _configured_bits(default_bits: float, schedule: tuple[float, ...] | None, layer_idx: int, name: str) -> float:
        if schedule is None:
            return default_bits
        if layer_idx >= len(schedule):
            raise ValueError(f"{name} layer bit schedule has {len(schedule)} entries, missing layer {layer_idx}")
        return float(schedule[layer_idx])

    @staticmethod
    def _configured_quantizer(default_quantizer: str, schedule: tuple[str, ...] | None, layer_idx: int, name: str) -> str:
        if schedule is None:
            return default_quantizer
        if layer_idx >= len(schedule):
            raise ValueError(f"{name} layer quantizer schedule has {len(schedule)} entries, missing layer {layer_idx}")
        return schedule[layer_idx]

    def _materialize_segments(self, segments: list[CacheSegment], *, name: str, layer_idx: int) -> torch.Tensor:
        if not segments:
            return torch.tensor([])
        tensors = [self._decode_segment(segment, name=name, layer_idx=layer_idx) for segment in segments]
        if len(tensors) == 1:
            return tensors[0]
        return torch.cat(tensors, dim=-2)

    def _materialize_layer(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.fast_materialized_eval:
            key_cache = self._materialized_key_cache[layer_idx]
            value_cache = self._materialized_value_cache[layer_idx]
            if key_cache is not None and value_cache is not None:
                return key_cache, value_cache
        return (
            self._materialize_segments(self.key_cache[layer_idx], name="key", layer_idx=layer_idx),
            self._materialize_segments(self.value_cache[layer_idx], name="value", layer_idx=layer_idx),
        )

    def _append_materialized_layer(
        self,
        layer_idx: int,
        key_segment: CacheSegment,
        value_segment: CacheSegment,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_states = self._decode_segment(key_segment, name="key", layer_idx=layer_idx)
        value_states = self._decode_segment(value_segment, name="value", layer_idx=layer_idx)
        previous_keys = self._materialized_key_cache[layer_idx]
        previous_values = self._materialized_value_cache[layer_idx]
        if previous_keys is not None:
            key_states = torch.cat([previous_keys, key_states], dim=-2)
        if previous_values is not None:
            value_states = torch.cat([previous_values, value_states], dim=-2)
        self._materialized_key_cache[layer_idx] = key_states.detach()
        self._materialized_value_cache[layer_idx] = value_states.detach()
        return self._materialized_key_cache[layer_idx], self._materialized_value_cache[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_layer(layer_idx)

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        is_prefill = key_states.shape[-2] > 1
        should_quantize = self.config.quantize_prefill if is_prefill else self.config.quantize_decode

        key_bits = self._configured_key_bits(layer_idx) if should_quantize else 16
        value_bits = self._configured_value_bits(layer_idx) if should_quantize else 16
        key_quantizer = self._configured_key_quantizer(layer_idx)
        value_quantizer = self._configured_value_quantizer(layer_idx)
        key_token_scores = None
        value_token_scores = None
        query_states = None
        scaling = 1.0
        if cache_kwargs is not None:
            key_token_scores = cache_kwargs.get("key_token_scores")
            value_token_scores = cache_kwargs.get("value_token_scores")
            query_states = cache_kwargs.get("query_states")
            scaling = float(cache_kwargs.get("scaling", 1.0))
            if self.config.token_protection_policy == "attention_error_budget":
                if query_states is None:
                    raise ValueError("attention_error_budget requires query_states in cache_kwargs")
                quant_bits = self.config.token_quant_bits or max(1, floor(min(key_bits, value_bits)))
                if self.config.token_protection_targets in {"both", "key"}:
                    key_token_scores = self._attention_quantization_error_scores(
                        key_states,
                        name="key",
                        quantizer_kind=key_quantizer,
                        layer_idx=layer_idx,
                        quant_bits=quant_bits,
                        query_states=query_states,
                        key_states=key_states,
                        scaling=scaling,
                    )
                if self.config.token_protection_targets in {"both", "value"}:
                    value_token_scores = self._attention_quantization_error_scores(
                        value_states,
                        name="value",
                        quantizer_kind=value_quantizer,
                        layer_idx=layer_idx,
                        quant_bits=quant_bits,
                        query_states=query_states,
                        key_states=key_states,
                        scaling=scaling,
                    )

        if should_quantize and key_quantizer == value_quantizer == "shared_rotated_outlier_mse":
            key_segment, value_segment = self._quantize_shared_rotated_outlier_effective_to_segments(
                key_states,
                value_states,
                key_bits=key_bits,
                value_bits=value_bits,
                layer_idx=layer_idx,
            )
        elif should_quantize and self.config.outlier_policy in {"joint_error_gain", "joint_attention_error_gain"}:
            key_segment, value_segment = self._quantize_joint_effective_to_segments(
                key_states,
                value_states,
                key_bits=key_bits,
                value_bits=value_bits,
                key_quantizer=key_quantizer,
                value_quantizer=value_quantizer,
                layer_idx=layer_idx,
                query_states=query_states,
                scaling=scaling,
                policy=self.config.outlier_policy,
            )
        else:
            key_segment = self._quantize_effective_to_segment(
                key_states,
                bits=key_bits,
                name="key",
                quantizer_kind=key_quantizer,
                layer_idx=layer_idx,
                token_scores=key_token_scores,
                query_states=query_states,
                key_states_for_attention=key_states,
                scaling=scaling,
            )
            value_segment = self._quantize_effective_to_segment(
                value_states,
                bits=value_bits,
                name="value",
                quantizer_kind=value_quantizer,
                layer_idx=layer_idx,
                token_scores=value_token_scores,
                query_states=query_states,
                key_states_for_attention=key_states,
                scaling=scaling,
            )

        self.key_cache[layer_idx].append(key_segment)
        self.value_cache[layer_idx].append(value_segment)
        self._seq_lengths[layer_idx] += key_states.shape[-2]

        if self.config.fast_materialized_eval:
            return self._append_materialized_layer(layer_idx, key_segment, value_segment)
        return self._materialize_layer(layer_idx)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            layer_idx = 0
        if layer_idx >= len(self._seq_lengths):
            return 0
        return self._seq_lengths[layer_idx]

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def storage_nbytes(self, layer_idx: Optional[int] = None) -> int:
        """Return bytes used by stored cache segments, excluding quantizer matrices."""

        layer_indices = range(len(self.key_cache)) if layer_idx is None else range(layer_idx, layer_idx + 1)
        total = 0
        for idx in layer_indices:
            if idx >= len(self.key_cache):
                continue
            total += sum(segment.nbytes for segment in self.key_cache[idx])
            total += sum(segment.nbytes for segment in self.value_cache[idx])
        return total

    def fast_materialized_nbytes(self, layer_idx: Optional[int] = None) -> int:
        """Return transient dense bytes used by the fast evaluation cache."""

        layer_indices = range(len(self.key_cache)) if layer_idx is None else range(layer_idx, layer_idx + 1)
        total = 0
        for idx in layer_indices:
            if idx >= len(self._materialized_key_cache):
                continue
            for tensor in (self._materialized_key_cache[idx], self._materialized_value_cache[idx]):
                if tensor is not None:
                    total += tensor.numel() * tensor.element_size()
        return total

    def materialized_nbytes(self, layer_idx: Optional[int] = None) -> int:
        """Return bytes the cache would use if stored as dense tensors."""

        layer_indices = range(len(self.key_cache)) if layer_idx is None else range(layer_idx, layer_idx + 1)
        total = 0
        for idx in layer_indices:
            if idx >= len(self.key_cache):
                continue
            for segment in self.key_cache[idx] + self.value_cache[idx]:
                if isinstance(segment, RawTensorSegment):
                    total += segment.nbytes
                else:
                    total += _numel(segment.shape) * torch.tensor([], dtype=segment.dtype).element_size()
        return total

    def compression_summary(self) -> dict[str, Any]:
        """Return a compact summary of stored segment types and bit allocation."""

        segment_types: dict[str, int] = {}
        effective_bits: list[float] = []
        outlier_counts: list[int] = []
        regular_bits: list[int] = []
        outlier_bits: list[int] = []
        for layer_segments in self.key_cache + self.value_cache:
            for segment in layer_segments:
                segment_type = type(segment).__name__
                segment_types[segment_type] = segment_types.get(segment_type, 0) + 1
                if isinstance(segment, OutlierMSESegment):
                    effective_bits.append(segment.effective_index_bits)
                    outlier_counts.append(int(segment.outlier_indices.numel()))
                    regular_bits.append(segment.regular_bits)
                    outlier_bits.append(segment.outlier_bits)
                elif isinstance(segment, RotatedOutlierMSESegment):
                    effective_bits.append(segment.effective_index_bits)
                    outlier_counts.append(int(segment.outlier_indices.numel()))
                    regular_bits.append(segment.regular_bits)
                    outlier_bits.append(segment.outlier_bits)
                elif isinstance(segment, HeadAdaptiveOutlierMSESegment):
                    effective_bits.append(segment.effective_index_bits)
                    for head_segment in segment.head_segments:
                        outlier_counts.append(int(head_segment.outlier_indices.numel()))
                        regular_bits.append(head_segment.regular_bits)
                        outlier_bits.append(head_segment.outlier_bits)
                elif isinstance(segment, TokenProtectedSegment):
                    effective_bits.append(segment.effective_token_bits)
                elif isinstance(segment, CenteredSegment):
                    effective_bits.append(float(segment.bits))
                elif isinstance(segment, OutlierHadamardSegment):
                    effective_bits.append(float(segment.bits))
                elif isinstance(segment, RmsScaledSegment):
                    effective_bits.append(float(segment.bits))
                elif isinstance(segment, VectorAdaptiveRotationSegment):
                    effective_bits.append(float(segment.bits))
                elif isinstance(segment, HeadRotationMSESegment):
                    effective_bits.append(float(segment.bits))
                    for head_segment in segment.head_segments:
                        head_type = type(head_segment).__name__
                        segment_types[head_type] = segment_types.get(head_type, 0) + 1
                        if isinstance(head_segment, OutlierMSESegment):
                            outlier_counts.append(int(head_segment.outlier_indices.numel()))
                            regular_bits.append(head_segment.regular_bits)
                            outlier_bits.append(head_segment.outlier_bits)
                        elif isinstance(head_segment, RotatedOutlierMSESegment):
                            outlier_counts.append(int(head_segment.outlier_indices.numel()))
                            regular_bits.append(head_segment.regular_bits)
                            outlier_bits.append(head_segment.outlier_bits)
                elif isinstance(segment, HeadHadamardSegment):
                    effective_bits.append(float(segment.bits))
                elif isinstance(segment, HadamardResidualSegment):
                    effective_bits.append(segment.effective_index_bits)
                elif isinstance(segment, (PackedMSESegment, PackedProdSegment, UniformAffineSegment)):
                    effective_bits.append(float(segment.bits))

        return {
            "segment_types": segment_types,
            "avg_effective_index_bits": sum(effective_bits) / len(effective_bits) if effective_bits else None,
            "outlier_policy": self.config.outlier_policy,
            "key_outlier_policy": self.config.key_outlier_policy,
            "value_outlier_policy": self.config.value_outlier_policy,
            "outlier_counts": sorted(set(outlier_counts)),
            "regular_bits": sorted(set(regular_bits)),
            "outlier_bits": sorted(set(outlier_bits)),
            "fast_materialized_eval": self.config.fast_materialized_eval,
            "layer_key_quantizers": list(self.config.layer_key_quantizers) if self.config.layer_key_quantizers is not None else None,
            "layer_value_quantizers": (
                list(self.config.layer_value_quantizers) if self.config.layer_value_quantizers is not None else None
            ),
            "layer_key_bits": list(self.config.layer_key_bits) if self.config.layer_key_bits is not None else None,
            "layer_value_bits": list(self.config.layer_value_bits) if self.config.layer_value_bits is not None else None,
            "layer_key_rotation_indices": (
                list(self.config.layer_key_rotation_indices) if self.config.layer_key_rotation_indices is not None else None
            ),
            "layer_value_rotation_indices": (
                list(self.config.layer_value_rotation_indices) if self.config.layer_value_rotation_indices is not None else None
            ),
            "layer_key_channel_scores": self.config.layer_key_channel_scores is not None,
            "layer_value_channel_scores": self.config.layer_value_channel_scores is not None,
            "layer_key_rotation_matrices": self.config.layer_key_rotation_matrices is not None,
            "layer_value_rotation_matrices": self.config.layer_value_rotation_matrices is not None,
            "sensitivity_score_power": self.config.sensitivity_score_power,
            "token_protection_policy": self.config.token_protection_policy,
            "protected_start_tokens": self.config.protected_start_tokens,
            "token_quant_bits": self.config.token_quant_bits,
            "token_protection_target_ratio": self.config.token_protection_target_ratio,
            "token_protection_targets": self.config.token_protection_targets,
            "attention_error_query_tokens": self.config.attention_error_query_tokens,
            "attention_entropy_threshold": self.config.attention_entropy_threshold,
            "rotation_bank_size": self.config.rotation_bank_size,
            "outlier_hadamard_block_size": self.config.outlier_hadamard_block_size,
        }

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple(self[layer_idx] for layer_idx in range(len(self)))

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        config: KVQuantConfig | None = None,
    ) -> "TurboQuantDynamicCache":
        cache = cls(config)
        if past_key_values is not None:
            for layer_idx, (key_states, value_states) in enumerate(past_key_values):
                cache.update(key_states, value_states, layer_idx)
        return cache

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for layer_idx in range(len(self.key_cache)):
            key_states, value_states = self._materialize_layer(layer_idx)
            key_states = key_states[..., :max_length, :]
            value_states = value_states[..., :max_length, :]
            self.key_cache[layer_idx] = [
                self._quantize_effective_to_segment(
                    key_states,
                    bits=self._configured_key_bits(layer_idx),
                    name="key",
                    quantizer_kind=self.config.key_quantizer,
                    layer_idx=layer_idx,
                )
            ]
            self.value_cache[layer_idx] = [
                self._quantize_effective_to_segment(
                    value_states,
                    bits=self._configured_value_bits(layer_idx),
                    name="value",
                    quantizer_kind=self.config.value_quantizer,
                    layer_idx=layer_idx,
                )
            ]
            self._seq_lengths[layer_idx] = max_length
            self._materialized_key_cache[layer_idx] = key_states.detach() if self.config.fast_materialized_eval else None
            self._materialized_value_cache[layer_idx] = value_states.detach() if self.config.fast_materialized_eval else None

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        for layer_idx in range(len(self.key_cache)):
            key_states, value_states = self._materialize_layer(layer_idx)
            key_states = key_states.index_select(0, beam_idx.to(key_states.device))
            value_states = value_states.index_select(0, beam_idx.to(value_states.device))
            self.key_cache[layer_idx] = [
                self._quantize_effective_to_segment(
                    key_states,
                    bits=self._configured_key_bits(layer_idx),
                    name="key",
                    quantizer_kind=self.config.key_quantizer,
                    layer_idx=layer_idx,
                )
            ]
            self.value_cache[layer_idx] = [
                self._quantize_effective_to_segment(
                    value_states,
                    bits=self._configured_value_bits(layer_idx),
                    name="value",
                    quantizer_kind=self.config.value_quantizer,
                    layer_idx=layer_idx,
                )
            ]
            self._materialized_key_cache[layer_idx] = key_states.detach() if self.config.fast_materialized_eval else None
            self._materialized_value_cache[layer_idx] = value_states.detach() if self.config.fast_materialized_eval else None


def make_kv_config_from_effective_bits(
    bits: float,
    *,
    seed: int = 0,
    codebook_grid_size: int = 20_001,
    key_quantizer: str = "mse",
    value_quantizer: str = "mse",
    effective_bit_allocation: str = "blend",
    outlier_policy: str = "dynamic_absmean",
    key_outlier_policy: str | None = None,
    value_outlier_policy: str | None = None,
    fast_materialized_eval: bool = False,
    layer_key_quantizers: tuple[str, ...] | None = None,
    layer_value_quantizers: tuple[str, ...] | None = None,
    layer_key_bits: tuple[float, ...] | None = None,
    layer_value_bits: tuple[float, ...] | None = None,
    layer_key_rotation_indices: tuple[int, ...] | None = None,
    layer_value_rotation_indices: tuple[int, ...] | None = None,
    layer_key_channel_scores: tuple[tuple[float, ...], ...] | None = None,
    layer_value_channel_scores: tuple[tuple[float, ...], ...] | None = None,
    layer_key_rotation_matrices: tuple[tuple[tuple[float, ...], ...], ...] | None = None,
    layer_value_rotation_matrices: tuple[tuple[tuple[float, ...], ...], ...] | None = None,
    sensitivity_score_power: float = 1.0,
    token_protection_policy: str = "none",
    protected_start_tokens: int = 4,
    token_quant_bits: int | None = None,
    token_protection_target_ratio: float | None = None,
    token_protection_targets: str = "both",
    attention_error_query_tokens: int = 1,
    attention_entropy_threshold: float = 0.80,
    rotation_bank_size: int = 4,
    outlier_hadamard_block_size: int = 16,
) -> KVQuantConfig:
    """Create a KV cache config for integer or fractional effective bits."""

    return KVQuantConfig(
        key_bits=bits,
        value_bits=bits,
        seed=seed,
        codebook_grid_size=codebook_grid_size,
        key_quantizer=key_quantizer,
        value_quantizer=value_quantizer,
        effective_bit_allocation=effective_bit_allocation,
        outlier_policy=outlier_policy,
        key_outlier_policy=key_outlier_policy,
        value_outlier_policy=value_outlier_policy,
        fast_materialized_eval=fast_materialized_eval,
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
        sensitivity_score_power=sensitivity_score_power,
        token_protection_policy=token_protection_policy,
        protected_start_tokens=protected_start_tokens,
        token_quant_bits=token_quant_bits,
        token_protection_target_ratio=token_protection_target_ratio,
        token_protection_targets=token_protection_targets,
        attention_error_query_tokens=attention_error_query_tokens,
        attention_entropy_threshold=attention_entropy_threshold,
        rotation_bank_size=rotation_bank_size,
        outlier_hadamard_block_size=outlier_hadamard_block_size,
    )
