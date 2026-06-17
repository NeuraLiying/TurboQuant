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
)

QUANTIZER_KINDS = {
    "mse",
    "unit_mse",
    "lsq_mse",
    "lsq_unit_mse",
    "gain_mse",
    "lowbit_gain_mse",
    "lowbit_half_gain_mse",
    "lowbit_clipped_gain_mse",
    "lowbit_selected_gain_mse",
    "dot_gain_mse",
    "hadamard_mse",
    "centered_mse",
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
    layer_key_channel_scores: tuple[tuple[float, ...], ...] | None = None
    layer_value_channel_scores: tuple[tuple[float, ...], ...] | None = None
    sensitivity_score_power: float = 1.0
    token_protection_policy: str = "none"
    protected_start_tokens: int = 4
    token_quant_bits: int | None = None
    token_protection_target_ratio: float | None = None
    token_protection_targets: str = "both"
    attention_error_query_tokens: int = 1

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
        if self.layer_key_channel_scores is not None and not self.layer_key_channel_scores:
            raise ValueError("layer_key_channel_scores must be non-empty when provided")
        if self.layer_value_channel_scores is not None and not self.layer_value_channel_scores:
            raise ValueError("layer_value_channel_scores must be non-empty when provided")
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
        return self.packed_indices.numel() * self.packed_indices.element_size() + self.norms.numel() * self.norms.element_size()


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
    | OutlierMSESegment
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
    ) -> TurboQuantMSE | TurboQuantProd | TurboQuantBlockMSE | TurboQuantLearnedBlockMSE:
        device_index = device.index if device.index is not None else -1
        key = (quantizer_kind, name, dimension, dtype, bits, layer_idx, device.type, device_index)
        quantizer = self._quantizers.get(key)
        if quantizer is None:
            seed = self.config.seed + 1_000 * layer_idx + (0 if name == "key" else 500)
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
            else:
                quantizer = TurboQuantMSE(
                    dimension,
                    bits,
                    seed=seed,
                    device=device,
                    dtype=dtype,
                    codebook_grid_size=self.config.codebook_grid_size,
                    project_unit_norm=quantizer_kind in {"unit_mse", "lsq_unit_mse"},
                    reconstruction_scale=(
                        "lsq"
                        if quantizer_kind in {"lsq_mse", "lsq_unit_mse"}
                        else "norm_gain"
                        if quantizer_kind == "gain_mse"
                        else "norm_gain"
                        if quantizer_kind == "lowbit_gain_mse" and bits <= 2
                        else "half_gain"
                        if quantizer_kind == "lowbit_half_gain_mse" and bits <= 2
                        else "clipped_gain"
                        if quantizer_kind == "lowbit_clipped_gain_mse" and bits <= 2
                        else "selected_gain"
                        if quantizer_kind == "lowbit_selected_gain_mse" and bits <= 2
                        else "dot_gain"
                        if quantizer_kind == "dot_gain_mse"
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
    ) -> CacheSegment:
        if bits >= 16:
            return RawTensorSegment(tensor=states.detach())
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
        if quantizer_kind in {"uniform_token", "uniform_channel"}:
            return self._quantize_uniform_to_segment(
                states,
                bits=bits,
                granularity="channel" if quantizer_kind == "uniform_channel" else "token",
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
        elif policy == "error_gain":
            scores = self._score_error_gain_channels(
                states,
                regular_bits=regular_bits,
                outlier_bits=outlier_bits,
                name=name,
                quantizer_kind=quantizer_kind,
                layer_idx=layer_idx,
            )
        elif policy == "attention_error_gain":
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
        regular_bits, outlier_bits, outlier_count = self._resolve_effective_bit_allocation(bits, states.shape[-1])
        if outlier_count == 0:
            return self._quantize_to_segment(
                states,
                bits=regular_bits,
                name=name,
                quantizer_kind=quantizer_kind,
                layer_idx=layer_idx,
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
            policy=self._outlier_policy_for(name),
        )

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
        if isinstance(segment, OutlierMSESegment):
            regular = self._decode_segment(segment.regular, name=f"{name}_regular", layer_idx=layer_idx)
            outlier = self._decode_segment(segment.outlier, name=f"{name}_outlier", layer_idx=layer_idx)
            output = torch.empty(segment.shape, device=regular.device, dtype=segment.dtype)
            output.index_copy_(-1, segment.regular_indices.to(device=regular.device, dtype=torch.long), regular)
            output.index_copy_(-1, segment.outlier_indices.to(device=regular.device, dtype=torch.long), outlier)
            return output
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

    def _configured_key_bits(self, layer_idx: int) -> float:
        return self._configured_bits(self.config.key_bits, self.config.layer_key_bits, layer_idx, "key")

    def _configured_value_bits(self, layer_idx: int) -> float:
        return self._configured_bits(self.config.value_bits, self.config.layer_value_bits, layer_idx, "value")

    def _configured_key_quantizer(self, layer_idx: int) -> str:
        return self._configured_quantizer(self.config.key_quantizer, self.config.layer_key_quantizers, layer_idx, "key")

    def _configured_value_quantizer(self, layer_idx: int) -> str:
        return self._configured_quantizer(self.config.value_quantizer, self.config.layer_value_quantizers, layer_idx, "value")

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
            "layer_key_channel_scores": self.config.layer_key_channel_scores is not None,
            "layer_value_channel_scores": self.config.layer_value_channel_scores is not None,
            "sensitivity_score_power": self.config.sensitivity_score_power,
            "token_protection_policy": self.config.token_protection_policy,
            "protected_start_tokens": self.config.protected_start_tokens,
            "token_quant_bits": self.config.token_quant_bits,
            "token_protection_target_ratio": self.config.token_protection_target_ratio,
            "token_protection_targets": self.config.token_protection_targets,
            "attention_error_query_tokens": self.config.attention_error_query_tokens,
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
    layer_key_channel_scores: tuple[tuple[float, ...], ...] | None = None,
    layer_value_channel_scores: tuple[tuple[float, ...], ...] | None = None,
    sensitivity_score_power: float = 1.0,
    token_protection_policy: str = "none",
    protected_start_tokens: int = 4,
    token_quant_bits: int | None = None,
    token_protection_target_ratio: float | None = None,
    token_protection_targets: str = "both",
    attention_error_query_tokens: int = 1,
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
        layer_key_channel_scores=layer_key_channel_scores,
        layer_value_channel_scores=layer_value_channel_scores,
        sensitivity_score_power=sensitivity_score_power,
        token_protection_policy=token_protection_policy,
        protected_start_tokens=protected_start_tokens,
        token_quant_bits=token_quant_bits,
        token_protection_target_ratio=token_protection_target_ratio,
        token_protection_targets=token_protection_targets,
        attention_error_query_tokens=attention_error_query_tokens,
    )
