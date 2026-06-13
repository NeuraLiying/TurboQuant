"""Transformers cache wrappers for TurboQuant KV experiments."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import ceil, floor
from operator import mul
from typing import Any, Optional

import torch
from transformers.cache_utils import Cache

from .core import MSEQuantized, ProdQuantized, TurboQuantMSE, TurboQuantProd

QUANTIZER_KINDS = {"mse", "prod"}
EFFECTIVE_BIT_ALLOCATION_POLICIES = {"blend", "quarter_high2"}


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
    fast_materialized_eval: bool = False

    def __post_init__(self) -> None:
        if self.key_quantizer not in QUANTIZER_KINDS:
            raise ValueError(f"unsupported key quantizer: {self.key_quantizer}")
        if self.value_quantizer not in QUANTIZER_KINDS:
            raise ValueError(f"unsupported value quantizer: {self.value_quantizer}")
        if self.effective_bit_allocation not in EFFECTIVE_BIT_ALLOCATION_POLICIES:
            raise ValueError(f"unsupported effective-bit allocation: {self.effective_bit_allocation}")


@dataclass
class PackedMSESegment:
    """Packed TurboQuantMSE representation for one cache update segment."""

    packed_indices: torch.Tensor
    norms: torch.Tensor
    shape: tuple[int, ...]
    bits: int
    dtype: torch.dtype

    @property
    def sequence_length(self) -> int:
        return self.shape[-2]

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
class RawTensorSegment:
    """Fallback segment for unquantized cache settings."""

    tensor: torch.Tensor

    @property
    def sequence_length(self) -> int:
        return self.tensor.shape[-2]

    @property
    def nbytes(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()


CacheSegment = PackedMSESegment | PackedProdSegment | OutlierMSESegment | RawTensorSegment


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
        self._quantizers: dict[tuple[str, str, int, torch.dtype, int, int, str, int], TurboQuantMSE | TurboQuantProd] = {}
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
    ) -> TurboQuantMSE | TurboQuantProd:
        device_index = device.index if device.index is not None else -1
        key = (quantizer_kind, name, dimension, dtype, bits, layer_idx, device.type, device_index)
        quantizer = self._quantizers.get(key)
        if quantizer is None:
            seed = self.config.seed + 1_000 * layer_idx + (0 if name == "key" else 500)
            quantizer_cls = TurboQuantProd if quantizer_kind == "prod" else TurboQuantMSE
            quantizer = quantizer_cls(
                dimension,
                bits,
                seed=seed,
                device=device,
                dtype=dtype,
                codebook_grid_size=self.config.codebook_grid_size,
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
        if isinstance(quantized, MSEQuantized):
            packed = pack_indices(quantized.indices, bits)
            return PackedMSESegment(
                packed_indices=packed,
                norms=quantized.norms.reshape(original_shape[:-1]).detach(),
                shape=original_shape,
                bits=bits,
                dtype=states.dtype,
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

    def _select_outlier_channels(self, states: torch.Tensor, outlier_count: int) -> torch.Tensor:
        if outlier_count <= 0:
            return torch.empty(0, device=states.device, dtype=torch.long)
        scores = states.detach().abs().to(dtype=torch.float32).mean(dim=tuple(range(states.ndim - 1)))
        return torch.topk(scores, k=outlier_count, largest=True, sorted=True).indices.sort().values

    def _quantize_effective_to_segment(
        self,
        states: torch.Tensor,
        *,
        bits: float,
        name: str,
        quantizer_kind: str,
        layer_idx: int,
    ) -> CacheSegment:
        regular_bits, outlier_bits, outlier_count = self._resolve_effective_bit_allocation(bits, states.shape[-1])
        if outlier_count == 0:
            return self._quantize_to_segment(
                states,
                bits=regular_bits,
                name=name,
                quantizer_kind=quantizer_kind,
                layer_idx=layer_idx,
            )

        original_shape = tuple(states.shape)
        all_indices = torch.arange(states.shape[-1], device=states.device, dtype=torch.long)
        outlier_indices = self._select_outlier_channels(states, outlier_count)
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
            policy=self.config.outlier_policy,
        )

    def _decode_segment(self, segment: CacheSegment, *, name: str, layer_idx: int) -> torch.Tensor:
        if isinstance(segment, RawTensorSegment):
            return segment.tensor
        if isinstance(segment, OutlierMSESegment):
            regular = self._decode_segment(segment.regular, name=f"{name}_regular", layer_idx=layer_idx)
            outlier = self._decode_segment(segment.outlier, name=f"{name}_outlier", layer_idx=layer_idx)
            output = torch.empty(segment.shape, device=regular.device, dtype=segment.dtype)
            output.index_copy_(-1, segment.regular_indices.to(device=regular.device, dtype=torch.long), regular)
            output.index_copy_(-1, segment.outlier_indices.to(device=regular.device, dtype=torch.long), outlier)
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
        quantizer = self._get_quantizer(
            quantizer_kind="mse",
            name=name,
            dimension=dimension,
            bits=segment.bits,
            layer_idx=layer_idx,
            device=segment.packed_indices.device,
            dtype=segment.dtype,
        )
        indices = unpack_indices(segment.packed_indices, bits=segment.bits, shape=segment.shape)
        quantized = MSEQuantized(indices=indices.reshape(-1, dimension), norms=segment.norms.reshape(-1))
        return quantizer.dequantize(quantized).reshape(segment.shape)

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

        key_bits = self.config.key_bits if should_quantize else 16
        value_bits = self.config.value_bits if should_quantize else 16
        key_segment = self._quantize_effective_to_segment(
            key_states,
            bits=key_bits,
            name="key",
            quantizer_kind=self.config.key_quantizer,
            layer_idx=layer_idx,
        )
        value_segment = self._quantize_effective_to_segment(
            value_states,
            bits=value_bits,
            name="value",
            quantizer_kind=self.config.value_quantizer,
            layer_idx=layer_idx,
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
                elif isinstance(segment, (PackedMSESegment, PackedProdSegment)):
                    effective_bits.append(float(segment.bits))

        return {
            "segment_types": segment_types,
            "avg_effective_index_bits": sum(effective_bits) / len(effective_bits) if effective_bits else None,
            "outlier_policy": self.config.outlier_policy,
            "outlier_counts": sorted(set(outlier_counts)),
            "regular_bits": sorted(set(regular_bits)),
            "outlier_bits": sorted(set(outlier_bits)),
            "fast_materialized_eval": self.config.fast_materialized_eval,
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
                    bits=self.config.key_bits,
                    name="key",
                    quantizer_kind=self.config.key_quantizer,
                    layer_idx=layer_idx,
                )
            ]
            self.value_cache[layer_idx] = [
                self._quantize_effective_to_segment(
                    value_states,
                    bits=self.config.value_bits,
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
                    bits=self.config.key_bits,
                    name="key",
                    quantizer_kind=self.config.key_quantizer,
                    layer_idx=layer_idx,
                )
            ]
            self.value_cache[layer_idx] = [
                self._quantize_effective_to_segment(
                    value_states,
                    bits=self.config.value_bits,
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
    fast_materialized_eval: bool = False,
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
        fast_materialized_eval=fast_materialized_eval,
    )
