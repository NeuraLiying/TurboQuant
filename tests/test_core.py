import torch

from turboquant import TurboQuantDynamicCache, TurboQuantMSE, TurboQuantProd
from turboquant.kv_cache import (
    OutlierMSESegment,
    PackedMSESegment,
    PackedProdSegment,
    make_kv_config_from_effective_bits,
    pack_indices,
    unpack_indices,
)


def test_mse_roundtrip_shapes_and_monotonic_error():
    torch.manual_seed(0)
    x = torch.randn(64, 32)
    x = x / x.norm(dim=-1, keepdim=True)

    errors = []
    for bits in [1, 2, 3]:
        quantizer = TurboQuantMSE(32, bits, seed=123, codebook_grid_size=20_001)
        q = quantizer.quantize(x)
        x_hat = quantizer.dequantize(q)
        assert q.indices.shape == x.shape
        assert x_hat.shape == x.shape
        errors.append(torch.mean(torch.sum((x - x_hat) ** 2, dim=-1)).item())

    assert errors[2] < errors[1] < errors[0]


def test_prod_shapes():
    torch.manual_seed(1)
    x = torch.randn(8, 32)
    y = torch.randn(8, 32)
    quantizer = TurboQuantProd(32, 2, seed=123, codebook_grid_size=20_001)
    q = quantizer.quantize(x)
    x_hat = quantizer.dequantize(q)
    assert q.mse.indices.shape == x.shape
    assert q.qjl.shape == x.shape
    assert q.residual_norms.shape == (8,)
    assert x_hat.shape == x.shape
    assert torch.isfinite((x_hat * y).sum(dim=-1)).all()


def test_turboquant_dynamic_cache_shapes():
    torch.manual_seed(2)
    cache = TurboQuantDynamicCache()
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)
    out_k, out_v = cache.update(key, value, layer_idx=0)
    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    next_k = torch.randn(1, 2, 1, 16)
    next_v = torch.randn(1, 2, 1, 16)
    out_k, out_v = cache.update(next_k, next_v, layer_idx=0)
    assert out_k.shape == (1, 2, 5, 16)
    assert out_v.shape == (1, 2, 5, 16)


def test_pack_indices_roundtrip_for_non_byte_aligned_bits():
    indices = torch.arange(35, dtype=torch.int16) % 8
    packed = pack_indices(indices, bits=3)
    restored = unpack_indices(packed, bits=3, shape=tuple(indices.shape))
    assert packed.dtype == torch.uint8
    assert packed.numel() == (indices.numel() * 3 + 7) // 8
    assert torch.equal(restored, indices)


def test_turboquant_dynamic_cache_stores_packed_segments():
    torch.manual_seed(3)
    cache = TurboQuantDynamicCache()
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    dense_nbytes = 2 * key.numel() * key.element_size()
    out_k, out_v = cache.update(key, value, layer_idx=0)

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    assert cache.get_seq_length() == 4
    assert isinstance(cache.key_cache[0][0], PackedMSESegment)
    assert isinstance(cache.value_cache[0][0], PackedMSESegment)
    assert cache.key_cache[0][0].packed_indices.dtype == torch.uint8
    assert cache.storage_nbytes() < dense_nbytes

    next_k = torch.randn(1, 2, 1, 16)
    next_v = torch.randn(1, 2, 1, 16)
    out_k, out_v = cache.update(next_k, next_v, layer_idx=0)

    assert out_k.shape == (1, 2, 5, 16)
    assert out_v.shape == (1, 2, 5, 16)
    assert cache.get_seq_length() == 5
    assert len(cache.key_cache[0]) == 2
    assert torch.isfinite(out_k).all()
    assert torch.isfinite(out_v).all()


def test_turboquant_dynamic_cache_fractional_bits_use_outlier_segments():
    torch.manual_seed(4)
    cache = TurboQuantDynamicCache(make_kv_config_from_effective_bits(2.5, codebook_grid_size=10_001))
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    out_k, out_v = cache.update(key, value, layer_idx=0)
    key_segment = cache.key_cache[0][0]
    value_segment = cache.value_cache[0][0]

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    assert isinstance(key_segment, OutlierMSESegment)
    assert isinstance(value_segment, OutlierMSESegment)
    assert key_segment.regular_bits == 2
    assert key_segment.outlier_bits == 3
    assert key_segment.outlier_indices.numel() == 8
    assert key_segment.regular_indices.numel() == 8
    assert key_segment.effective_index_bits == 2.5
    assert cache.storage_nbytes() < cache.materialized_nbytes()
    summary = cache.compression_summary()
    assert summary["segment_types"] == {"OutlierMSESegment": 2}
    assert summary["avg_effective_index_bits"] == 2.5
    assert summary["outlier_counts"] == [8]
    assert summary["regular_bits"] == [2]
    assert summary["outlier_bits"] == [3]


def test_fractional_bits_can_use_quarter_high2_allocation():
    cfg = make_kv_config_from_effective_bits(
        2.5,
        codebook_grid_size=10_001,
        effective_bit_allocation="quarter_high2",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    cache.update(key, value, layer_idx=0)
    key_segment = cache.key_cache[0][0]

    assert isinstance(key_segment, OutlierMSESegment)
    assert key_segment.regular_bits == 2
    assert key_segment.outlier_bits == 4
    assert key_segment.outlier_indices.numel() == 4
    assert key_segment.regular_indices.numel() == 12
    assert key_segment.effective_index_bits == 2.5


def test_turboquant_fast_materialized_eval_matches_regular_materialization():
    torch.manual_seed(5)
    cfg = make_kv_config_from_effective_bits(2.5, codebook_grid_size=10_001)
    fast_cfg = make_kv_config_from_effective_bits(2.5, codebook_grid_size=10_001, fast_materialized_eval=True)
    cache = TurboQuantDynamicCache(cfg)
    fast_cache = TurboQuantDynamicCache(fast_cfg)

    for _ in range(3):
        key = torch.randn(1, 2, 2, 16)
        value = torch.randn(1, 2, 2, 16)
        out_k, out_v = cache.update(key, value, layer_idx=0)
        fast_k, fast_v = fast_cache.update(key, value, layer_idx=0)
        assert torch.equal(out_k, fast_k)
        assert torch.equal(out_v, fast_v)

    assert fast_cache.fast_materialized_nbytes() == fast_cache.materialized_nbytes()
    assert fast_cache.storage_nbytes() < fast_cache.materialized_nbytes()
    assert fast_cache.compression_summary()["fast_materialized_eval"] is True


def test_turboquant_dynamic_cache_supports_prod_key_segments():
    torch.manual_seed(6)
    cfg = make_kv_config_from_effective_bits(
        3,
        codebook_grid_size=10_001,
        key_quantizer="prod",
        value_quantizer="mse",
    )
    cache = TurboQuantDynamicCache(cfg)
    key = torch.randn(1, 2, 4, 16)
    value = torch.randn(1, 2, 4, 16)

    out_k, out_v = cache.update(key, value, layer_idx=0)

    assert out_k.shape == key.shape
    assert out_v.shape == value.shape
    assert isinstance(cache.key_cache[0][0], PackedProdSegment)
    assert isinstance(cache.value_cache[0][0], PackedMSESegment)
    assert cache.storage_nbytes() < cache.materialized_nbytes()
