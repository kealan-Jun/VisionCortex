from pathlib import Path

import pytest

from visioncortex.cache_paths import check_path_budget, model_cache_directory


def test_reported_windows_installation_keeps_sam2_leaf_below_legacy_limit():
    root = Path('E:/VisionCortex-RTX4050-Desktop-Offline-20260907-R6/Runtime/Cache')
    leaf = model_cache_directory(root, 's2', 'a' * 64) / 'frames/00000.jpg'
    assert len(str(leaf).encode('utf-16-le')) // 2 < 235
    assert leaf.parts[-3] == 'a' * 64
    # Even the share-specific local runtime has room, without expanding Y:.
    root = Path('E:/VisionCortex-RTX4050-Desktop-Offline-20260907-R6/Runtime/NetworkStores') / ('b' * 20) / 'Cache'
    check_path_budget(model_cache_directory(root, 's2', 'a' * 64) / 'frames/00000.jpg')
    assert model_cache_directory(Path('Y:/Cache'), 'work', 'a' * 20).parts[0] == 'Y:'


def test_path_guard_counts_windows_utf16_units_and_rejects_before_io():
    with pytest.raises(ValueError, match='缓存路径过长'):
        check_path_budget(Path('E:/' + '🧪' * 117))
    with pytest.raises(ValueError, match='Invalid'):
        model_cache_directory(Path('E:/Cache'), 's2', '../escape')
    assert model_cache_directory(Path('E:/Cache'), 's2', 'a' * 64) != model_cache_directory(Path('E:/Cache'), 's2', 'a' * 63 + 'b')
