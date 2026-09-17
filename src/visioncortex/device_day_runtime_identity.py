"""Exact, bounded runtime-only compatibility reviewed in FeedBatchOptimization20260911.

Real FP/TP ledger comparison covers 27,720 frames. This is not a quality
acceptance claim for all NAS media. Unknown code/config changes invalidate.
"""
import hashlib
import json
from pathlib import Path

# The exact short-source recovery revision retains every successful path;
# only a previously failed coverage check can request one native-FPS rescan.
# Completed artifacts still require their source and content hash checks.
_REVIEWED = {'detection.py': ('6693f6903e7871a3357bf6fe5a62ed03dae274f408e033e8b1d55d439d45168f', '70ade25421dd7f548450e670143eda25201e360e669998ac7f2348735129595f'), 'device_day_models.py': ('a2396878a631ca4e61d0159322bc0c25ad73cef162ab4c3e64497bcea525c943', '18676c4cf12d85406ce35f781bebad0ad990f214a8f6582589fd26785429aa62'), 'scan_scheduler.py': ('74961086aa700c28df181570012056be6e72ae2fcbca018e098a699dd790a209', None), 'shared_inference.py': ('358f149b01de6da7b5445cfb2c18ae357250079be9bfabd6a53a42b4d87e28eb', None)}
_PERFORMANCE = ['fc756fe919bdaf341fffb32a771e485d07650d403f37ccfac6ca7a052845f3de', 'ddf83ed275e789d12f5adbde35d04756d3e711f181b60e718eace58dc0c798d0', '3286e2f84266631cfb8abe02f26aceae0a8a4db3d2c04a1574c6870a94b1099d', 'b7011226efcb2d0c3f5f86b2d25ec8c87bdb85b1968117475b3627774a8f7415']


# Sparse RGB fallback changes only a bounded clock read that previously raised.
# All successful reads keep their exact points; 23 alignment tests plus the
# failing Aug 27 CSV verify recovery. Preserve completed scan identities only
# for this exact source revision, never for unknown alignment edits.
_REVIEWED["alignment.py"] = (
    "bc949dc1b84eaddce327593e9009fa3020388f4d1e14b322391ba891421de282",
    "678e32f73df6bb7f2d18421359f43dc0250b6b1c9f9072c3e83c2bccc884d2d3",
)


def compatible_runtime_hash(path, checksum):
    path = Path(path)
    pair = _REVIEWED.get(path.name)
    if path.parent.resolve() == Path(__file__).parent.resolve() and pair and checksum == pair[0]:
        return pair[1]
    return checksum


def compatible_performance(performance):
    checksum = hashlib.sha256(json.dumps(performance, sort_keys=True).encode()).hexdigest()
    if checksum in _PERFORMANCE:
        legacy = {k:v for k,v in performance.items() if k != "shared_inference_enabled"}
        return dict(legacy, coarse_decode_lanes=["cuda"], cpu_decode_threads=12)
    return performance


def independent_vision_backend_hash(path, checksum, *, legacy_understanding=False):
    """An understanding-only edit cannot invalidate identical CV code.

    Compare the complete AST except the understand method against the reviewed
    baseline; any other edit fails closed. Only callers whose stage uses the
    unchanged CV/legacy path may request this identity.
    """
    import ast
    from .device_day_cache_identity import _normalized
    if legacy_understanding and checksum != 'fd755dac9528bbc51d868de8ce8b08ed29cf43da4f1724c411af9927d8d2c4e4':
        return checksum
    tree = ast.parse(Path(path).read_text())
    backend = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DeviceDayModels')
    backend.body = [n for n in backend.body if not isinstance(n, ast.FunctionDef) or n.name != 'understand']
    fingerprint = hashlib.sha256(json.dumps(_normalized(tree), sort_keys=True).encode()).hexdigest()
    if fingerprint == '0d5a14817187f10a3c9b61e218a841cfbdb4544a627389a4db0d093d43c7bee6':
        return '3dda21db1a138404601bba1594ffb23f10eb97b933f15e78a46762d1c13a5e75'
    return checksum
