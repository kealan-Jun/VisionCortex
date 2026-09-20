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

# User-authorized content relocation only. Existing bytes/receipts remain
# verified through explicit path aliases; ASR requests and model inputs are
# unchanged. Unknown source revisions still invalidate normally.
_REVIEWED['device_day_contract.py'] = (
    '1df5edbf53357f9d67134b386480e92056c71f492ab909cd473544c9f246be0b',
    '91addcf12669693241471bd8ebb8da3b4c6ad952ed1aef5699e2dc71d553f25a')
_REVIEWED['device_day_stt.py'] = (
    '2da2d36d66ba60b07755c192237da12446edc24936b336dd59e178221f3af59a',
    '4809d85774f2e6e743bcbd37c19cdca73a23bcd09cc3922c8ae776ee2bdfee56')


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

# Exact storage-only change: the same input images, prompt and response
# bytes remain accessible through verified aliases.
_REVIEWED['device_day_models.py'] = (
    'dfae6567ae073aebca46740b39da955f8710eff8aad1078922969433455de409',
    'fd755dac9528bbc51d868de8ce8b08ed29cf43da4f1724c411af9927d8d2c4e4')
_REVIEWED['device_day_understanding.py'] = (
    '58c33e3f9daf018267846222a8986b1c345945cbe9cf2c2174eef14fdd113846',
    'c3b82cff2d2cf4b29c16fc1ed6a05569c489372c0cf2037d08075fa39e98641d')

# Only the native-probe deadline changes. Successful frame/PTS ledgers and
# all source/coverage checks are identical; unknown edits still invalidate.
_REVIEWED['source_frames.py'] = (
    'f27275927934c866a3e48dddd0af23ae9a0246940ade92517e82c6d995a52b9a',
    '3bf96e4268b9b164d75463b1b340037fc256315be94f6a1a2711da57712cb2fd')
