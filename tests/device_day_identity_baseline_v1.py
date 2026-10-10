"""Frozen device-day identity recipe v1, independent of Git history.

Only the historical descriptor calculation and pure alias decisions are kept.
Source/AST fingerprints are immutable witnesses, not operational source copies.
Extraction and deterministic vector equivalence are verified against a private
full baseline. No GPU, NAS or worker implementation is executed here.

The deterministic harness supplies today's contract, coverage, request-policy,
STT, binding and local marker ports, as in the original compatibility tests.
The frozen version covers descriptor and alias decisions, not old worker code.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import Any

from visioncortex.device_day_contract import VERSION, digest

RECIPE_VERSION = "visioncortex-historical-device-identity/1"

SOURCE_SHA256_V1 = {'action_semantics.py': 'c15630337a443d787eae98ebcc519378f8d4e5f9aeb1c3dd5317ede1586b174e',
 'action_state_machine.py': '1d93ac0bec0c36d03ac45ced1734ff0640daa9a977b809e2b8ece43fe223fd5a',
 'actions.py': '487f46fa65913cb791c07c321cb7774fbe5c52718b014fa77d17c5cf09dbd18e',
 'alignment.py': '3a5b5ad4ff85ccd8649f19d902c89f0c1efb91e1174159266efd13c4903262e8',
 'archive.py': '7a7fb4c76bc274c384b4c6a7faa878e35522de7f93d31c8eeb0fbce5ad3e3709',
 'candidate_index.py': '36899b383f759d9c40b5709265de207bd88cb536c79507a83209a0fb4b009d2a',
 'coarse_recall.py': '37007ba008926d52d9ab153aeb603482b31a7813c626fab77a1cdb677882a4c7',
 'cuda_decode_admission.py': 'c83b5f907ccc9870858c0ba48851e1bec95a7fc80544913c4acabba60b256c33',
 'daily_reports.py': '2281125ea9eb72210bae97f2045169b9bdf5f12289bd204d2acc866ec63a689b',
 'detection.py': '00c44f8fd0fe1de6a1b44a5befa19b62081e61fa052aabeb16b209616fdabc93',
 'detection_inference.py': 'c3d8dfe81421e95647b59278ff78c2e5181437b61916104c02b29f83ecd7e103',
 'device_day.py': 'a9379afbb28a23785dc5cabd8d5d6f2cf2795feb93a29f27619626139e045993',
 'device_day_audio.py': 'e1b8764c779f2ab681a94659b4a6ca533ae499468a8b85e8232acc577905b55e',
 'device_day_audit.py': '9250a97ddbce642dcfc50a78b41ba19a2b62136058a318239eae7f029eeb38ef',
 'device_day_cache_identity.py': 'e00d1f37392149be04238287fa9d141e53f98915ff296a57fa76f3f0a98f7bf3',
 'device_day_contract.py': '49a595c73bf960a3985e5d2bd877272a9e67be8da290a15196374f430f1fa9a8',
 'device_day_inplace.py': '47253fe0ea0a056686b8e9dcef34c68a6dd046d009c8530521afffd165a121e2',
 'device_day_inputs.py': '3713f16c0b05c3c5ed850bc71d3a8f3d2c25f085cd655b9f9ca24b910871a61b',
 'device_day_models.py': '1b2f1213f58f9d7c4f99bd14b22cf6358e3ff7b73a72769c2f85d5649c0731e6',
 'device_day_reports.py': 'e4fe1dacc8503c74c7c3677b31aa1d900dcca108c10826e9d91a234b9c060564',
 'device_day_runtime_identity.py': '20c77d9dbfddf72823509ed81ca6d5db04e77d1798a0478ad33657a2c9425147',
 'device_day_semantic_cache.py': 'af90cea7dba54fc9fb655f6ed5c121e2157ae1cd9a49c70e600838739ffda15b',
 'device_day_steps.py': '98deead23f009ac4d83071d830a6449512f537a7e77c05575d5b30f3612c8fed',
 'device_day_stt.py': '08a23bc1a6b5c4ecf11544f411363bc2abbaa27ce9f2732187720d711ec4d2ac',
 'device_day_understanding.py': 'ca86f966bea04873d809262502352adc68018fb7894090de97a4d338ff0fb0bb',
 'device_day_verification.py': '9e7e062bb1d1a3e025e92b59f6d9864290bdbad81cd35656eabc15b016e6ab1b',
 'grouping.py': '8b8786e0d27e92ab1a4cf890e5081f353d7038ff5703f2cc1fd699281f2b5397',
 'media_time.py': '6419b4e822b7af532f6e7aa0d7e51960dab70b1f23d0a95c9b76b95f86ca9e97',
 'mllm.py': '4ab2f6521e43fcf87d77f4450dd8a92b3cd4e7d92216ae85ad5d25ee50c3077e',
 'mllm_provider.py': 'fdac3b8037763a6577b161015e836bf862e7d0e7bcc9f52a2e55bbd93301972d',
 'movement_verification.py': 'ff6bba3b448b7e476b036e269d4bd6a7ad0e6fef65fff2a73742d6b8f31f4811',
 'nas_recordings.py': '082eb5a45d277e4582b2e7b654fc60769a1e341b1c61dc68e92b713a09774b09',
 'open_vocabulary_runtime.py': '62bb0708b8a9155fa8166ac7b6780b0dad4582668f5f45d9b46962d5a4f1a463',
 'report_presentations.py': '42a15f17f3bf49bca1a1619411f6d5978baa488b6f136b26dc1dcbcbb344d82f',
 'scan_scheduler.py': 'b86a71f3cdb6b58f5378ce0c889a93b0a0f12a9ddef339a4886f359f4b6e3e5f',
 'scene_requests.py': '1792885ab4d7a2dde077cd807b9c7c8af146f23f2f2afe713899383bb0b73f3a',
 'scene_transport.py': 'd9092d64080d98528263dd25264fe4f596bfbdab70af5d177cac02c5d95e0264',
 'shared_inference.py': 'f56476d52b0aedaba492614c3f81998f4695c6081d4eef2ad2bc0c258c2dbfab',
 'source_frames.py': 'f27275927934c866a3e48dddd0af23ae9a0246940ade92517e82c6d995a52b9a',
 'speech.py': '1656e1478964380e01ddcb96ce4cdc7859dc8fd6136104291e5e49937f442a6e',
 'speech_qwen.py': '7f80e9a0420904ee7c044a52ce98792fb19453ca7995cd18c03ccfd41353ed6f',
 'speech_worker.py': '7b52906ffc73e2e1628a30ab5bafd5a5674325e95ff26b2ac99ca73a62c477ea',
 'video_io.py': '633f56bce97a3ed2098958b4e08a633c0de4d7865ab1d662963d9cdf4592ee92'}

DEVICE_SOURCES_V1 = {'report': ['device_day_reports', 'daily_reports', 'report_presentations'],
 'retention': ['nas_recordings', 'device_day_audio'],
 'stt': ['device_day_stt', 'speech', 'speech_worker', 'speech_qwen'],
 'understanding': ['device_day_models',
                   'device_day_audit',
                   'device_day_semantic_cache',
                   'scene_requests',
                   'scene_transport',
                   'mllm',
                   'mllm_provider',
                   'video_io',
                   'media_time'],
 'vision': ['shared_inference',
            'scan_scheduler',
            'device_day_models',
            'device_day_audit',
            'detection',
            'detection_inference',
            'actions',
            'candidate_index',
            'video_io',
            'cuda_decode_admission',
            'alignment',
            'movement_verification',
            'coarse_recall',
            'grouping',
            'action_state_machine',
            'action_semantics',
            'archive',
            'open_vocabulary_runtime',
            'source_frames',
            'media_time']}

EXECUTION_IDENTITY_V1 = 'ac03f3ba8e0281bc7db26b2efb98b525ea0bdcb1dbd902807437c6e92aac9e33'

INPLACE_EXECUTION_IDENTITY_V1 = '3bb8ceaea6c91571b0f6304c697ab870319ab6805125988ae8bde79d3e819c3a'

VISION_AST_SHA256_V1 = '0acf44ba502757b923d657785ae2742a1fc412ab44fb136dd63ce317c91ed1a4'

AUDIT_BACKEND_SHA256_V1 = '1b2f1213f58f9d7c4f99bd14b22cf6358e3ff7b73a72769c2f85d5649c0731e6'

AUDIT_HELPER_SHA256_V1 = '9250a97ddbce642dcfc50a78b41ba19a2b62136058a318239eae7f029eeb38ef'

PUBLICATION_AST_SHA256_V1 = '0672c774de57cc42959b468e95c5bea7e20ed8b03cba1bf1ef84bff1d7d782d7'

PUBLICATION_IDENTITY_V1 = '3bb8ceaea6c91571b0f6304c697ab870319ab6805125988ae8bde79d3e819c3a'

_REVIEWED_V1 = {'alignment.py': ('bc949dc1b84eaddce327593e9009fa3020388f4d1e14b322391ba891421de282',
                  '678e32f73df6bb7f2d18421359f43dc0250b6b1c9f9072c3e83c2bccc884d2d3'),
 'detection.py': ('6693f6903e7871a3357bf6fe5a62ed03dae274f408e033e8b1d55d439d45168f',
                  '70ade25421dd7f548450e670143eda25201e360e669998ac7f2348735129595f'),
 'device_day_contract.py': ('1df5edbf53357f9d67134b386480e92056c71f492ab909cd473544c9f246be0b',
                            '91addcf12669693241471bd8ebb8da3b4c6ad952ed1aef5699e2dc71d553f25a'),
 'device_day_inputs.py': ('3713f16c0b05c3c5ed850bc71d3a8f3d2c25f085cd655b9f9ca24b910871a61b',
                          '6a92d972cc2721f852d5774be6a2ea771eaf1a594f389778ebab0d691e6def20'),
 'device_day_models.py': ('47b7460f97d5c671894b49dfdf5a629813799fa7b9dde73b6ab5a02ad7303fd6',
                          '7c04a7a02843f482f30e5524a1e84cbbaea21278b03fbe8d494789fd24e1b3cd'),
 'device_day_stt.py': ('08a23bc1a6b5c4ecf11544f411363bc2abbaa27ce9f2732187720d711ec4d2ac',
                       '4809d85774f2e6e743bcbd37c19cdca73a23bcd09cc3922c8ae776ee2bdfee56'),
 'device_day_understanding.py': ('58c33e3f9daf018267846222a8986b1c345945cbe9cf2c2174eef14fdd113846',
                                 'c3b82cff2d2cf4b29c16fc1ed6a05569c489372c0cf2037d08075fa39e98641d'),
 'scan_scheduler.py': ('74961086aa700c28df181570012056be6e72ae2fcbca018e098a699dd790a209', None),
 'shared_inference.py': ('358f149b01de6da7b5445cfb2c18ae357250079be9bfabd6a53a42b4d87e28eb', None),
 'source_frames.py': ('f27275927934c866a3e48dddd0af23ae9a0246940ade92517e82c6d995a52b9a',
                      '3bf96e4268b9b164d75463b1b340037fc256315be94f6a1a2711da57712cb2fd'),
 'speech_qwen.py': ('7f80e9a0420904ee7c044a52ce98792fb19453ca7995cd18c03ccfd41353ed6f',
                    '888cc607b9243b31571c51d8cd8989e342f544263a8e133b5afd50426dad54d7'),
 'speech_worker.py': ('7b52906ffc73e2e1628a30ab5bafd5a5674325e95ff26b2ac99ca73a62c477ea',
                      'a5894d18a71afd228332dbc24b770c924509f60e9fd9372a902e55b1f3980b63')}

_PERFORMANCE = ['fc756fe919bdaf341fffb32a771e485d07650d403f37ccfac6ca7a052845f3de',
 'ddf83ed275e789d12f5adbde35d04756d3e711f181b60e718eace58dc0c798d0',
 '3286e2f84266631cfb8abe02f26aceae0a8a4db3d2c04a1574c6870a94b1099d',
 'b7011226efcb2d0c3f5f86b2d25ec8c87bdb85b1968117475b3627774a8f7415']

_AUDIT_BACKEND = ('de18f0bfdf127e0c3f1b5b732d150e53cca4c01a95b1146540d4b23ea0adaba5',
 '7c04a7a02843f482f30e5524a1e84cbbaea21278b03fbe8d494789fd24e1b3cd')

_AUDIT_HELPER = '9250a97ddbce642dcfc50a78b41ba19a2b62136058a318239eae7f029eeb38ef'

_FRAME_CACHE_BACKEND = '1b2f1213f58f9d7c4f99bd14b22cf6358e3ff7b73a72769c2f85d5649c0731e6'

_JSON_ENCODING = ('49a595c73bf960a3985e5d2bd877272a9e67be8da290a15196374f430f1fa9a8',
 '619f42061083ce5fe3c1d5b0d6637e1b98453f781c161439b1198de73a68f7e8')


def runtime_hash_v1(name, checksum, *, at_source_root, helper_checksum):
    """Historical alias decision with explicit, immutable source witnesses."""
    if at_source_root:
        if name == "device_day_contract.py" and checksum == _JSON_ENCODING[0]:
            return _JSON_ENCODING[1]
        if name == "device_day_audit.py" and checksum == _AUDIT_HELPER:
            return None
        if name == "device_day_models.py" and checksum in (_AUDIT_BACKEND[0], _FRAME_CACHE_BACKEND):
            if helper_checksum == _AUDIT_HELPER:
                return _AUDIT_BACKEND[1]
    pair = _REVIEWED_V1.get(name)
    return pair[1] if at_source_root and pair and checksum == pair[0] else checksum


def vision_backend_hash_v1(checksum, ast_fingerprint, *, legacy_understanding=False):
    """Historical CV alias decision; the witness comes from the fixed baseline."""
    if legacy_understanding and checksum != "fd755dac9528bbc51d868de8ce8b08ed29cf43da4f1724c411af9927d8d2c4e4":
        return checksum
    if ast_fingerprint == "0d5a14817187f10a3c9b61e218a841cfbdb4544a627389a4db0d093d43c7bee6":
        return "3dda21db1a138404601bba1594ffb23f10eb97b933f15e78a46762d1c13a5e75"
    return checksum


def runtime_recipes_v1(root):
    """Bind descriptor ports to this test's source witness directory only."""
    root = Path(root).resolve()

    def compatible_runtime_hash(path, checksum):
        path = Path(path)
        return runtime_hash_v1(path.name, checksum,
                               at_source_root=path.parent.resolve() == root,
                               helper_checksum=SOURCE_SHA256_V1["device_day_audit.py"])

    def independent_vision_backend_hash(path, checksum, *, legacy_understanding=False):
        path = Path(path)
        assert path.parent.resolve() == root and path.name == "device_day_models.py"
        return vision_backend_hash_v1(checksum, VISION_AST_SHA256_V1,
                                      legacy_understanding=legacy_understanding)

    return SimpleNamespace(compatible_runtime_hash=compatible_runtime_hash,
                           compatible_performance=performance_v1,
                           independent_vision_backend_hash=independent_vision_backend_hash)


def bind_stage_key_v1(root):
    """Supply an owned virtual source path without changing module globals."""
    namespace = dict(stage_key_v1.__globals__, __file__=str(Path(root) / "device_day.py"),
                     __package__="visioncortex", __name__="visioncortex._historical_identity_v1",
                     __spec__=None)
    function = FunctionType(stage_key_v1.__code__, namespace, stage_key_v1.__name__,
                            stage_key_v1.__defaults__)
    function.__kwdefaults__ = stage_key_v1.__kwdefaults__
    return function


def performance_v1(performance):
    checksum = hashlib.sha256(json.dumps(performance, sort_keys=True).encode()).hexdigest()
    if checksum in _PERFORMANCE:
        legacy = {k: v for k, v in performance.items() if k != 'shared_inference_enabled'}
        return dict(legacy, coarse_decode_lanes=['cuda'], cpu_decode_threads=12)
    return performance


def stage_key_v1(self, stage: str, recording: dict, inputs: Any, *, _binding=True) -> str:
    binding = self._completed_execution_bindings.candidate(stage, recording) if _binding else None
    original_inputs = inputs
    config = self._completed_execution_bindings.config_for(self.config, binding) if binding else self.config
    directory = Path(__file__).parent
    sources = [directory / 'device_day_contract.py', directory / 'device_day_verification.py', Path(__file__)]
    from .stage_dependencies import DEVICE_SOURCES as dependencies
    sources.extend((directory / f'{name}.py' for name in dependencies[stage]))
    from .device_day_understanding import enabled as full_coverage_enabled
    full_coverage = full_coverage_enabled(self.settings, recording)
    from .device_day_steps import enabled as step_structure_enabled
    structured_steps = step_structure_enabled(self.settings, recording)
    if stage == 'understanding' and full_coverage:
        sources.append(directory / 'device_day_understanding.py')
        if structured_steps:
            sources.append(directory / 'device_day_steps.py')
    keys = ('performance', 'segmentation', 'models', 'alignment', 'continuity') if stage == 'vision' else ()
    from .device_day_runtime_identity import compatible_performance
    settings = {key: compatible_performance(config.get(key)) if key == 'performance' else config.get(key) for key in keys}
    if stage == 'stt':
        settings['speech_recognition'] = deepcopy(config.get('speech_recognition'))
        if settings['speech_recognition'] and settings['speech_recognition'].get('adapter_sha256'):
            from .device_day_runtime_identity import compatible_runtime_hash
            settings['speech_recognition']['adapter_sha256'] = compatible_runtime_hash(directory / 'speech_qwen.py', settings['speech_recognition']['adapter_sha256'])
        registry = Path(config.get('speech_recognition', {}).get('model_registry') or 'configs/models/speech-recognition.json')
        if registry.is_file():
            settings['model_registry_hash'] = self._hash(registry)
    if stage == 'vision':
        settings['model_files'] = {k: {'path': v, 'sha256': self._hash(v)} for k, v in (config.get('models') or {}).items() if isinstance(v, str) and Path(v).is_file()}
    if stage == 'understanding':
        mllm = config.get('mllm') or {}
        settings['mllm'] = {k: mllm.get(k) for k in ('model', 'provider', 'base_url', 'enabled', 'max_images_per_group', 'temperature')}
        from .scene_requests import request_policy
        settings['request_policy'] = request_policy(config)
    stage_settings = {'schema_version': self.settings.get('schema_version', VERSION), 'timezone': self.settings.get('timezone', 'Asia/Shanghai')}
    semantic_keys = {'vision': ('chunk_seconds', 'inactive_frames'), 'understanding': ('inactive_frames', 'active_frames_per_window', 'active_window_seconds')}
    stage_settings.update({k: self.settings.get(k) for k in semantic_keys.get(stage, ())})
    if stage == 'understanding' and full_coverage:
        stage_settings.update({k: self.settings.get(k) for k in ('understanding_coverage_since_us', 'understanding_frames_per_request', 'understanding_window_seconds', 'inactive_sample_seconds')})
        if structured_steps:
            stage_settings['experiment_steps_since_us'] = self.settings['experiment_steps_since_us']
    if stage == 'retention':

        def retention_identity(value):
            if isinstance(value, list):
                return [retention_identity(item) for item in value]
            if isinstance(value, dict) and value.get('recording_id') == recording['recording_id']:
                return {k: v for k, v in value.items() if k not in {'processing_priority', 'archive_date', 'updated_at', 'source_expires_at', 'source_retention_status', 'archive_urgent', 'archive_aged', 'archive_deadline'}}
            return value
        inputs = retention_identity(inputs)
    from .device_day_inplace import marker
    from .device_day_runtime_identity import compatible_runtime_hash
    if recording.get('camera_key') and recording.get('recording_start_us') and marker(self, recording).is_file():
        stage_settings['inplace_execution_identity'] = self._inplace_execution_identity
        stage_settings['input_binding_version'] = 1
        input_contract = directory / 'device_day_inputs.py'
        stage_settings['input_contract_sha256'] = compatible_runtime_hash(input_contract, self._hash(input_contract))

    def code_hash(path):
        checksum = compatible_runtime_hash(path, self._hash(path))
        if path.name == 'device_day_models.py' and (stage == 'vision' or (stage == 'understanding' and (not full_coverage))):
            from .device_day_runtime_identity import independent_vision_backend_hash
            return independent_vision_backend_hash(path, checksum, legacy_understanding=stage == 'understanding')
        return checksum
    if stage == 'stt' and isinstance(inputs, dict) and (inputs.get('input_binding_version') == 1):
        from .device_day_inputs import canonical_stt
        inputs = canonical_stt(inputs)
    descriptor = {'stage': stage, 'source': recording['recording_id'] if stage == 'vision' else recording['source_signature'], 'inputs': inputs, 'settings': settings, 'device_day': stage_settings, 'code': [self._execution_identity if p == Path(__file__) else code_hash(p) for p in sources]}
    descriptor['code'] = [value for value in descriptor['code'] if value is not None]
    key = digest(descriptor)
    if binding and (not self._completed_execution_bindings.match_key(binding, key)):
        return self._key(stage, recording, original_inputs, _binding=False)
    from .device_day_cache_identity import BASELINE_FILE
    if stage == 'retention' and self._execution_identity == BASELINE_FILE:
        code = descriptor['code']
        legacy_code = [code[0], '36019d3ed2d5d5faec1a88604e4ec1835c915753c91bf3c47afff41c1cb91497', *code[3:]]
        self._key_aliases[key] = {digest(descriptor | {'code': legacy_code})}
    return key
