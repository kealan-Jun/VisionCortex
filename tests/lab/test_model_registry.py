"""Tiny local weight bytes only: explicit VC/FR reuse, no inference or downloads."""
from copy import deepcopy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from visioncortex.lab.processing import ModelRegistry, ModelUnavailable
import visioncortex.lab.processing as processing


@pytest.fixture
def models(tmp_path):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    return SimpleNamespace(runtime_root=runtime, model_manifest=tmp_path/'private-models.json',
                           customer_code='fixture-model-reuse', allow_external_inference=False)


def configure(settings, root, *, kind='vision', reuse_source=None):
    root.mkdir(parents=True,exist_ok=True)
    names = ['weights.bin','auxiliary.json'] if kind=='vision' else [
        d+'/'+n for d in ('det','rec') for n in ('inference.json','inference.pdiparams','inference.yml')]
    files={}
    for name in names:
        path=root/name; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(('explicit synthetic model bytes '+name).encode())
        files[name]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    model=dict(revision='synthetic-no-inference/1',python=sys.executable,files=files)
    model.update(dict(weights='weights.bin') if kind=='vision' else dict(det_directory='det',rec_directory='rec'))
    if reuse_source is not None:model.update(model_root=str(root),reuse_source=reuse_source)
    write(settings,kind,model)
    return model


def write(settings,kind,model):
    settings.model_manifest.write_text(json.dumps(dict(schema_version='vc-lab-models/1',
        customer_code=settings.customer_code,models={kind:model})))


def test_original_runtime_models_remains_compatible_without_reuse(models):
    root=models.runtime_root/'models'
    configure(models,root)
    result=ModelRegistry(models).verified('vision')
    assert result['model_root']==str(root) and 'reuse_source' not in result


@pytest.mark.parametrize('kind,origin',[('vision','visioncortex'),('ocr','field_recognition')])
def test_explicit_existing_vc_fr_roots_are_verified_without_copy_or_default_fallback(models,tmp_path,kind,origin):
    root=tmp_path/'existing-weights'
    model=configure(models,root,kind=kind,reuse_source=origin)
    before={p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
    verified=ModelRegistry(models).verified(kind)
    assert verified['model_root']==str(root) and verified['reuse_source']==origin
    assert verified['files']==model['files'] and not (models.runtime_root/'models').exists()
    assert before=={p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('changed',[
    {'model_root':None,'reuse_source':'visioncortex'},
    {'reuse_source':'field_recognition'},
    {'model_root':'relative/models','reuse_source':'visioncortex'},
    {'model_root':'/tmp/../tmp','reuse_source':'visioncortex'},
    {'model_root':'/tmp','reuse_source':'unknown'},
    {'model_root':'/tmp'},
])
def test_reuse_requires_explicit_paired_canonical_descriptor(models,changed):
    model=configure(models,models.runtime_root/'models')
    model.update(changed);write(models,'vision',model)
    with pytest.raises(ModelUnavailable):ModelRegistry(models).verified('vision')


@pytest.mark.parametrize('mode',['symlink','missing','file'])
def test_invalid_reuse_root_fails_instead_of_falling_back_to_valid_default(models,tmp_path,mode):
    model=configure(models,models.runtime_root/'models')
    root=tmp_path/'selected'
    if mode=='symlink':root.symlink_to(models.runtime_root/'models',target_is_directory=True)
    elif mode=='file':root.write_text('not a model directory')
    model.update(model_root=str(root),reuse_source='visioncortex');write(models,'vision',model)
    with pytest.raises(ModelUnavailable):ModelRegistry(models).verified('vision')


@pytest.mark.parametrize('root',['/mnt/fixture-unmounted/models','/media/fixture-unmounted/models'])
def test_nas_lexical_roots_rejected_before_hashing_or_reading_assets(models,monkeypatch,root):
    model=configure(models,models.runtime_root/'models')
    model.update(model_root=root,reuse_source='visioncortex');write(models,'vision',model)
    monkeypatch.setattr(processing,'hash_file',lambda *_:pytest.fail('NAS assets must not be read'))
    with pytest.raises(ModelUnavailable,match='local non-NAS'):ModelRegistry(models).verified('vision')


def test_remote_or_unverifiable_mount_fails_closed_for_explicit_reuse(models,tmp_path,monkeypatch):
    root=tmp_path/'weights';configure(models,root,reuse_source='visioncortex')
    monkeypatch.setattr(processing,'is_nas',lambda _:False)
    for mount in ([{'fstype':'nfs4','source':'fixture:/models'}],[],[{}]):
        monkeypatch.setattr(processing,'_mount_info',lambda _,m=mount:m)
        with pytest.raises(ModelUnavailable):ModelRegistry(models).verified('vision')


@pytest.mark.parametrize('changed',[{'sha256':'0'*64},{'bytes':1},{'bytes':True},{'sha256':None}])
def test_every_declared_file_including_auxiliary_requires_full_hash_and_strict_size(models,tmp_path,changed):
    root=tmp_path/'weights';model=configure(models,root,reuse_source='visioncortex')
    model['files']['auxiliary.json'].update(changed);write(models,'vision',model)
    with pytest.raises(ModelUnavailable):ModelRegistry(models).verified('vision')


def test_unlisted_runtime_reference_or_symlinked_asset_is_rejected(models,tmp_path):
    root=tmp_path/'weights';model=configure(models,root,reuse_source='visioncortex')
    bad=deepcopy(model);bad['weights']='unlisted.bin';write(models,'vision',bad)
    with pytest.raises(ModelUnavailable,match='runtime model reference'):ModelRegistry(models).verified('vision')
    write(models,'vision',model)
    path=root/'weights.bin';path.unlink();path.symlink_to(root/'auxiliary.json')
    with pytest.raises(ModelUnavailable):ModelRegistry(models).verified('vision')


def test_default_absent_manifest_does_not_discover_or_download_existing_weights(models,tmp_path):
    configure(models,tmp_path/'available-but-unselected',reuse_source='visioncortex')
    models.model_manifest=None
    with pytest.raises(ModelUnavailable,match='manifest is not configured'):ModelRegistry(models).verified('vision')
    assert not (models.runtime_root/'models').exists()
