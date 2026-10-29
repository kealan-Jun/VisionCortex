"""Neutral synthetic old metadata aliases; no customer/NAS reads or writes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

import visioncortex.lab.archive_layout as layout
from visioncortex.lab.domain import Conflict


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def compatible(tmp_path,monkeypatch):
    root=tmp_path/'archive'
    (root/'.System').mkdir(parents=True)
    current=layout.contract_bytes()
    old=json.loads(current)
    aliases=dict(project='FixtureLegacy',schema_version='fixture-archive/2',supersedes='fixture-archive/1')
    old.update(aliases)
    raw=(json.dumps(old,ensure_ascii=False,indent=1)+'\n').encode()
    path=root/layout.CONTRACT_PATH;path.write_bytes(raw)
    manifest=tmp_path/'private-compatibility.json'
    binding=dict(archive_root=str(root),source_sha256=sha(raw),installed_contract_sha256=sha(current),metadata_aliases=aliases)
    document=dict(schema_version='vc-lab-archive-compatibility/1',bindings=[binding])
    manifest.write_text(json.dumps(document))
    monkeypatch.setenv('VC_LAB_ARCHIVE_COMPATIBILITY_MANIFEST',str(manifest))
    return dict(root=root,path=path,raw=raw,manifest=manifest,document=document,current=current)


def save(fixture,document):
    fixture['manifest'].write_text(json.dumps(document))


def test_explicit_three_metadata_aliases_accept_without_writing_archive_or_current_contract(compatible):
    f=compatible
    (f['root']/'existing-data').mkdir();(f['root']/'existing-data'/'receipt.json').write_text('retained original')
    before={p.relative_to(f['root']).as_posix():p.read_bytes() for p in f['root'].rglob('*') if p.is_file()}
    assert layout.verify_archive_contract(f['root'])==f['path']
    assert before=={p.relative_to(f['root']).as_posix():p.read_bytes() for p in f['root'].rglob('*') if p.is_file()}
    assert layout.contract_bytes()==f['current']


def test_missing_manifest_retains_strict_original_contract_gate(compatible,monkeypatch):
    monkeypatch.delenv('VC_LAB_ARCHIVE_COMPATIBILITY_MANIFEST')
    with pytest.raises(Conflict):layout.verify_archive_contract(compatible['root'])
    compatible['path'].write_bytes(compatible['current'])
    assert layout.verify_archive_contract(compatible['root'])==compatible['path']


@pytest.mark.parametrize('field,value',[('archive_root','/fixture/wrong-root'),('source_sha256','0'*64),('installed_contract_sha256','1'*64)])
def test_root_source_and_installed_contract_must_all_match_exact_pins(compatible,field,value):
    d=deepcopy(compatible['document']);d['bindings'][0][field]=value;save(compatible,d)
    with pytest.raises(Conflict):layout.verify_archive_contract(compatible['root'])


@pytest.mark.parametrize('change',['directory','extra_field','typed_value'])
def test_aliases_never_allow_directory_or_other_field_semantic_changes(compatible,change):
    f=compatible;source=json.loads(f['raw'])
    if change=='directory':source['directories']['Comment']['children']['Audio']['content']='changed layout semantics'
    elif change=='extra_field':source['extra_ignored_field']='must fail'
    else:source['partition_by_processing_project']=0
    raw=json.dumps(source).encode();f['path'].write_bytes(raw)
    d=deepcopy(f['document']);d['bindings'][0]['source_sha256']=sha(raw);save(f,d)
    with pytest.raises(Conflict):layout.verify_archive_contract(f['root'])


@pytest.mark.parametrize('alias,value',[('directories','forbidden'),('project',True),('schema_version','wrong-old-value')])
def test_only_three_exact_old_string_aliases_are_allowed(compatible,alias,value):
    d=deepcopy(compatible['document']);d['bindings'][0]['metadata_aliases'][alias]=value;save(compatible,d)
    with pytest.raises(Conflict):layout.verify_archive_contract(compatible['root'])


@pytest.mark.parametrize('kind',['symlink','relative','nas','oversized'])
def test_private_manifest_must_be_bounded_canonical_local_file(compatible,monkeypatch,kind):
    f=compatible;p=f['manifest']
    if kind=='symlink':
        link=p.with_name('alias.json');link.symlink_to(p);p=link
    elif kind=='relative':p=Path('private-compatibility.json')
    elif kind=='nas':p=Path('/mnt/fixture-unmounted/private.json')
    else:p.write_text(' '*65537)
    monkeypatch.setenv('VC_LAB_ARCHIVE_COMPATIBILITY_MANIFEST',str(p))
    with pytest.raises(Conflict):layout.verify_archive_contract(f['root'])


def test_duplicate_binding_or_json_keys_cannot_select_an_ambiguous_alias(compatible):
    d=deepcopy(compatible['document']);d['bindings'].append(deepcopy(d['bindings'][0]));save(compatible,d)
    with pytest.raises(Conflict):layout.verify_archive_contract(compatible['root'])
    compatible['manifest'].write_text('{"schema_version":"ignored","schema_version":"vc-lab-archive-compatibility/1","bindings":[]}')
    with pytest.raises(Conflict):layout.verify_archive_contract(compatible['root'])
