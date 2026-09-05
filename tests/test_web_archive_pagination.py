import shutil
import subprocess
from pathlib import Path

import pytest


def test_sectioned_archive_loaders_preserve_pagination_and_local_library_state():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for frontend loader characterization")
    source = (Path(__file__).parents[1] / "src/visioncortex/web/app.js").read_text()
    names = (
        "loadArchive", "loadArchiveSection", "loadArchiveMaterials",
        "loadArchiveExperiments", "loadArchiveView", "materialQueryParameters",
        "ensureMaterialFilters", "archiveProcessStopped", "loadLibraryDetail",
        "libraryQueryKey", "cachedArchiveDetail",
    )
    functions = []
    for name in names:
        prefix = f"async function {name}(" if f"async function {name}(" in source else f"function {name}("
        body = source.split(prefix, 1)[1].split("\n}\n", 1)[0]
        functions.append(prefix + body + "\n}\n")
    script = r'''
const assert = require("node:assert/strict");
const state = {
  archiveCache: new Map(), materialCache: new Map(),
  materialFilters: {}, globalMaterialFilters: {action:"all",object:"all",support:"all"},
  search: "", selectedMaterials: new Set(),
};
let release = "v1";
const requests = [];
async function api(url) {
  requests.push(url);
  const query = new URL(url,"http://test").searchParams;
  if (url.startsWith("/api/key-events?")) return {
    items: [{event_id:query.has("cursor")?"E2":"E1",release_id:release}],
    next_cursor:query.has("cursor")?null:"next",total_count:2,
  };
  assert.ok(query.has("section"), "complete archives must use bounded sections");
  const summary = {name:"Archive",release_id:release,counts:{experiments:2,key_events:2},observability:{status:{stage:"completed"}}};
  if (query.get("section")==="summary") return summary;
  if (query.get("section")==="reports") return {...summary,daily_report:{report_id:"R1"}};
  return {...summary,experiment_groups:[],experiments:[{name:query.has("cursor")?"G2":"G1"}],next_cursor:query.has("cursor")?null:"next"};
}
'''
    script += "\n".join(functions)
    script += r'''
(async()=>{
  const first = await loadArchiveView("Archive","experiments");
  assert.equal(first.key_events.length,0);
  assert.equal(first.counts.key_events,2);
  await loadArchiveExperiments("Archive",first.next_cursor);
  assert.deepEqual((await loadArchiveView("Archive","experiments")).experiments.map(x=>x.name),["G1","G2"]);
  await loadArchiveMaterials("Archive");
  await loadArchiveMaterials("Archive","next");
  assert.equal((await loadArchiveView("Archive","materials")).key_events.length,2);
  await loadLibraryDetail("Archive","reports");
  await loadLibraryDetail("Archive","materials");
  await loadLibraryDetail("Archive","materials","next");
  assert.equal(cachedArchiveDetail("Archive").daily_report.report_id,"R1");
  assert.equal(cachedArchiveDetail("Archive").key_events.length,2);
  assert.equal(cachedArchiveDetail("Archive").libraryKeys.reports,"reports");
  state.materialFilters.query="开盖";
  state.materialFilters.object="tube";
  assert.equal(materialQueryParameters("Archive").get("q"),"开盖 tube");
  release="v2";
  state.archiveCache.get("Archive:summary").loadedAt=0;
  assert.equal((await loadArchive("Archive")).release_id,"v2");
  assert.equal(cachedArchiveDetail("Archive"),null);
  assert.equal(state.materialCache.size,0);
  assert.ok(![...state.archiveCache.keys()].some(k=>k.includes(":v1:")));
})().catch(error=>{console.error(error);process.exitCode=1;});
'''
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
