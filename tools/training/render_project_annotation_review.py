"""Render a local Chinese review page; this does not certify or export training truth."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def render_review(inventory: Path, proposals: Path, decisions: Path, ontology: Path, output: Path) -> dict:
    source = json.loads(inventory.read_text(encoding="utf-8"))
    review = json.loads(decisions.read_text(encoding="utf-8"))
    classes = json.loads(ontology.read_text(encoding="utf-8"))["classes"]
    if review.get("truth_status") != "project_annotations" or review.get("independent_ground_truth") is not False:
        raise ValueError("This renderer requires explicitly non-independent project annotations")
    proposed = {row["image_id"]: row for row in map(json.loads, proposals.read_text().splitlines())}
    items = []
    for item in source["images"]:
        key = item["image_id"]
        record = review["images"].get(key, {})
        if not record:
            continue
        path = Path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Source hash changed: {key}")
        draft = {int(b["proposal_id"][1:]): b for b in proposed[key]["proposals"]}
        boxes = []
        for number in record.get("accepted_proposals", []):
            b = draft[number]
            boxes.append({"class_id": record.get("reclassified_proposals", {}).get(str(number), b["class_id"]),
                          "xyxy_px": record.get("revised_boxes", {}).get(str(number), b["xyxy_px"]),
                          "source_proposal": b["proposal_id"]})
        boxes.extend(record.get("added_boxes", []))
        counts = Counter()
        for b in boxes:
            cid = b["class_id"]
            x1, y1, x2, y2 = b["xyxy_px"]
            if not (0 <= cid < len(classes) and 0 <= x1 < x2 <= item["width"] and 0 <= y1 < y2 <= item["height"]):
                raise ValueError(f"Invalid reviewed box in {key}: {b}")
            counts[cid] += 1
            b["display_name"] = classes[cid]["display_name"]
            b["instance_name"] = f'{b["display_name"]} {counts[cid]:02d}'
        im = Image.open(path).convert("RGB")
        im.thumbnail((1280, 960))
        buffer = io.BytesIO()
        im.save(buffer, format="JPEG", quality=90)
        items.append({"id": key, "width": item["width"], "height": item["height"],
                      "status": record.get("status", "pending"), "group": item["source_group"],
                      "boxes": boxes, "ignore": record.get("ignore_regions", []),
                      "image": "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")})
    # Prioritize the user's dense-instance examples without changing any source/split.
    order = {key: i for i, key in enumerate(["F103", "F104", "F109", "F107", "F111", "F098", "F015"])}
    items.sort(key=lambda x: (order.get(x["id"], 1000), x["id"]))
    summary = {"source_images": len(source["images"]), "review_records": len(items),
               "status_counts": dict(Counter(x["status"] for x in items)),
               "instance_count": sum(len(x["boxes"]) for x in items),
               "truth_status": "project_annotations", "independent_ground_truth": False,
               "display_language": "zh-CN", "training_exported": False}
    data = json.dumps({"items": items, "classes": classes, "summary": summary}, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    receipt = {**summary, "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                          for p in [inventory, proposals, decisions, ontology]},
               "html_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return receipt


TEMPLATE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>项目标注 · 实物逐件核对</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f7f7;color:#183e46;font:16px/1.55 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1600px;margin:auto;padding:28px}h1{margin:0;font-size:30px}p{margin:8px 0}header,.controls,.viewer,aside{background:white;border:1px solid #d8e3e4;border-radius:12px;padding:20px;margin-bottom:16px}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}select,button{font:inherit;border:1px solid #b8cbce;background:white;color:#183e46;border-radius:6px;padding:7px 12px}button{cursor:pointer}button.active{background:#195661;color:white}.layout{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:16px}.viewer{padding:10px;align-self:start}svg{display:block;width:100%;background:#edf2f2}.instance{cursor:pointer;fill:transparent;stroke-width:2;vector-effect:non-scaling-stroke}.instance:hover,.instance.selected{stroke-width:5;fill:#fff2}.marker{pointer-events:none;font-weight:bold;paint-order:stroke;stroke:#111;stroke-width:3px;fill:white}#list{max-height:70vh;overflow:auto}.item{display:block;width:100%;text-align:left;margin:5px 0}.muted{color:#58747b;font-size:14px}.legend{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}.legend button{font-size:14px}.ignored{fill:#ffc10722;stroke:#c48500;stroke-dasharray:8 6;stroke-width:2;vector-effect:non-scaling-stroke}@media(max-width:900px){.layout{grid-template-columns:1fr}main{padding:12px}}
</style><main><header><h1>项目标注 · 实物逐件核对</h1><p>每支移液枪、每根试管分别标注；架子、盒子和可辨认的盖子另列。</p><p class="muted" id="summary"></p><p class="muted">当前为项目标注首轮检查，尚未完成全部复核或模型训练。黄色虚线区域需要进一步核对，不能当作背景负样本。</p></header>
<div class="controls"><button id="prev">上一张</button><label>查看图片 <select id="images"></select></label><button id="next">下一张</button><button id="boxes" class="active">显示标注框</button><span id="status"></span></div>
<div class="legend" id="legend"></div><div class="layout"><section class="viewer"><svg id="canvas" xmlns="http://www.w3.org/2000/svg"></svg></section><aside><strong>图中实物</strong><p class="muted">点击名称或框查看对应实物。框边数字对应列表序号。</p><div id="list"></div><p id="detail" class="muted"></p></aside></div>
<p class="muted">类别名称均以中文展示；内部训练编号不作为查看标签。人称来源另按拍摄组核对，训练与验收分别进行。</p></main><script id="review-data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('review-data').textContent);let current=0,filter=null,visible=true;const $=id=>document.getElementById(id);const ns='http://www.w3.org/2000/svg';const colors=['#00efff','#ff7e3d','#86ff3d','#ffdc33','#ed64f5','#68a7ff'];function el(name,attrs,text){const node=document.createElementNS(ns,name);Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,v));if(text)node.textContent=text;return node}D.items.forEach((x,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`图片 ${x.id.slice(1)} · ${x.boxes.length} 件已标实物`;$('images').append(o)});$('summary').textContent=`原始图片 ${D.summary.source_images} 张；已有处理记录 ${D.summary.review_records} 张；当前记录实例 ${D.summary.instance_count} 件。`;
function select(n){document.querySelectorAll('.selected').forEach(x=>x.classList.remove('selected'));const box=$('box-'+n);if(box)box.classList.add('selected');const b=D.items[current].boxes[n];$('detail').textContent=b.instance_name+'；框覆盖可见部分，严重遮挡处不补画。'}
function render(){const x=D.items[current];$('images').value=current;$('canvas').replaceChildren();$('canvas').setAttribute('viewBox',`0 0 ${x.width} ${x.height}`);$('canvas').append(el('image',{href:x.image,width:x.width,height:x.height}));$('list').replaceChildren();$('legend').replaceChildren();$('status').textContent=({reviewed:'已初标 · 待复核',reviewed_with_ignore_regions:'已初标 · 有待核对区域',needs_second_pass:'需要再次核对',excluded:'本轮不纳入训练'})[x.status]||'待标注';const counts=new Map; x.boxes.forEach(b=>counts.set(b.class_id,(counts.get(b.class_id)||0)+1));for(const [cid,name] of [[null,'全部实物'],...[...counts].map(([cid,n])=>[cid,D.classes[cid].display_name+' '+n+' 件'])]){let b=document.createElement('button');b.textContent=name;b.classList.toggle('active',filter===cid);b.onclick=()=>{filter=cid;render()};$('legend').append(b)}x.boxes.forEach((b,n)=>{if(filter!==null&&b.class_id!==filter)return;const [l,t,r,d]=b.xyxy_px;const color=colors[b.class_id%colors.length];if(visible){const g=el('g',{});const box=el('rect',{id:'box-'+n,x:l,y:t,width:r-l,height:d-t,stroke:color,class:'instance'});box.append(el('title',{},b.instance_name));box.onclick=()=>select(n);g.append(box,el('text',{x:l+3,y:t+Math.max(18,x.width/60),class:'marker','font-size':Math.max(18,x.width/60)},String(n+1)));$('canvas').append(g)}let row=document.createElement('button');row.className='item';row.textContent=`${n+1}. ${b.instance_name}`;row.style.borderLeft='5px solid '+color;row.onclick=()=>select(n);$('list').append(row)});if(visible)x.ignore.forEach(b=>{let [l,t,r,d]=b.xyxy_px;let q=el('rect',{x:l,y:t,width:r-l,height:d-t,class:'ignored'});q.append(el('title',{},'边界不清，暂不作为普通训练区域'));$('canvas').append(q)});$('detail').textContent='';}
$('images').onchange=e=>{current=Number(e.target.value);filter=null;render()};$('prev').onclick=()=>{current=(current+D.items.length-1)%D.items.length;filter=null;render()};$('next').onclick=()=>{current=(current+1)%D.items.length;filter=null;render()};$('boxes').onclick=()=>{visible=!visible;$('boxes').textContent=visible?'显示标注框':'只看原图';$('boxes').classList.toggle('active',visible);render()};if(D.items.length)render();
</script></html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["inventory", "proposals", "decisions", "ontology", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(render_review(**vars(parser.parse_args())), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
