"""竞技场对局查看器（只在本机）：http://127.0.0.1:8767/

读 arena/results/history.jsonl（每局结束时连接器给的结果：名次、积分、8 席最终阵容）、
arena/results/site_matches.json（从竞技场公开接口同步的本账号历史对局，补上本机没记下的；只读公开接口，不用密钥）和
arena/records/<对局>.jsonl（agent.py 每一步记下的可见局面、模型前 5 个选择和预计名次），显示：
  - 战绩：各赛道局数、平均名次、第一 / 前四比例、Elo 变化，按模型版本分开；名次分布（条形图）、近期表现（折线图）；
  - 单局：预计名次曲线、每回合每一步（选了什么、前 5 个候选的概率）、当时的场面 / 手牌 / 酒馆、最终 8 席阵容；
  - 正在打的对局每几秒刷新。
只用标准库；除了同步公开战绩不连网，不读密钥。用法：arena_view.bat（或 python arena/viewer.py --port 8767）
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import threading
import time
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "history.jsonl"
RECORDS = HERE / "records"
LOGS = HERE / "logs"
SITE = HERE / "results" / "site_matches.json"
SYNC = {"time": None, "n": 0, "error": None}


def _url() -> str:
    f = HERE / "url.txt"
    u = f.read_text(encoding="utf-8").strip() if f.exists() else ""
    return (u or "https://rizling.top").rstrip("/")


def _local() -> list[dict]:
    out, seen = [], set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            m = d.get("match") if isinstance(d, dict) else None
            if d.get("type") == "match_finished" and isinstance(m, dict) and m.get("id") not in seen:
                seen.add(m.get("id"))
                out.append(m)
    return out


def _history() -> list[dict]:
    """本机结果（信息全）+ 竞技场同步来的（本机没有的局）。"""
    out = _local()
    seen = {m.get("id") for m in out}
    if SITE.exists():
        try:
            for m in json.loads(SITE.read_text(encoding="utf-8")):
                if isinstance(m, dict) and m.get("id") not in seen and m.get("participants"):
                    seen.add(m.get("id"))
                    out.append(m)
        except (OSError, ValueError):
            pass
    return out


def sync(max_pages: int = 20) -> dict:
    """从竞技场公开接口 /api/v1/matches?users=<本账号> 拉本账号所有已结束的对局（不需要密钥）。"""
    me = _me(_local())
    if not me:
        SYNC.update(error="本机还没有对局结果，认不出账号名")
        return dict(SYNC)
    got, page = [], 1
    try:
        while page <= max_pages:
            req = urllib.request.Request(f"{_url()}/api/v1/matches?users={quote(me)}&ruleset=all&category=all&page={page}&page_size=100",
                                         headers={"Accept": "application/json", "User-Agent": "TavernArena-Connector/1"})
            with urllib.request.urlopen(req, timeout=20, context=ssl.create_default_context()) as r:
                d = json.loads(r.read(2 ** 23))
            items = d.get("matches") if isinstance(d, dict) else d
            if not items:
                break
            got += [m for m in items if isinstance(m, dict)]
            total = d.get("total") if isinstance(d, dict) else None
            if len(items) < 100 or (total is not None and len(got) >= total):
                break
            page += 1
        SITE.parent.mkdir(parents=True, exist_ok=True)
        SITE.write_text(json.dumps(got, ensure_ascii=False), encoding="utf-8")
        SYNC.update(time=time.time(), n=len(got), error=None)
    except Exception as ex:                          # noqa: BLE001
        SYNC.update(error=f"{type(ex).__name__}: {ex}")
    return dict(SYNC)


def _me(matches: list[dict]) -> str | None:
    """自己的账号名：每一局都出现的那个（练习局只有自己）。"""
    sets = [{p.get("username") for p in m.get("participants") or []} for m in matches]
    if not sets:
        return None
    common = set.intersection(*sets)
    if len(common) == 1:
        return common.pop()
    c = Counter(u for s in sets for u in s)
    return c.most_common(1)[0][0] if c else None


def _row(m: dict, me: str | None) -> dict:
    p = next((x for x in m.get("participants") or [] if x.get("username") == me), {})
    lu = next((x for x in m.get("final_lineups") or [] if x.get("username") == me), {})
    return {"id": m.get("id"), "category": m.get("category"), "created": m.get("created"), "finished": m.get("finished"),
            "turns": m.get("turns"), "agents": m.get("human_count"), "bots": m.get("bot_count"),
            "placement": p.get("placement"), "forfeited": p.get("forfeited"), "model": p.get("model_version"),
            "score_before": p.get("score_before"), "score_after": p.get("score_after"),
            "hero": (lu.get("hero") or {}).get("name_zh") or (lu.get("hero") or {}).get("name"),
            "tier": lu.get("tavern_tier"), "records": (RECORDS / f"{m.get('id')}.jsonl").exists(), "live": False}


def summary() -> dict:
    ms = _history()
    me = _me(_local()) or _me(ms)
    rows = [_row(m, me) for m in ms]
    done = {r["id"] for r in rows}
    live = []
    if RECORDS.exists():
        for f in sorted(RECORDS.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
            if f.stem in done or f.stem == "test":
                continue
            age = time.time() - f.stat().st_mtime
            last = {}
            try:
                with open(f, "rb") as fh:
                    fh.seek(max(0, f.stat().st_size - 200_000))
                    tail = fh.read().decode("utf-8", "ignore").splitlines()
                last = json.loads(tail[-1]) if tail else {}
            except (OSError, ValueError):
                pass
            live.append({"id": f.stem, "category": "进行中" if age < 300 else "未完成", "live": age < 300,
                         "created": f.stat().st_ctime, "turns": last.get("turn"), "model": last.get("model"),
                         "place_est": last.get("place"), "records": True, "placement": None})
    return {"me": me, "matches": live + sorted(rows, key=lambda r: r.get("created") or 0, reverse=True),
            "now": time.time(), "sync": SYNC, "site": _url()}


def match(mid: str) -> dict:
    mid = "".join(ch for ch in mid if ch.isalnum())[:64]
    res = next((m for m in _history() if m.get("id") == mid), None)
    recs = []
    f = RECORDS / f"{mid}.jsonl"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    return {"id": mid, "result": res, "records": recs, "me": _me(_local()) or _me(_history())}


class H(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        u = urlsplit(self.path)
        try:
            if u.path == "/":
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif u.path == "/api/summary":
                self._send(200, json.dumps(summary(), ensure_ascii=False).encode("utf-8"), "application/json")
            elif u.path == "/api/sync":
                self._send(200, json.dumps(sync(), ensure_ascii=False).encode("utf-8"), "application/json")
            elif u.path == "/api/match":
                mid = (parse_qs(u.query).get("id") or [""])[0]
                self._send(200, json.dumps(match(mid), ensure_ascii=False).encode("utf-8"), "application/json")
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as ex:                     # noqa: BLE001
            self._send(500, f"{type(ex).__name__}: {ex}".encode("utf-8"), "text/plain; charset=utf-8")

    def log_message(self, *a):
        pass


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>竞技场对局</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1f;--ink2:#3f3f46;--mute:#6b6b70;--line:#e3e1dc;--grid:#ecebe7;--acc:#2f6fdb;--acc-soft:rgba(47,111,219,.10);--good:#1f8a4c;--gold:#b7791f;--bar:#dbe6fb}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--panel:#1d1d21;--ink:#ececf0;--ink2:#c9c9d0;--mute:#9a9aa2;--line:#2e2e34;--grid:#28282d;--acc:#6f9cff;--acc-soft:rgba(111,156,255,.12);--good:#4cc27e;--gold:#e3b04b;--bar:#2a3550}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:18px;margin:0}h2{font-size:16px;margin:0 0 6px}h3{font-size:14px;margin:0 0 2px}small,.mute{color:var(--mute)}
button.btn{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:8px;padding:4px 10px;cursor:pointer;font:inherit}
button.btn:hover{border-color:var(--acc)}
.wrap{padding:16px 20px}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}.chips button{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:999px;padding:4px 12px;cursor:pointer;font:inherit}
.chips button.sel{background:var(--acc);border-color:var(--acc);color:#fff}.chips button.sel small{color:rgba(255,255,255,.85)}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin-bottom:12px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}.tile b{display:block;font-size:22px;font-weight:600}
.charts{display:grid;grid-template-columns:1fr 1.4fr;gap:12px;margin-bottom:16px}@media (max-width:1000px){.charts{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.roll{display:flex;gap:18px;flex-wrap:wrap;margin-top:6px}.roll div b{font-size:16px}
main{display:grid;grid-template-columns:340px 1fr;gap:16px;padding:0 20px 20px}@media (max-width:900px){main{grid-template-columns:1fr}}
.list{max-height:70vh;overflow:auto}.item{padding:8px;border-radius:8px;cursor:pointer;display:flex;gap:10px;align-items:center}
.item:hover,.item.sel{background:var(--bg)}.pl{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;font-weight:700;color:#fff;flex:none}
.p1{background:#c99a12}.p2{background:#7d8ea8}.p3{background:#a86b3c}.p4{background:#2f6fdb}.p5,.p6,.p7,.p8{background:#8a8a92}.plive{background:var(--good)}
.cards{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 10px}.card{border:1px solid var(--line);border-radius:8px;padding:5px 7px;min-width:92px;background:var(--bg)}
.card .n{font-weight:600}.card.g{border-color:var(--gold)}
table{border-collapse:collapse;width:100%}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.dec{padding:6px 8px;border-radius:6px;cursor:pointer}.dec:hover,.dec.sel{background:var(--bg)}
.bar{height:6px;background:var(--bar);border-radius:3px;margin-top:2px}.bar i{display:block;height:100%;background:var(--acc);border-radius:3px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media (max-width:1100px){.grid2{grid-template-columns:1fr}}
.turns{display:flex;flex-wrap:wrap;gap:4px;margin:8px 0}.turns button{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;padding:3px 8px;cursor:pointer}
.turns button.sel{background:var(--acc);color:#fff;border-color:var(--acc)}
pre{white-space:pre-wrap;word-break:break-all;max-height:300px;overflow:auto;background:var(--bg);padding:8px;border-radius:8px;font-size:12px}
svg .ax{fill:var(--mute);font-size:11px}svg .val{fill:var(--ink2);font-size:11px;font-weight:600}
#tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 9px;box-shadow:0 4px 14px rgba(0,0,0,.12);font-size:12px;display:none;z-index:9}
#tip b{font-size:14px}
</style></head><body>
<header><h1>竞技场对局</h1><span id="me" class="mute"></span><button class="btn" onclick="doSync()">同步竞技场战绩</button><span id="sync" class="mute"></span><span class="mute" id="upd"></span></header>
<div class="wrap">
 <div class="chips" id="chips"></div>
 <div class="tiles" id="tiles"></div>
 <div class="charts">
  <div class="panel"><h3>名次分布</h3><div class="mute" style="font-size:12px">第 1～8 名 · 每个名次的局数</div><div id="dist"></div></div>
  <div class="panel"><h3>近期表现</h3><div class="mute" style="font-size:12px" id="recentSub">最近 100 条参赛结果</div><div id="recent"></div><div class="roll" id="roll"></div></div>
 </div>
</div>
<main>
 <section><div class="panel"><div class="mute" style="margin-bottom:6px">对局（新的在上）</div><div class="list" id="list"></div></div></section>
 <section id="detail" class="panel"><div class="mute">选左边一局查看。</div></section>
</main>
<div id="tip"></div>
<script>
const $=s=>document.querySelector(s);let SEL=null,DATA=null,TURN=null,DEC=null,timer=null,SUM=null,CAT='all';
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const cat={standard:'纯 AI',bots:'补位',arena:'补位',practice:'练习',custom:'约战'};
const catKey=c=>c=='arena'?'bots':c;
function fmtT(t){if(!t)return'';const d=new Date(t*1000);return d.toLocaleString('zh-CN',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'})}
const tip=$('#tip');
function showTip(e,lines){tip.replaceChildren();lines.forEach((l,i)=>{const d=document.createElement(i?'div':'b');d.textContent=l;if(i)d.className='mute';tip.appendChild(d)});tip.style.display='block';
 const x=Math.min(e.clientX+14,innerWidth-tip.offsetWidth-8),y=Math.max(8,e.clientY-tip.offsetHeight-10);tip.style.left=x+'px';tip.style.top=y+'px'}
function hideTip(){tip.style.display='none'}
const NS='http://www.w3.org/2000/svg';function el(n,a,p){const e=document.createElementNS(NS,n);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e}
function txt(p,x,y,s,cls,anchor){const t=el('text',{x,y,class:cls||'ax','text-anchor':anchor||'middle'},p);t.textContent=s;return t}

function distChart(games){const box=$('#dist');box.replaceChildren();const cnt=Array(9).fill(0);games.forEach(g=>cnt[g.placement]++);
 const W=460,H=210,L=28,R=8,T=18,B=26,max=Math.max(1,...cnt.slice(1));const step=niceStep(max);const top=Math.ceil(max/step)*step;
 const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,width:'100%',role:'img','aria-label':'名次分布条形图'},box);
 const y=v=>T+(H-T-B)*(1-v/top);for(let v=0;v<=top;v+=step){el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),stroke:'var(--grid)','stroke-width':1},svg);txt(svg,L-6,y(v)+4,v,'ax','end')}
 const band=(W-L-R)/8,bw=Math.min(24,band*0.6);
 for(let p=1;p<=8;p++){const cx=L+band*(p-0.5),h=(H-T-B)-(y(cnt[p])-T),x0=cx-bw/2,y0=y(cnt[p]),yb=H-B;
  if(cnt[p]>0){const r=Math.min(4,h);el('path',{d:`M${x0},${yb}V${y0+r}Q${x0},${y0} ${x0+r},${y0}H${x0+bw-r}Q${x0+bw},${y0} ${x0+bw},${y0+r}V${yb}Z`,fill:'var(--acc)'},svg);txt(svg,cx,y0-5,cnt[p],'val')}
  txt(svg,cx,H-8,'#'+p);
  const hit=el('rect',{x:L+band*(p-1),y:T,width:band,height:H-T-B,fill:'transparent',tabindex:0},svg);
  const lines=[`${cnt[p]} 局`,`第 ${p} 名 · 占 ${games.length?(cnt[p]/games.length*100).toFixed(0):0}%`];
  hit.addEventListener('pointermove',e=>showTip(e,lines));hit.addEventListener('pointerleave',hideTip);
  hit.addEventListener('focus',e=>{const r=hit.getBoundingClientRect();showTip({clientX:r.x+r.width/2,clientY:r.y+40},lines)});hit.addEventListener('blur',hideTip)}
 el('line',{x1:L,x2:W-R,y1:H-B,y2:H-B,stroke:'var(--line)','stroke-width':1},svg)}
function niceStep(m){return m<=5?1:m<=10?2:m<=25?5:m<=50?10:Math.ceil(m/50)*10}

function recentChart(games){const box=$('#recent');box.replaceChildren();const g=games.slice(-100);
 $('#recentSub').textContent=`最近 ${g.length} 条参赛结果（共 ${games.length} 条）`;
 const W=640,H=210,L=26,R=12,T=12,B=26;const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,width:'100%',role:'img','aria-label':'近期名次折线图'},box);
 const y=p=>T+(H-T-B)*(p-1)/7;for(let p=1;p<=8;p++){el('line',{x1:L,x2:W-R,y1:y(p),y2:y(p),stroke:'var(--grid)','stroke-width':1},svg)}[1,4,8].forEach(p=>txt(svg,L-6,y(p)+4,p,'ax','end'));
 txt(svg,L,H-6,'较早','ax','start');txt(svg,W-R,H-6,'较近','ax','end');
 if(!g.length){txt(svg,W/2,H/2,'还没有结果');return}
 const n=g.length,x=i=>n==1?(L+W-R)/2:L+8+i*(W-L-R-16)/(n-1);
 el('path',{d:g.map((m,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(m.placement).toFixed(1)).join(''),fill:'none',stroke:'var(--acc)','stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round'},svg);
 const showDots=n<=40;g.forEach((m,i)=>{if(showDots||i==n-1)el('circle',{cx:x(i),cy:y(m.placement),r:4,fill:'var(--acc)',stroke:'var(--panel)','stroke-width':2},svg)});
 $('#recentSub').textContent+=` · 最近一局第 ${g[n-1].placement} 名`;
 const cross=el('line',{y1:T,y2:H-B,stroke:'var(--mute)','stroke-width':1,visibility:'hidden'},svg);const ring=el('circle',{r:6,fill:'none',stroke:'var(--acc)','stroke-width':2,visibility:'hidden'},svg);
 const hit=el('rect',{x:L,y:T,width:W-L-R,height:H-T-B,fill:'transparent',tabindex:0},svg);
 const pick=e=>{const pt=svg.createSVGPoint();pt.x=e.clientX;pt.y=e.clientY;const q=pt.matrixTransform(svg.getScreenCTM().inverse());let i=Math.round((q.x-L-8)/((W-L-R-16)/Math.max(1,n-1)));i=Math.max(0,Math.min(n-1,i));return i};
 hit.addEventListener('pointermove',e=>{const i=pick(e),m=g[i];cross.setAttribute('x1',x(i));cross.setAttribute('x2',x(i));cross.setAttribute('visibility','visible');ring.setAttribute('cx',x(i));ring.setAttribute('cy',y(m.placement));ring.setAttribute('visibility','visible');
  showTip(e,[`第 ${m.placement} 名`,`${cat[m.category]||m.category} · ${fmtT(m.created)}`,`${m.model||''}${m.score_after!=null?` · Elo ${(m.score_after-m.score_before>=0?'+':'')+(m.score_after-m.score_before).toFixed(1)}`:''}`])});
 hit.addEventListener('pointerleave',()=>{hideTip();cross.setAttribute('visibility','hidden');ring.setAttribute('visibility','hidden')});
 hit.addEventListener('click',e=>{const m=g[pick(e)];if(m)pick_(m.id)})}

function rolling(games){const box=$('#roll');box.replaceChildren();for(const k of [20,50,100]){const g=games.slice(-k);const d=document.createElement('div');
 const b=document.createElement('b');b.textContent=g.length?(g.reduce((a,m)=>a+m.placement,0)/g.length).toFixed(2):'-';const s=document.createElement('small');s.textContent=` 均名 · 样本 ${g.length}`;
 const h=document.createElement('div');h.className='mute';h.style.fontSize='12px';h.textContent=`最近 ${k} 条`;d.append(h,b,s);box.appendChild(d)}}

function renderStats(){if(!SUM)return;const all=SUM.matches.filter(m=>m.placement).sort((a,b)=>(a.created||0)-(b.created||0));
 const cats=['all',...new Set(all.map(m=>catKey(m.category)))];$('#chips').innerHTML=cats.map(c=>`<button class="${c==CAT?'sel':''}" onclick="CAT='${c}';renderStats()">${c=='all'?'全部':esc(cat[c]||c)} <small>${c=='all'?all.length:all.filter(m=>catKey(m.category)==c).length}</small></button>`).join('');
 const g=CAT=='all'?all:all.filter(m=>catKey(m.category)==CAT);
 const avg=g.length?g.reduce((a,m)=>a+m.placement,0)/g.length:null,t1=g.filter(m=>m.placement==1).length,t4=g.filter(m=>m.placement<=4).length;
 let t=`<div class="tile"><small>参赛结果</small><b>${g.length}</b></div><div class="tile"><small>平均名次</small><b>${avg?avg.toFixed(2):'-'}</b></div><div class="tile"><small>第一</small><b>${g.length?(t1/g.length*100).toFixed(0)+'%':'-'}</b><small>${t1} 局</small></div><div class="tile"><small>前四</small><b>${g.length?(t4/g.length*100).toFixed(0)+'%':'-'}</b><small>${t4} 局</small></div>`;
 const elo=g.filter(m=>m.score_after!=null);if(elo.length)t+=`<div class="tile"><small>纯 AI Elo</small><b>${elo.at(-1).score_after.toFixed(1)}</b><small>${elo.length} 局${elo.length>=5?'':' · 满 5 局才有正式排名'}</small></div>`;
 const models={};g.forEach(m=>{const k=m.model||'?';(models[k]=models[k]||[]).push(m.placement)});
 t+=`<div class="tile" style="grid-column:span 2"><small>按模型版本</small>${Object.entries(models).map(([k,v])=>`<div>${esc(k)}：${v.length} 局，平均 <b style="display:inline;font-size:14px">${(v.reduce((a,b)=>a+b,0)/v.length).toFixed(2)}</b> 名</div>`).join('')||'-'}</div>`;
 $('#tiles').innerHTML=t;distChart(g);recentChart(g);rolling(g)}

async function loadSummary(){const r=await fetch('/api/summary');SUM=await r.json();const s=SUM;$('#me').textContent=s.me?('账号 '+s.me):'';$('#upd').textContent='更新于 '+new Date().toLocaleTimeString('zh-CN');
 const sy=s.sync||{};$('#sync').textContent=sy.error?('同步失败：'+String(sy.error).slice(0,80)):sy.time?(`已同步 ${sy.n} 局（${new Date(sy.time*1000).toLocaleTimeString('zh-CN')}）`):'';
 renderStats();
 $('#list').innerHTML=s.matches.map(m=>`<div class="item ${m.id==SEL?'sel':''}" onclick="pick_('${esc(m.id)}')"><div class="pl ${m.placement?'p'+m.placement:'plive'}">${m.placement||(m.live?'●':'?')}</div><div><div>${esc(cat[m.category]||m.category)}${m.hero?' · '+esc(m.hero):''}${m.tier?' '+esc(m.tier)+' 本':''}</div><small>${fmtT(m.created)} · ${m.turns??'-'} 回合 · ${esc(m.model||'')}${m.score_after!=null?` · Elo ${(m.score_after-m.score_before>=0?'+':'')+(m.score_after-m.score_before).toFixed(1)}`:''}${m.live&&m.place_est?` · 预计 ${m.place_est} 名`:''}${m.records?'':' · 无逐步记录'}</small></div></div>`).join('')||'<div class="mute">还没有对局。</div>'}
async function doSync(){$('#sync').textContent='同步中…';await fetch('/api/sync');loadSummary()}
async function pick_(id){SEL=id;TURN=null;DEC=null;await loadMatch();loadSummary();$('#detail').scrollIntoView({behavior:'smooth',block:'start'})}
async function loadMatch(){if(!SEL)return;const r=await fetch('/api/match?id='+encodeURIComponent(SEL));DATA=await r.json();
 const turns=[...new Set(DATA.records.map(x=>x.turn))];if(TURN==null||!turns.includes(TURN))TURN=turns.at(-1);render();
 clearTimeout(timer);if(!DATA.result)timer=setTimeout(loadMatch,3000)}
function cards(lst){if(!Array.isArray(lst)||!lst.length)return'<div class="mute">（空）</div>';return'<div class="cards">'+lst.map(c=>{if(!c||typeof c!='object')return'';const a=c.attack??c.atk,h=c.health??c.hp;const g=c.golden||String(c.id||'').endsWith('_G');
 return`<div class="card ${g?'g':''}"><div class="n">${g?'★ ':''}${esc(c.name_zh||c.name||c.id||'?')}</div><div class="mute">${a!=null?esc(a)+'/'+esc(h):''}${c.tier?' · '+esc(c.tier)+' 本':''}${c.cost!=null?' · '+esc(c.cost)+' 金':''}</div></div>`}).join('')+'</div>'}
function zone(me,keys){for(const k of keys){const v=me[k];if(Array.isArray(v))return v;if(v&&typeof v=='object')for(const kk of['cards','options','minions','items'])if(Array.isArray(v[kk]))return v[kk]}return null}
function scalars(me){return Object.entries(me).filter(([k,v])=>v!=null&&typeof v!='object').slice(0,24).map(([k,v])=>`<span style="margin-right:12px"><small>${esc(k)}</small> <b>${esc(v)}</b></span>`).join('')}
function curve(recs){const pts=recs.filter(r=>r.place!=null);if(pts.length<2)return'';const W=640,Hh=120,n=pts.length;
 const x=i=>30+i*(W-40)/(n-1),y=p=>10+(p-1)*(Hh-20)/7;let d=pts.map((r,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(r.place).toFixed(1)).join(' ');
 let marks='';let last=null;pts.forEach((r,i)=>{if(r.turn!==last){marks+=`<line x1="${x(i)}" x2="${x(i)}" y1="10" y2="${Hh-10}" stroke="var(--grid)"/><text class="ax" x="${x(i)+2}" y="${Hh-1}">${esc(r.turn)}</text>`;last=r.turn}});
 return`<svg viewBox="0 0 ${W} ${Hh+4}" width="100%" style="max-width:${W}px">${[1,4,8].map(p=>`<text class="ax" x="2" y="${y(p)+4}">${p}</text>`).join('')}${marks}<path d="${d}" fill="none" stroke="var(--acc)" stroke-width="2"/></svg>`}
function render(){const D=DATA;if(!D)return;const res=D.result,recs=D.records;let h='';
 if(res){const me=(res.participants||[]).find(p=>p.username==D.me)||{};h+=`<h2>第 ${esc(me.placement??'?')} 名 <small>${esc(cat[res.category]||res.category)} · ${esc(res.turns)} 回合 · ${Math.round(res.duration_seconds||0)} 秒 · ${esc(res.human_count)} 个 AI + ${esc(res.bot_count)} 个机器人</small></h2>`;
  if(me.score_after!=null)h+=`<div>Elo ${me.score_before.toFixed(1)} → <b>${me.score_after.toFixed(1)}</b></div>`}
 else h+=`<h2>进行中 <small>每 3 秒刷新</small></h2>`;
 if(recs.length)h+=`<div class="mute" style="margin-top:8px">模型预计名次（每一步；越低越好）</div>`+curve(recs);
 const turns=[...new Set(recs.map(x=>x.turn))];if(turns.length)h+='<div class="turns">'+turns.map(t=>`<button class="${t==TURN?'sel':''}" onclick="TURN=${JSON.stringify(t)};DEC=null;render()">${t==0?'选将':'第 '+esc(t)+' 回合'}</button>`).join('')+'</div>';
 const tr=recs.filter(r=>r.turn==TURN);if(DEC==null||!tr.find(r=>r.n==DEC))DEC=tr.length?tr[0].n:null;const cur=tr.find(r=>r.n==DEC);
 let view=null;for(const r of recs){if(r.n>DEC)break;if(r.view&&r.view.opponents)view=Object.assign({},r.view);if(r.n==DEC&&r.view)view=Object.assign({},view||{},r.view)}
 if(tr.length){h+='<div class="grid2"><div><div class="mute">这一回合的操作</div>'+tr.map(r=>`<div class="dec ${r.n==DEC?'sel':''}" onclick="DEC=${r.n};render()"><b>${esc(r.text)}</b>${r.guard?` <small>（${esc(r.guard)}）</small>`:''}<small class="mute"> · 预计 ${esc(r.place??'-')} 名 · 已用 ${esc(r.used??'-')} 次</small></div>`).join('')+'</div>';
  h+='<div>';if(cur){h+='<div class="mute">这一步的候选（模型概率）</div>'+(cur.top||[]).map(([a,p,t])=>`<div style="margin:4px 0">${a==cur.action?'▶ ':''}${esc(t)} <small class="mute">${(p*100).toFixed(1)}%</small><div class="bar"><i style="width:${(p*100).toFixed(1)}%"></i></div></div>`).join('')}h+='</div></div>';
  const me=(view&&view.me)||{};h+='<div style="margin-top:12px">'+scalars(me)+'</div>';
  for(const [lbl,keys] of [['场面',['board','warband']],['手牌',['hand']],['酒馆',['shop','tavern','store']],['待选',['choices','options','pending','discover','choice']]]){const z=zone(me,keys);if(z)h+=`<div class="mute">${lbl}</div>`+cards(z)}
  if(view)h+=`<details><summary class="mute">原始可见局面（JSON）</summary><pre>${esc(JSON.stringify(view,null,1))}</pre></details>`}
 if(res&&res.final_lineups&&res.final_lineups.length){h+='<div class="mute" style="margin-top:12px">最终 8 席</div><table><tr><th>名次</th><th>选手</th><th>英雄</th><th>等级</th><th>最后阵容</th></tr>'+[...res.final_lineups].sort((a,b)=>a.placement-b.placement).map(l=>`<tr style="${l.username==D.me?'font-weight:700':''}"><td>${esc(l.placement)}${l.forfeited?' <small>退赛</small>':''}</td><td>${esc(l.agent||l.kind)}<br><small>${esc(l.username||'')}</small></td><td>${esc(l.hero?.name_zh||'')}</td><td>${esc(l.tavern_tier)}</td><td>${(l.board||[]).map(c=>esc((c.golden?'★':'')+c.name_zh+' '+c.attack+'/'+c.health)).join('，')}</td></tr>`).join('')+'</table>'}
 if(!recs.length)h+='<div class="mute" style="margin-top:8px">这一局没有逐步记录（记录功能加上之前打的，或从竞技场同步来的）。</div>';
 $('#detail').innerHTML=h}
loadSummary();setInterval(loadSummary,10000);
</script></body></html>"""


def print_last(path: str) -> int:
    """arena_loop.bat 用：把一局结果打印成一行（不打印整段 JSON）。"""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8").strip().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        print("（没有结果）")
        return 1
    m = d.get("match") or {}
    me = _me(_history() or [m])
    p = next((x for x in m.get("participants") or [] if x.get("username") == me), {})
    elo = (f"，Elo {p['score_before']:.1f} → {p['score_after']:.1f}" if p.get("score_after") is not None else "")
    print(f"结果：第 {p.get('placement')} 名（{m.get('category')}，{m.get('human_count')} 个 AI + {m.get('bot_count')} 个机器人，"
          f"{m.get('turns')} 回合）{elo}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--print-last", default="")
    a = ap.parse_args()
    if a.print_last:
        return print_last(a.print_last)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    threading.Thread(target=sync, daemon=True).start()          # 启动时同步一次公开战绩
    print(f"竞技场对局查看器：http://127.0.0.1:{a.port}/  （关掉窗口即停止）", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
