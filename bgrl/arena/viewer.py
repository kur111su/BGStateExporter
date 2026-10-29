"""竞技场对局查看器（只在本机）：http://127.0.0.1:8767/

读 arena/results/history.jsonl（每局结束时连接器给的结果：名次、积分、8 席最终阵容）和
arena/records/<对局>.jsonl（agent.py 每一步记下的可见局面、模型前 5 个选择和预计名次），显示：
  - 战绩：各赛道局数、平均名次、第一 / 前四比例、Elo 变化，按模型版本分开；
  - 单局：预计名次曲线、每回合每一步（选了什么、前 5 个候选的概率）、当时的场面 / 手牌 / 酒馆、最终 8 席阵容；
  - 正在打的对局每几秒刷新。
只用标准库；不连网，不读密钥。用法：arena_view.bat（或 python arena/viewer.py --port 8767）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results" / "history.jsonl"
RECORDS = HERE / "records"
LOGS = HERE / "logs"


def _history() -> list[dict]:
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
    me = _me(ms)
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
            "now": time.time()}


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
    return {"id": mid, "result": res, "records": recs, "me": _me(_history())}


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
:root{--bg:#f6f5f2;--panel:#fff;--ink:#1d1d1f;--mute:#6b6b70;--line:#e3e1dc;--acc:#2f6fdb;--good:#1f8a4c;--bad:#c2410c;--gold:#b7791f;--bar:#dbe6fb}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--panel:#1d1d21;--ink:#ececf0;--mute:#9a9aa2;--line:#2e2e34;--acc:#6f9cff;--good:#4cc27e;--bad:#f08a5d;--gold:#e3b04b;--bar:#2a3550}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:18px;margin:0}small,.mute{color:var(--mute)}
main{display:grid;grid-template-columns:340px 1fr;gap:16px;padding:16px 20px}
@media (max-width:900px){main{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;margin-bottom:12px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px}.tile b{display:block;font-size:20px}
.list{max-height:70vh;overflow:auto}.item{padding:8px;border-radius:8px;cursor:pointer;display:flex;gap:10px;align-items:center}
.item:hover,.item.sel{background:var(--bg)}.pl{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;font-weight:700;color:#fff;flex:none}
.p1{background:#c99a12}.p2{background:#7d8ea8}.p3{background:#a86b3c}.p4{background:#2f6fdb}.p5,.p6,.p7,.p8{background:#8a8a92}.plive{background:var(--good)}
.cards{display:flex;flex-wrap:wrap;gap:6px;margin:4px 0 10px}.card{border:1px solid var(--line);border-radius:8px;padding:5px 7px;min-width:92px;background:var(--bg)}
.card .n{font-weight:600}.card.g{border-color:var(--gold)}.card .s{font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.dec{padding:6px 8px;border-radius:6px;cursor:pointer}.dec:hover,.dec.sel{background:var(--bg)}
.bar{height:6px;background:var(--bar);border-radius:3px;margin-top:2px}.bar i{display:block;height:100%;background:var(--acc);border-radius:3px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media (max-width:1100px){.grid2{grid-template-columns:1fr}}
.turns{display:flex;flex-wrap:wrap;gap:4px;margin:8px 0}.turns button{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:6px;padding:3px 8px;cursor:pointer}
.turns button.sel{background:var(--acc);color:#fff;border-color:var(--acc)}
pre{white-space:pre-wrap;word-break:break-all;max-height:300px;overflow:auto;background:var(--bg);padding:8px;border-radius:8px;font-size:12px}
svg text{fill:var(--mute);font-size:11px}
</style></head><body>
<header><h1>竞技场对局</h1><span id="me" class="mute"></span><span class="mute" id="upd"></span></header>
<main>
 <section><div class="tiles" id="tiles"></div><div class="panel"><div class="mute" style="margin-bottom:6px">对局（新的在上）</div><div class="list" id="list"></div></div></section>
 <section id="detail" class="panel"><div class="mute">选左边一局查看。</div></section>
</main>
<script>
const $=s=>document.querySelector(s);let SEL=null,DATA=null,TURN=null,DEC=null,timer=null;
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const cat={standard:'纯 AI',bots:'补位',arena:'补位',practice:'练习',custom:'约战'};
function fmtT(t){if(!t)return'';const d=new Date(t*1000);return d.toLocaleString('zh-CN',{month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'})}
async function loadSummary(){const r=await fetch('/api/summary');const s=await r.json();$('#me').textContent=s.me?('账号 '+s.me):'';$('#upd').textContent='更新于 '+new Date().toLocaleTimeString('zh-CN');
 const fin=s.matches.filter(m=>m.placement);const by={};for(const m of fin){(by[m.category]=by[m.category]||[]).push(m)}
 let tiles=`<div class="tile"><small>总局数</small><b>${fin.length}</b></div>`;
 for(const [c,a] of Object.entries(by)){const avg=a.reduce((x,m)=>x+m.placement,0)/a.length;const t1=a.filter(m=>m.placement==1).length/a.length;const t4=a.filter(m=>m.placement<=4).length/a.length;
  tiles+=`<div class="tile"><small>${cat[c]||c} ${a.length} 局</small><b>${avg.toFixed(2)}</b><small>平均名次 · 第一 ${(t1*100).toFixed(0)}% · 前四 ${(t4*100).toFixed(0)}%</small></div>`}
 const elo=fin.filter(m=>m.score_after!=null).sort((a,b)=>a.created-b.created);if(elo.length){tiles+=`<div class="tile"><small>纯 AI Elo（${elo.length} 局）</small><b>${elo.at(-1).score_after.toFixed(1)}</b><small>${elo.length>=5?'已满 5 局':'满 5 局才有正式排名'}</small></div>`}
 const models={};for(const m of fin){const k=m.model||'?';(models[k]=models[k]||[]).push(m.placement)}
 tiles+=`<div class="tile" style="grid-column:1/-1"><small>按模型版本</small>${Object.entries(models).map(([k,v])=>`<div>${esc(k)}：${v.length} 局，平均 ${(v.reduce((a,b)=>a+b,0)/v.length).toFixed(2)} 名</div>`).join('')}</div>`;
 $('#tiles').innerHTML=tiles;
 $('#list').innerHTML=s.matches.map(m=>`<div class="item ${m.id==SEL?'sel':''}" onclick="pick('${m.id}')"><div class="pl ${m.placement?'p'+m.placement:'plive'}">${m.placement||(m.live?'●':'?')}</div><div><div>${esc(cat[m.category]||m.category)} · ${esc(m.hero||'')} ${m.tier?esc(m.tier)+' 本':''}</div><small>${fmtT(m.created)} · ${m.turns??'-'} 回合 · ${esc(m.model||'')}${m.score_after!=null?` · Elo ${(m.score_after-m.score_before>=0?'+':'')+(m.score_after-m.score_before).toFixed(1)}`:''}${m.live&&m.place_est?` · 预计 ${m.place_est} 名`:''}${m.records?'':' · 无逐步记录'}</small></div></div>`).join('')||'<div class="mute">还没有对局。</div>'}
async function pick(id){SEL=id;TURN=null;DEC=null;await loadMatch();loadSummary()}
async function loadMatch(){if(!SEL)return;const r=await fetch('/api/match?id='+SEL);DATA=await r.json();
 const turns=[...new Set(DATA.records.map(x=>x.turn))];if(TURN==null||!turns.includes(TURN))TURN=turns.at(-1);render();
 clearTimeout(timer);if(!DATA.result)timer=setTimeout(loadMatch,3000)}
function cards(lst){if(!Array.isArray(lst)||!lst.length)return'<div class="mute">（空）</div>';return'<div class="cards">'+lst.map(c=>{if(!c||typeof c!='object')return'';const a=c.attack??c.atk,h=c.health??c.hp;const g=c.golden||String(c.id||'').endsWith('_G');
 return`<div class="card ${g?'g':''}"><div class="n">${g?'★ ':''}${esc(c.name_zh||c.name||c.id||'?')}</div><div class="s mute">${a!=null?a+'/'+h:''}${c.tier?' · '+c.tier+' 本':''}${c.cost!=null?' · '+c.cost+' 金':''}</div></div>`}).join('')+'</div>'}
function zone(me,keys){for(const k of keys){const v=me[k];if(Array.isArray(v))return v;if(v&&typeof v=='object')for(const kk of['cards','options','minions','items'])if(Array.isArray(v[kk]))return v[kk]}return null}
function scalars(me){return Object.entries(me).filter(([k,v])=>v!=null&&typeof v!='object').slice(0,24).map(([k,v])=>`<span style="margin-right:12px"><small>${esc(k)}</small> <b>${esc(v)}</b></span>`).join('')}
function curve(recs){const pts=recs.filter(r=>r.place!=null);if(pts.length<2)return'';const W=640,Hh=120,n=pts.length;
 const x=i=>30+i*(W-40)/(n-1),y=p=>10+(p-1)*(Hh-20)/7;let d=pts.map((r,i)=>(i?'L':'M')+x(i).toFixed(1)+' '+y(r.place).toFixed(1)).join(' ');
 let marks='';let last=null;pts.forEach((r,i)=>{if(r.turn!==last){marks+=`<line x1="${x(i)}" x2="${x(i)}" y1="10" y2="${Hh-10}" stroke="var(--line)"/><text x="${x(i)+2}" y="${Hh-1}">${r.turn}</text>`;last=r.turn}});
 return`<svg viewBox="0 0 ${W} ${Hh+4}" width="100%" style="max-width:${W}px">${[1,4,8].map(p=>`<text x="2" y="${y(p)+4}">${p}</text>`).join('')}${marks}<path d="${d}" fill="none" stroke="var(--acc)" stroke-width="2"/></svg>`}
function render(){const D=DATA;if(!D)return;const res=D.result,recs=D.records;let h='';
 if(res){const me=(res.participants||[]).find(p=>p.username==D.me)||{};h+=`<h2 style="margin:0 0 4px">第 ${me.placement??'?'} 名 <small>${esc(cat[res.category]||res.category)} · ${res.turns} 回合 · ${Math.round(res.duration_seconds||0)} 秒 · ${res.human_count} 个 AI + ${res.bot_count} 个机器人</small></h2>`;
  if(me.score_after!=null)h+=`<div>Elo ${me.score_before.toFixed(1)} → <b>${me.score_after.toFixed(1)}</b></div>`}
 else h+=`<h2 style="margin:0 0 4px">进行中 <small>每 3 秒刷新</small></h2>`;
 h+=`<div class="mute" style="margin-top:8px">模型预计名次（每一步；越低越好）</div>`+curve(recs);
 const turns=[...new Set(recs.map(x=>x.turn))];h+='<div class="turns">'+turns.map(t=>`<button class="${t==TURN?'sel':''}" onclick="TURN=${JSON.stringify(t)};DEC=null;render()">${t==0?'选将':'第 '+t+' 回合'}</button>`).join('')+'</div>';
 const tr=recs.filter(r=>r.turn==TURN);if(DEC==null||!tr.find(r=>r.n==DEC))DEC=tr.length?tr[0].n:null;const cur=tr.find(r=>r.n==DEC);
 let view=null;for(const r of recs){if(r.n>DEC)break;if(r.view&&r.view.opponents)view=Object.assign({},r.view);if(r.n==DEC&&r.view)view=Object.assign({},view||{},r.view)}
 h+='<div class="grid2"><div><div class="mute">这一回合的操作</div>'+tr.map(r=>`<div class="dec ${r.n==DEC?'sel':''}" onclick="DEC=${r.n};render()"><b>${esc(r.text)}</b>${r.guard?` <small>（${esc(r.guard)}）</small>`:''}<small class="mute"> · 预计 ${r.place??'-'} 名 · 已用 ${r.used??'-'} 次</small></div>`).join('')+'</div>';
 h+='<div>';if(cur){h+='<div class="mute">这一步的候选（模型概率）</div>'+(cur.top||[]).map(([a,p,t])=>`<div style="margin:4px 0">${a==cur.action?'▶ ':''}${esc(t)} <small class="mute">${(p*100).toFixed(1)}%</small><div class="bar"><i style="width:${(p*100).toFixed(1)}%"></i></div></div>`).join('')}
 h+='</div></div>';
 const me=(view&&view.me)||{};h+='<div style="margin-top:12px">'+scalars(me)+'</div>';
 for(const [lbl,keys] of [['场面',['board','warband']],['手牌',['hand']],['酒馆',['shop','tavern','store']],['待选',['choices','options','pending','discover','choice']]]){const z=zone(me,keys);if(z)h+=`<div class="mute">${lbl}</div>`+cards(z)}
 if(view)h+=`<details><summary class="mute">原始可见局面（JSON）</summary><pre>${esc(JSON.stringify(view,null,1))}</pre></details>`;
 if(res&&res.final_lineups){h+='<div class="mute" style="margin-top:12px">最终 8 席</div><table><tr><th>名次</th><th>选手</th><th>英雄</th><th>等级</th><th>最后阵容</th></tr>'+[...res.final_lineups].sort((a,b)=>a.placement-b.placement).map(l=>`<tr style="${l.username==D.me?'font-weight:700':''}"><td>${l.placement}${l.forfeited?' <small>退赛</small>':''}</td><td>${esc(l.agent||l.kind)}<br><small>${esc(l.username||'')}</small></td><td>${esc(l.hero?.name_zh||'')}</td><td>${l.tavern_tier}</td><td>${(l.board||[]).map(c=>esc((c.golden?'★':'')+c.name_zh+' '+c.attack+'/'+c.health)).join('，')}</td></tr>`).join('')+'</table>'}
 if(!recs.length)h+='<div class="mute" style="margin-top:8px">这一局没有逐步记录（记录功能加上之前打的）。</div>';
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
    print(f"竞技场对局查看器：http://127.0.0.1:{a.port}/  （关掉窗口即停止）", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
