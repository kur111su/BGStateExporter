"""C1 影子模式：读插件的实时局面，算出模型（PPO）给的本回合方案，显示在 http://localhost:8766/。
只显示，不操作游戏。插件 v0.10 起去掉了自己的搜索建议器和 8765 网页面板，这一页是唯一的界面：
  模型建议（本回合方案、预计名次、第一步备选）+ 插件的 Bob's Buddy 模拟（本场胜率、对各对手胜率、站位优化按钮）。

用法：python shadow.py [--run runs/v3 | --ckpt 某个检查点.pt] [--port 8766] [--device cpu]
  训练还在跑时，latest.pt 每次更新后（换局时）自动换成新模型。
  每回合的方案记在 bgrl/reports/shadow/日期.jsonl，之后可以和你的实际操作对照。
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from paths import DEFAULT_RUN, PLUGIN_PORT, REPORTS, SHADOW_PORT  # noqa: E402

from tg.bridge import build_game  # noqa: E402
from tg.logs import GameHistory, _key, _normalize  # noqa: E402

PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>模型建议（影子模式）</title>
<style>
:root{--bg:#f6f5f2;--fg:#1d1d1b;--mut:#6b6a66;--card:#fff;--line:#e3e1dc;--acc:#2f6fd1;--good:#2e8b57;--warn:#b7791f}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#ecebe8;--mut:#a09e99;--card:#212121;--line:#333;--acc:#7aa7f0;--good:#6cc08b;--warn:#e0a84a}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:720px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 4px} .mut{color:var(--mut);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0}
ol{margin:6px 0 0;padding-left:22px} li{margin:3px 0} .p{color:var(--mut);font-size:12px;margin-left:6px}
.big{font-size:26px;font-weight:600} .row{display:flex;gap:18px;flex-wrap:wrap;align-items:baseline}
.bar{display:flex;gap:2px;align-items:flex-end;height:40px;margin-top:6px}.bar div{flex:1;background:var(--acc);opacity:.75;border-radius:2px 2px 0 0}
.bl{display:flex;gap:2px}.bl span{flex:1;text-align:center;font-size:11px;color:var(--mut)}
table{width:100%;border-collapse:collapse;font-size:14px} td{padding:3px 4px;border-top:1px solid var(--line)} td.n{text-align:right;white-space:nowrap;color:var(--mut)}
.note{color:var(--warn);font-size:13px;margin-top:6px} .st{font-size:13px}
button{font:inherit;padding:4px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.next{font-weight:600}
</style></head><body><main>
<h1>模型建议 <span class="mut">影子模式：只看不操作，你照常打</span></h1>
<div class="st mut" id="st">连接中…</div>
<div id="out"></div>
<div id="sim"></div>
</main><script>
const $=s=>document.querySelector(s);
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function tick(){
 try{const r=await fetch('/advice.json',{cache:'no-store'});const d=await r.json();render(d)}catch(e){$('#st').textContent='影子服务没有响应'}
 setTimeout(tick,1000)}
const pct=x=>x==null?'-':(x*100).toFixed(0)+'%';
function renderSim(sm){
 if(!sm){$('#sim').innerHTML='';return}
 if(!sm.available){$('#sim').innerHTML=`<div class="card mut">Bob's Buddy 模拟不可用${sm.message?'：'+esc(sm.message):''}</div>`;return}
 let h='';const lv=sm.live;
 if(lv&&lv.hasResult)h+=`<div class="card"><b>${lv.state==='Combat'?'本场':'上一场'}战斗（第 ${lv.turn} 回合，HDT 的 Bob's Buddy）</b>
 <div class="row"><div>胜 <span class="big">${pct(lv.winRate)}</span></div><div>平 ${pct(lv.tieRate)}</div><div>负 ${pct(lv.lossRate)}</div>
 <div class="mut">斩杀 ${pct(lv.winLethal)} · 被斩杀 ${pct(lv.lossLethal)}</div></div></div>`;
 const mu=sm.matchups||{};const rs=mu.results||[];
 if(rs.length||mu.error)h+=`<div class="card"><b>对各对手胜率</b> <span class="mut">用你现在的场面打对手上次的阵容${mu.status==='running'?' · 计算中…':''}</span>
 ${mu.error?`<div class="note">${esc(mu.error)}</div>`:''}<table>${rs.map(r=>`<tr><td class="${r.isNextOpponent?'next':''}">${esc(r.heroName||r.heroCardId||'?')}${r.isNextOpponent?'（下个对手）':''}
 <span class="p">第 ${r.boardSeenTurn} 回合的阵容</span></td>${r.error?`<td colspan=3 class="n">${esc(r.error)}</td>`:`<td class="n">胜 ${pct(r.winRate)}</td><td class="n">平 ${pct(r.tieRate)}</td><td class="n">负 ${pct(r.lossRate)}</td>`}</tr>`).join('')}</table></div>`;
 const op=sm.optimizer||{};
 h+=`<div class="card"><b>站位优化</b> <button id="optb">对下个对手优化站位</button>`;
 if(op.status==='running')h+=`<div class="mut">计算中，已试 ${op.evaluated||0} 种站位…</div>`;
 if(op.message)h+=`<div class="${op.status==='error'?'note':'mut'}">${esc(op.message)}</div>`;
 if(op.status==='done'&&op.best){h+=`<div>对 ${esc(op.opponentHeroName||'')}：当前 胜 ${pct(op.current&&op.current.winRate)} / 负 ${pct(op.current&&op.current.lossRate)} →
 建议 胜 ${pct(op.best.winRate)} / 负 ${pct(op.best.lossRate)}</div><ol>${(op.bestOrderNames||[]).map(n=>`<li>${esc(n)}</li>`).join('')}</ol>`}
 h+=`</div>`;
 $('#sim').innerHTML=h;
 $('#optb').onclick=async()=>{try{const r=await fetch('/optimize',{cache:'no-store'});const j=await r.json();if(!j.accepted)alert(j.message||'没有接受')}catch(e){}};
}
function render(d){
 renderSim(d.simulation);
 const m=d.model||{};
 $('#st').innerHTML=esc(d.status||'')+(m.path?` · 模型 ${esc(m.name||'')}${m.iter!=null?' 第 '+m.iter+' 轮':''}`:'');
 const a=d.advice; if(!a){$('#out').innerHTML='';return}
 const pd=a.placeDist||[];const mx=Math.max(...pd,0.01);
 let h=`<div class="card"><div class="row"><div><div class="mut">第 ${a.turn} 回合 · 预计名次</div><div class="big">${a.expectedPlace}</div></div>
 <div style="flex:1;min-width:200px"><div class="bar">${pd.map(x=>`<div style="height:${Math.max(2,x/mx*40)}px" title="${(x*100).toFixed(0)}%"></div>`).join('')}</div>
 <div class="bl">${pd.map((_,i)=>`<span>${i+1}</span>`).join('')}</div></div></div></div>`;
 h+=`<div class="card"><b>本回合方案</b><ol>${(a.steps||[]).map(s=>`<li>${esc(s.text)}<span class="p">${(s.prob*100).toFixed(0)}%</span></li>`).join('')}</ol>`;
 if(a.note)h+=`<div class="note">${esc(a.note)}</div>`;
 h+=`</div><div class="card"><b>第一步的备选</b><table>${(a.alternatives||[]).map(s=>`<tr><td>${esc(s.text)}</td><td class="n">${(s.prob*100).toFixed(0)}%</td><td class="n">之后预计 ${s.placeAfter} 名</td></tr>`).join('')}</table></div>`;
 if(a.warnings&&a.warnings.length)h+=`<div class="mut">重建提示：${a.warnings.map(esc).join('；')}</div>`;
 h+=`<div class="mut">计算 ${a.ms} ms · 局面 #${a.seq}</div>`;
 $('#out').innerHTML=h}
tick();
</script></body></html>"""


class Shadow:
    def __init__(self, a):
        self.a = a
        self.status = "等待插件局面"
        self.advice = None
        self.sim = None                                   # 插件的 Bob's Buddy 模拟结果（原样转给页面）
        self.model = self.advisor = None
        self.minfo = {}
        self.hist = GameHistory()                         # 对手上次阵容、已买的黑暗之赐（和回放共用）
        self.game_id = None
        self.last_key = None
        self.pending_key = None
        self.lock = threading.Lock()
        self.logdir = REPORTS / "shadow"
        self.logdir.mkdir(parents=True, exist_ok=True)

    # ---------------- 模型 ----------------

    def ckpt_path(self) -> Path | None:
        if self.a.ckpt:
            return Path(self.a.ckpt)
        p = Path(self.a.run) / "latest.pt"
        if p.exists():
            return p
        # 新阶段刚开始、还没存第一个检查点时：用最近更新过的其它训练目录的 latest.pt
        others = sorted(Path(self.a.run).parent.glob("*/latest.pt"), key=lambda x: x.stat().st_mtime)
        return others[-1] if others else None

    def maybe_reload(self):
        from rl.advisor import Advisor, load_model
        p = self.ckpt_path()
        if p is None:
            self.status = f"找不到模型：{self.a.ckpt or Path(self.a.run) / 'latest.pt'}（训练每 10 轮存一次）"
            return
        if self.minfo.get("path") == str(p) and self.minfo.get("mtime") == p.stat().st_mtime:
            return
        for _ in range(3):                         # 训练程序可能正在写这个文件
            try:
                self.model, self.minfo = load_model(p, self.a.device)
                break
            except Exception as ex:                # noqa: BLE001
                err = ex
                time.sleep(1.0)
        else:
            self.status = f"读取模型失败：{err}"
            return
        self.minfo["name"] = p.name
        self.advisor = Advisor(self.model, self.a.device)
        print(f"已加载模型 {p}（第 {self.minfo.get('iter')} 轮，{self.minfo.get('size')}）", flush=True)

    # ---------------- 局面 ----------------

    def fetch(self) -> dict | None:
        try:
            with urllib.request.urlopen(f"http://localhost:{self.a.plugin_port}/state.json", timeout=3) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:                          # noqa: BLE001
            return None

    def step(self):
        s = self.fetch()
        if s is None:
            self.status = f"连不上插件（http://localhost:{self.a.plugin_port}/），请确认 HDT 和插件在运行"
            return
        self.sim = s.get("simulation") if s.get("isBattlegrounds") else None
        if not s.get("inGame") or not s.get("isBattlegrounds"):
            self.status = "不在酒馆战旗对局中"
            return
        if s.get("mode") not in (None, "solo"):
            self.status = "双人模式暂不支持"
            return
        s = _normalize(s)
        p = s.get("player") or {}
        turn = int(s.get("turn") or 0)
        hero = p.get("heroCardId")
        if p and (self.game_id is None or hero != self.game_id or turn < self.last_turn_seen):
            self.hist = GameHistory()                     # 新的一局
            self.game_id = hero
            self.last_key = self.pending_key = None
            self.maybe_reload()                           # 换局时才换模型
        self.last_turn_seen = turn
        self.hist.feed(s)
        if s.get("phase") == "combat":
            self.status = f"第 {turn} 回合战斗中"
            return
        if s.get("phase") != "recruit" or not p:
            self.status = f"阶段：{s.get('phase')}"
            return
        if self.advisor is None:
            self.maybe_reload()
            if self.advisor is None:
                return
        if s.get("offered"):
            self.status = f"第 {turn} 回合：有待选项（{(s['offered'] or {}).get('kind')}），影子模式第一版暂不处理，选完后给方案"
            return
        key = (_key(s), turn)
        if key == self.last_key:
            self.status = f"第 {turn} 回合招募中"
            return
        if key != self.pending_key:                       # 等局面稳定（连续两次一样）再算
            self.pending_key = key
            return
        self.last_key = key
        g, warns = build_game(self.hist.annotate(s))
        adv = self.advisor.plan(g)
        adv.update(turn=turn, seq=s.get("seq"), time=datetime.now().isoformat(timespec="seconds"), warnings=warns[:5])
        with self.lock:
            self.advice = adv
        self.status = f"第 {turn} 回合招募中（已更新方案）"
        rec = {"time": adv["time"], "seq": s.get("seq"), "turn": turn, "hero": p.get("heroCardId"),
               "model": {k: self.minfo.get(k) for k in ("name", "iter")}, "steps": [x["text"] for x in adv["steps"]],
               "actions": [x["action"] for x in adv["steps"]], "expectedPlace": adv["expectedPlace"]}
        with open(self.logdir / f"{datetime.now():%Y%m%d}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    last_turn_seen = 0

    def loop(self):
        while True:
            try:
                self.step()
            except Exception as ex:                       # noqa: BLE001
                self.status = f"出错：{type(ex).__name__}: {ex}"
                traceback.print_exc()
            time.sleep(self.a.interval)

    def snapshot(self) -> dict:
        with self.lock:
            return {"status": self.status, "advice": self.advice, "simulation": self.sim, "service": "bgrl-shadow", "version": 2,
                    "model": {k: self.minfo.get(k) for k in ("name", "iter", "path")}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(DEFAULT_RUN))
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--port", type=int, default=SHADOW_PORT)
    ap.add_argument("--plugin-port", type=int, default=PLUGIN_PORT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--interval", type=float, default=0.4)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    sh = Shadow(a)

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/advice.json"):
                body = json.dumps(sh.snapshot(), ensure_ascii=False).encode("utf-8")
                ctype = "application/json; charset=utf-8"
            elif self.path.startswith("/optimize"):        # 站位优化：转给插件（结果出现在下一次的 simulation.optimizer）
                try:
                    q = self.path.split("?", 1)[1] if "?" in self.path else ""
                    with urllib.request.urlopen(f"http://localhost:{a.plugin_port}/optimize" + (f"?{q}" if q else ""),
                                                timeout=3) as r:
                        body = r.read()
                except Exception as ex:                    # noqa: BLE001
                    body = json.dumps({"accepted": False, "message": f"连不上插件：{ex}"}, ensure_ascii=False).encode("utf-8")
                ctype = "application/json; charset=utf-8"
            elif self.path in ("/", "/index.html"):
                body, ctype = PAGE.encode("utf-8"), "text/html; charset=utf-8"
            else:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    threading.Thread(target=sh.loop, daemon=True).start()
    print(f"影子模式已启动：打开 http://localhost:{a.port}/（模型建议 + Bob's Buddy 胜率 / 站位优化；"
          f"局面来自插件 http://localhost:{a.plugin_port}/state.json）。关掉这个窗口就停止。", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
