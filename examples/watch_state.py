"""
订阅 BGStateExporter 推送的局面，打印简要信息。
后续接 Claude API / PPO 时，在 on_state() 里处理即可。

用法：python watch_state.py            （默认连 http://localhost:8765）
      python watch_state.py --file     （改为读取 state.json 文件）
只用到 Python 标准库。
"""
import json
import os
import sys
import time
import urllib.request

SERVER = os.environ.get("BG_SERVER", "http://localhost:8765")
STATE_FILE = os.path.join(os.environ.get("APPDATA", ""), "HearthstoneDeckTracker", "BGStateExporter", "state.json")


def on_state(s: dict) -> None:
    """每次局面变化时调用。"""
    p = s.get("player")
    if not s.get("isBattlegrounds") or not p:
        print(f"[{s.get('seq')}] {s.get('phase')}")
        return
    board = " | ".join(f"{m['name']} {m['attack']}/{m['health']}" for m in p["board"])
    line = f"[{s['seq']}] 回合{s['turn']} {s['phase']} 血{p['health']} {p['tavernTier']}本 金{p['gold']}/{p['maxGold']}  场面: {board}"
    if s["phase"] == "recruit":
        line += "\n      酒馆: " + " | ".join(f"{m['name']}{'(冻)' if m['frozen'] else ''}" for m in s["tavern"])
    elif s.get("opponent"):
        line += f"\n      对手: {s['opponent']['heroName']} " + " | ".join(f"{m['name']} {m['attack']}/{m['health']}" for m in s["opponent"]["board"])
    sim = s.get("simulation") or {}
    live = sim.get("live")
    if live and live.get("hasResult"):
        line += f"\n      Bob's Buddy 第{live['turn']}回合: 胜 {live['winRate']:.0%} 平 {live['tieRate']:.0%} 负 {live['lossRate']:.0%}"
    for r in (sim.get("matchups") or {}).get("results", []):
        tag = "（下个对手）" if r.get("isNextOpponent") else ""
        line += f"\n      对 {r['heroName']}{tag}: 胜 {r['winRate']:.0%} 负 {r['lossRate']:.0%}"
    opt = sim.get("optimizer") or {}
    if opt.get("status") == "done":
        line += f"\n      站位建议: {' > '.join(opt.get('bestOrderNames', []))}（{opt.get('message')}）"
    print(line)


def request_optimize(opponent_player_id: int = 0) -> dict:
    """请求站位优化（0 = 下个对手），结果稍后出现在 simulation.optimizer。"""
    with urllib.request.urlopen(f"{SERVER}/optimize?opponent={opponent_player_id}", timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def watch_events() -> None:
    while True:
        try:
            with urllib.request.urlopen(SERVER + "/events", timeout=60) as resp:
                print("已连接", SERVER)
                for raw in resp:
                    line = raw.decode("utf-8").rstrip("\n")
                    if line.startswith("data: "):
                        on_state(json.loads(line[6:]))
        except Exception as e:  # HDT 没开或插件没启用
            print("连接失败，3 秒后重试：", e)
            time.sleep(3)


def watch_file() -> None:
    last = None
    while True:
        try:
            with open(STATE_FILE, encoding="utf-8") as f:
                s = json.load(f)
            if s.get("seq") != last:
                last = s.get("seq")
                on_state(s)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.2)


if __name__ == "__main__":
    watch_file() if "--file" in sys.argv else watch_events()
