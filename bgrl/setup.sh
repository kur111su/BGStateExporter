#!/bin/bash
# 一次性安装（Mac / Linux；可以重复运行）：Python 环境、TavernGym 模拟器、PyTorch，然后跑测试、一次小训练、测设备速度。
#   ./setup.sh          安装 / 更新并检查（Mac 上最后测一下 MPS / CPU 哪个快，约 5–15 分钟）
#   ./setup.sh quick    不测设备速度
# 全部装在 bgrl/.venv 和 bgrl/wheels；记录在 bgrl/reports/setup.txt（出问题把它发给 Claude）。
set -u
cd "$(dirname "$0")"
mkdir -p reports wheels
LOG=reports/setup.txt
echo "setup $(date)" > "$LOG"
say() { echo "$@"; echo "$@" >> "$LOG"; }
fail() { say "失败：$1（详细见 bgrl/$LOG）"; exit 1; }

# --- 1. Python 3.10–3.13 ---
ok_py() { "$1" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' >/dev/null 2>&1; }
PY=""
if [ ! -x .venv/bin/python ]; then
  for c in python3.12 python3.13 python3.11 python3.10 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.13 \
           /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 python3; do
    if command -v "$c" >/dev/null 2>&1 && ok_py "$c"; then PY="$c"; break; fi
  done
  if [ -z "$PY" ]; then
    say "没找到 Python 3.10–3.13，用 uv 装一个 Python 3.12（只装在你的用户目录里）……"
    if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
      curl -LsSf https://astral.sh/uv/install.sh | sh >> "$LOG" 2>&1 || fail "装 uv"
    fi
    UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
    "$UV" python install 3.12 >> "$LOG" 2>&1 || fail "装 Python 3.12"
    PY="$("$UV" python find 3.12)"
  fi
  say "建 .venv（$PY）……"
  "$PY" -m venv .venv >> "$LOG" 2>&1 || fail "建 .venv"
fi
VPY=.venv/bin/python
$VPY -m pip install -q --upgrade pip >> "$LOG" 2>&1
$VPY -m pip install -q -r requirements.txt >> "$LOG" 2>&1 || {
  say "pip 换清华镜像再试……"
  $VPY -m pip install -q -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple >> "$LOG" 2>&1 || fail "pip install"
}

# --- 2. TavernGym（我们的 GitHub Actions 从 MCG-NJU/AutoTavern 编译，MIT；版本锁在 taverngym/PIN.txt）---
COMMIT=$(grep '^commit=' taverngym/PIN.txt | cut -d= -f2)
TAG="taverngym-${COMMIT:0:7}"
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) PAT="macosx.*arm64" ;;
  Darwin-x86_64) PAT="macosx.*x86_64" ;;
  Linux-x86_64) PAT="linux_x86_64|manylinux.*x86_64" ;;
  Linux-aarch64) PAT="linux_aarch64|manylinux.*aarch64" ;;
  *) fail "不支持的系统 $(uname -s)-$(uname -m)" ;;
esac
STAMP=wheels/.installed
if ! $VPY -c "import taverngym" >/dev/null 2>&1 || [ "$(cat "$STAMP" 2>/dev/null)" != "$TAG" ]; then
  say "下载 TavernGym $TAG ……"
  WHL=$(curl -fsSL "https://api.github.com/repos/kur111su/BGStateExporter/releases/tags/$TAG" |
        $VPY -c "import json,re,sys; a=[x['name'] for x in json.load(sys.stdin)['assets'] if x['name'].endswith('.whl') and re.search(r'$PAT', x['name'])]; print(a[0] if a else '')")
  [ -n "$WHL" ] || fail "$TAG 里没有这台电脑能用的安装包（$PAT）——要等 GitHub 上的 TavernGym 工作流编译好"
  curl -fsSL -o "wheels/$WHL" "https://github.com/kur111su/BGStateExporter/releases/download/$TAG/$WHL" || fail "下载 $WHL"
  $VPY -m pip install -q --force-reinstall --no-deps "wheels/$WHL" >> "$LOG" 2>&1 || fail "装 TavernGym"
  echo "$TAG" > "$STAMP"
fi

# --- 3. PyTorch（Mac：PyPI 的版本自带 MPS；Linux 有 NVIDIA 显卡装 CUDA 版）---
if ! $VPY -c "import torch" >/dev/null 2>&1; then
  say "装 PyTorch（几百 MB）……"
  if [ "$(uname -s)" = "Linux" ] && command -v nvidia-smi >/dev/null 2>&1; then
    $VPY -m pip install torch --index-url https://download.pytorch.org/whl/cu124 >> "$LOG" 2>&1 || fail "装 PyTorch"
  else
    $VPY -m pip install torch >> "$LOG" 2>&1 || fail "装 PyTorch"
  fi
fi

# --- 4. 检查 ---
$VPY -X utf8 bg.py info 2>&1 | tee -a "$LOG"
echo; say "=== 测试 ==="
$VPY -X utf8 bg.py test > reports/setup_step.txt 2>&1; RC=$?
tail -n 20 reports/setup_step.txt; cat reports/setup_step.txt >> "$LOG"
[ $RC -eq 0 ] || fail "测试"
echo; say "=== 小训练 ==="
$VPY -X utf8 bg.py train --smoke > reports/setup_step.txt 2>&1; RC=$?
tail -n 8 reports/setup_step.txt; cat reports/setup_step.txt >> "$LOG"
[ $RC -eq 0 ] || fail "小训练"
if [ "${1:-}" != "quick" ]; then
  echo; say "=== 测设备速度（约 5–15 分钟）==="
  $VPY -X utf8 bg.py bench-device 2>&1 | tee -a "$LOG"
fi
chmod +x panel.command 2>/dev/null
echo; say "装好了。以后双击 bgrl/panel.command 打开面板。"
