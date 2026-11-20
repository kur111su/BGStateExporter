"""课程里改局面用的 JSON 读写（BatchEnv.save / load 的存档）：有 orjson 就用（快 5–10 倍），没有就用标准库。"""
from __future__ import annotations

import json

try:
    import orjson

    def _loads(s):
        return orjson.loads(s)

    def _dumps(o):
        return orjson.dumps(o).decode("utf-8")
except ImportError:                    # pragma: no cover
    _loads, _dumps = json.loads, lambda o: json.dumps(o, separators=(",", ":"))
