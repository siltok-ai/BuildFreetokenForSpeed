"""Import the bundled compiled application before running its acceptance tool."""
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys

import silitok_speed
import freetoken
import freetoken.kernel._row_store as row_store
import freetoken.models.qwen4_exp.q2_source as q2_source
import freetoken.moe.fused_q2_0 as q2_kernel

application = Path(os.environ["SPEED_REVIEW_APP"]).resolve()
origin = Path(silitok_speed.__spec__.origin).resolve()
expected = application / "python/Lib/site-packages/silitok_speed/__init__.pyd"
if origin != expected or Path(sys.executable).resolve() != application / "python/python.exe":
    raise RuntimeError("acceptance is not using the bundled compiled Speed and Python")
for module in (freetoken, row_store, q2_source, q2_kernel):
    if not Path(module.__file__).resolve().is_relative_to(application):
        raise RuntimeError(f"module escaped application: {module.__file__}")
version = importlib.metadata.version("freetoken")
if version != "0.1.3+speed.q2exp1":
    raise RuntimeError(f"unexpected FreeToken version: {version}")
identity = {
    "speed_origin": str(origin), "freetoken_origin": freetoken.__file__,
    "row_store": row_store.__file__, "q2_source": q2_source.__file__,
    "q2_kernel": q2_kernel.__file__, "freetoken_version": version,
    "python": sys.version, "executable": sys.executable,
    "path": os.environ["PATH"], "build_manifest": silitok_speed._sp_manifest,
}
Path(os.environ["SPEED_REVIEW_IDENTITY_PATH"]).write_text(
    json.dumps(identity, indent=2) + "\n", encoding="utf-8")
script = Path(sys.argv[1]).resolve(strict=True)
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name="__main__")
