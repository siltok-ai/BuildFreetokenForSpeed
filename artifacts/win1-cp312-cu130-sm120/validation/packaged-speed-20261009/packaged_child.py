"""Run a reviewed smoke tool after importing the actual bundled Speed package."""
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys

import silitok_speed
import freetoken
import freetoken.kernel._row_store as row_store

application = Path(os.environ["SPEED_REVIEW_APP"]).resolve()
origin = Path(silitok_speed.__spec__.origin).resolve()
expected = application / "python/Lib/site-packages/silitok_speed/__init__.pyd"
if origin != expected:
    raise RuntimeError(f"unexpected Speed runtime: {origin}")
if not Path(freetoken.__file__).resolve().is_relative_to(application):
    raise RuntimeError("FreeToken is not owned by the packaged application")
if importlib.metadata.version("freetoken") != "0.1.3+speed.win1":
    raise RuntimeError("packaged FreeToken version mismatch")
identity = {"speed_origin": str(origin), "freetoken_origin": freetoken.__file__,
            "row_store": row_store.__file__, "python": sys.version, "executable": sys.executable,
            "path": os.environ["PATH"], "build_manifest": silitok_speed._sp_manifest}
Path(os.environ["SPEED_REVIEW_IDENTITY_PATH"]).write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
script = Path(sys.argv[1]).resolve()
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name="__main__")
