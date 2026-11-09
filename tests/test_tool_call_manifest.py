"""工具部署显式开关不改变 Flash 权重、MTP 和资源合同。"""

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from flash_next_speed_config import configuration


class ToolManifestTests(unittest.TestCase):
    def test_flash_tools_preserves_every_mode_and_resource_option(self):
        preset = json.loads(
            (
                ROOT / "integration/speed/presets/flash-next-sm120-auto-cloverq8.json"
            ).read_text()
        )
        original = copy.deepcopy(preset)
        for mode in preset["modes"]:
            with self.subTest(mode=mode):
                plain, plain_resource = configuration(mode, preset)
                manifest, resource = configuration(mode, preset, enable_tools=True)
                entry = manifest["models"][0]
                self.assertEqual(entry.pop("tool_call_format"), "qwen3_coder")
                self.assertEqual(entry["features"].pop(), "tools")
                self.assertEqual(manifest, plain)
                self.assertEqual(resource, plain_resource)
        self.assertEqual(preset, original)
