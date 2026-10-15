"""为本轮验收worker注入仓外数值观察器，不替换产品模块。"""
from pathlib import Path
import argparse
import runpy
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--diagnostic-output', type=Path, required=True)
parser.add_argument('--target-script', type=Path, required=True)
parser.add_argument('--hook-file', type=Path)
args, forwarded = parser.parse_known_args()
original = subprocess.Popen
hook = args.hook_file or Path(__file__).with_name('worker-numeric-hook.py')


class ObservedPopen(original):
    """保持Popen类型供Windows asyncio继承，只注入目标worker启动代码。"""
    def __init__(self, command, *positional, **keywords):
        if isinstance(command, list) and '-c' in command:
            index = command.index('-c') + 1
            marker = 'from silitok_speed.qwen36.freetoken_worker import main;'
            if marker in command[index]:
                command = list(command)
                injection = 'import runpy; runpy.run_path(' + repr(str(hook)) + \
                    ', init_globals={"OUTPUT":' + repr(str(args.diagnostic_output)) + '}); '
                command[index] = command[index].replace(marker, injection + marker)
        super().__init__(command, *positional, **keywords)


subprocess.Popen = ObservedPopen
sys.argv = [str(args.target_script), *forwarded]
runpy.run_path(str(args.target_script), run_name='__main__')
