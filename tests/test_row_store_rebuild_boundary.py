"""选择性原生重编必须与纯Python复用、其他CUDA/扩展变更严格区分。"""
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'build'))
from derive_python_materials import validate_row_store_changes,validate_python_only_changes,derive,ROW_STORE_CPP


class RowStoreRebuildBoundary(unittest.TestCase):
    def test_changed_row_store_is_not_python_only_and_is_explicitly_allowed(self):
        """同一真实文件差异，默认复用拒绝，指定行存储重编才允许。"""
        with tempfile.TemporaryDirectory() as temporary:
            before,after=Path(temporary)/'before',Path(temporary)/'after'
            for root,text in ((before,'old native bytes'),(after,'new native bytes')):
                path=root/ROW_STORE_CPP;path.parent.mkdir(parents=True);path.write_text(text)
            with self.assertRaisesRegex(ValueError,'binary reuse forbidden'):
                validate_python_only_changes(before,after)
            self.assertEqual(validate_row_store_changes(before,after),{ROW_STORE_CPP})
            extra=after/'python/freetoken/kernel/csrc/jit/unrelated.cuh'
            extra.parent.mkdir(parents=True);extra.write_text('unrelated CUDA code')
            with self.assertRaisesRegex(ValueError,'source boundary differs'):
                validate_row_store_changes(before,after)

    def test_rebuild_flag_requires_an_actual_row_store_change(self):
        """不能给纯Python修订贴一个未发生的原生重编标签。"""
        with tempfile.TemporaryDirectory() as temporary:
            before,after=Path(temporary)/'before',Path(temporary)/'after'
            for root,text in ((before,'old'),(after,'new')):
                path=root/'python/freetoken/version.py';path.parent.mkdir(parents=True);path.write_text(text)
            with self.assertRaisesRegex(ValueError,'source boundary differs'):
                validate_row_store_changes(before,after)

    def test_two_native_rebuild_modes_are_not_silently_combined(self):
        """当前每次只证明一个独立重编边界，不能隐式放宽为任意原生更新。"""
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError,'one independently bounded'):
                derive('unused','unused',Path(temporary),Path(temporary)/'new',
                    rebuild_gguf=True,rebuild_row_store_module=True)


if __name__=='__main__':unittest.main()
