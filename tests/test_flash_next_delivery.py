"""真实文件的损坏、搬迁、默认版本及发布边界，不加载 GPU 或模拟大权重。"""
from pathlib import Path
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from current_release import load_current
from flash_next_weights import layout, plan, prepare, verify_checkpoint, verify_file, write, read, download
from flash_next_mtp import canonical_manifest, finalize_runtime
from flash_next_speed_config import configuration, link_checkpoints


def fixture(root):
    lock = dict(model='test-full-gguf', scope='test',
        gguf=dict(repo='test/gguf', revision='fixed', files=[]),
        metadata=dict(repo='test/tokenizer', revision='fixed', files=[]),
        mtp=dict(revision='fixed'),
        expert_geometry=dict(layers=48, experts_per_layer=512, hidden=2560, intermediate=640, expert_bytes=33973862400))
    paths = layout(root, lock)
    for name in ('gguf','metadata'):
        paths[name].mkdir(parents=True)
    for i in range(2):
        path = paths['gguf']/f'part-{i}.gguf'
        data = b'GGUF\x03\x00\x00\x00'+bytes([i])*64
        path.write_bytes(data)
        lock['gguf']['files'].append(dict(name=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    for name, data in [('config.json', b'{"model_type":"qwen4_exp","quantization_config":{}}'),
                       ('tokenizer.json', b'{"test":"tokenizer"}'), ('generation_config.json', b'{}')]:
        (paths['metadata']/name).write_bytes(data)
        lock['metadata']['files'].append(dict(name=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    return lock, paths


class DeliveryBoundaries(unittest.TestCase):
    def test_default_points_to_current_speed(self):
        from prepare_source import DEFAULT_PROFILE, load_profile
        current = load_current()
        self.assertEqual(DEFAULT_PROFILE, ROOT/current['profile'])
        self.assertEqual(load_profile(DEFAULT_PROFILE)[0]['custom_version'], current['version'])

    def test_plan_and_config_dry_run_create_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'absent'
            for script, arguments in [('flash_next_weights.py', ['plan']),
                ('flash_next_speed_config.py', ['--dry-run'])]:
                subprocess.run([sys.executable,'-B',str(ROOT/'tools'/script),*arguments,'--root',str(target)],
                               check=True,capture_output=True)
                self.assertFalse(target.exists())

    def test_complete_checkpoint_records_portable_sources_and_preserves_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock, paths = fixture(root)
            old = {str(p):p.read_bytes() for key in ('gguf','metadata') for p in paths[key].iterdir()}
            prepare(root, lock)
            self.assertEqual(verify_checkpoint(root,lock)['status'], 'verified')
            config = read(paths['checkpoint']/'config.json')
            self.assertEqual(config['freetoken_gguf_source']['format'], 'gguf_strata')
            marker=read(paths['checkpoint']/'strata-q2-preparation.json')
            self.assertTrue(all(not Path(p).is_absolute() for p in marker['sources'].values()))
            self.assertTrue(all(Path(r['path']).is_absolute() for r in config['freetoken_gguf_source']['shards']))
            self.assertNotIn('quantization_config', config)
            self.assertTrue(all(Path(name).read_bytes() == value for name,value in old.items()))

    def test_same_length_corruption_cannot_be_accepted_by_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            lock, paths = fixture(directory)
            prepare(directory,lock)
            config = (paths['checkpoint']/'config.json').read_bytes()
            file = next(paths['gguf'].glob('*.gguf'))
            data = file.read_bytes();file.write_bytes(data[:-1]+b'x')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                prepare(directory,lock,refresh=True)
            self.assertEqual((paths['checkpoint']/'config.json').read_bytes(),config)

    def test_relocated_tree_refreshes_only_after_full_hash_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory)/'before', Path(directory)/'after'
            lock, paths = fixture(before)
            prepare(before,lock)
            shutil.copytree(before,after)
            relocated = layout(after,lock)
            for file in relocated['gguf'].glob('*.gguf'):
                info=file.stat();os.utime(file,ns=(info.st_atime_ns,info.st_mtime_ns+10_000_000))
            with self.assertRaisesRegex(ValueError,'timestamps'):
                verify_checkpoint(after,lock)
            prepare(after,lock,refresh=True)
            self.assertEqual(verify_checkpoint(after,lock)['status'],'verified')

    def test_unowned_and_modified_configuration_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            lock, paths = fixture(directory)
            prepare(directory,lock)
            with self.assertRaises(FileExistsError):
                prepare(directory,lock)
            config = read(paths['checkpoint']/'config.json');config['hidden_size']=999
            write(paths['checkpoint']/'config.json',config)
            with self.assertRaisesRegex(ValueError,'configuration changed'):
                prepare(directory,lock,refresh=True)
            with self.assertRaisesRegex(ValueError,'certified metadata'):
                verify_checkpoint(directory,lock)

    def test_metadata_only_reuses_verified_files_without_requiring_gguf(self):
        with tempfile.TemporaryDirectory() as directory:
            lock, paths = fixture(directory)
            shutil.rmtree(paths['gguf'])
            self.assertEqual(download(directory,lock,metadata_only=True)['status'], 'downloaded_and_verified')
            self.assertFalse(paths['gguf'].exists())

    def test_refresh_preserves_receipts_when_original_metadata_is_corrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            lock, paths = fixture(directory)
            prepare(directory,lock)
            config=(paths['checkpoint']/'config.json').read_bytes()
            original=paths['checkpoint']/'hf-config.original.json'
            data=original.read_bytes();original.write_bytes(data[:-1]+b'x')
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                prepare(directory,lock,refresh=True)
            self.assertEqual((paths['checkpoint']/'config.json').read_bytes(),config)

    def test_mtp_canonical_identity_is_the_existing_speed_release(self):
        data, manifest = canonical_manifest()
        self.assertEqual(hashlib.sha256(data).hexdigest(),'610c51dadbfa1b38775a88da708270c0084c84214d934d9192542267f68dc81d')
        self.assertEqual(len(manifest['tensors']),31)
        self.assertEqual(manifest['total_bytes'],823887872)

    def test_mtp_failed_conversion_never_installs_a_runtime_manifest(self):
        _, manifest = canonical_manifest()
        report = copy.deepcopy(manifest)
        report['tensors'][0]['sha256']='0'*64
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'approved recipe'):
                finalize_runtime(directory,report)
            self.assertFalse((Path(directory)/'runtime.json').exists())

    def test_same_budget_on_off_and_best_off_are_explicit(self):
        preset=read(ROOT/'integration/speed/presets/flash-next-sm120-16g-portable1.json')
        on, config=configuration('on',preset)
        off, _=configuration('off-matched',preset)
        best, _=configuration('off',preset)
        a=on['models'][0];b=off['models'][0];c=best['models'][0]
        self.assertEqual(a['engine']['extra']['moe_cache_size'],b['engine']['extra']['moe_cache_size'])
        self.assertEqual(c['engine']['extra']['moe_cache_size'],7808)
        self.assertIn('mtp',a['artifacts']);self.assertNotIn('mtp',b['artifacts'])
        self.assertEqual(config['resource']['gpu_probe'],'nvml')
        self.assertFalse(b['engine']['extra']['mtp_enabled'])

    def test_current_cli_generates_auto_tiers_without_fixed_capacity_overrides(self):
        """真实CLI输出供Speed分档，OFF/ON保持共用权重与MTP开关合同。"""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'absent'
            entries=[]
            for mode in ('off','on'):
                result=subprocess.run([sys.executable,'-B',str(ROOT/'tools/flash_next_speed_config.py'),
                    '--root',str(root),'--mode',mode,'--dry-run'],check=True,capture_output=True,text=True)
                data=json.loads(result.stdout)
                entry=data['manifest']['models'][0]
                self.assertEqual(data['freetoken_version'],load_current()['version'])
                self.assertEqual(entry['engine']['resource_profile'],'auto')
                self.assertEqual(entry['engine']['extra']['mtp_enabled'],mode=='on')
                self.assertEqual('mtp' in entry['artifacts'],mode=='on')
                self.assertTrue({'moe_cache_size','moe_q2_resident_experts','max_prefill_length',
                    'vram_estimate_bytes','host_commit_estimate_bytes'}.isdisjoint(entry['engine']['extra']))
                entries.append(entry)
            self.assertEqual(entries[0]['artifacts']['weights'],entries[1]['artifacts']['weights'])
            self.assertFalse(root.exists())



    def test_shared_checkpoint_entry_reads_original_bytes_and_refuses_replacement(self):
        """实际目录共享必须读取源文件，并拒绝覆盖已有入口。"""
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source';source.mkdir()
            (source/'weight.bin').write_bytes(b'original-weight-bytes')
            link=root/'checkpoint-link'
            link_checkpoints(source,link)
            self.assertTrue(os.path.samefile(source/'weight.bin',link/'weight.bin'))
            self.assertEqual((link/'weight.bin').read_bytes(),(source/'weight.bin').read_bytes())
            with self.assertRaises(OSError):
                link_checkpoints(source,link)
            self.assertEqual((source/'weight.bin').read_bytes(),b'original-weight-bytes')



if __name__=='__main__':
    unittest.main()
