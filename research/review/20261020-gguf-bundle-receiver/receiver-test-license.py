"""测试专用授权边界；退出作用域后恢复原生接口与验证器。"""
from contextlib import ExitStack, contextmanager
import hmac
import secrets
from unittest.mock import patch


@contextmanager
def test_license_monitor():
    """只替换原生 ABI，通过真实 LicenseMonitor 生成和校验测试 grant。"""
    import silitok_speed as bundle
    import silitok_speed.license as license_module

    token = secrets.token_bytes(32)
    with ExitStack() as stack:
        stack.enter_context(patch.object(license_module, "_native", None))
        stack.enter_context(patch.object(bundle, "_license_verify", lambda path: (token, b"test-abi"), create=True))
        stack.enter_context(patch.object(bundle, "_license_check",
            lambda value: isinstance(value, bytes) and hmac.compare_digest(value, token), create=True))
        stack.enter_context(patch.object(bundle, "_license_id", lambda: "speed-test-fixture", create=True))
        monitor = license_module.LicenseMonitor("test-fixture")
        try:
            yield monitor
        finally:
            monitor.stop()
