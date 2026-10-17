"""Exercise duplicate keyed streaming against the real Engine and scheduler."""
import time
import pytest
from silitok_speed import Engine, EngineConfig, ModelSpec, TaskState
from silitok_speed.engine.types import IdempotencyConflictError
from silitok_speed.pipelines import ChatPipeline
from silitok_speed.pipelines.base import Result
from tests.support.mocks import MockLLMBackend, make_mock_hw

@pytest.mark.parametrize('initial_stream', [False, True])
@pytest.mark.parametrize('completed', [False, True])
def test_duplicate_stream_cannot_take_cleanup_ownership(tmp_path, initial_stream, completed):
    engine=Engine(EngineConfig(models_dir='',result_store_dir=str(tmp_path/'results'),
                               task_retention_seconds=60),hw=make_mock_hw())
    engine.register_pipeline('chat',ChatPipeline)
    engine.register_model(ModelSpec('m1','/fake'),MockLLMBackend)
    params=dict(model='m1',messages=[dict(role='user',content='hello')])
    events=[];retry_events=[]
    engine.start()
    try:
        # Hold the real submission lock used by scheduler dequeue, so the first
        # request is deterministically QUEUED while we exercise its duplicate.
        with engine.task_queue._lock:
            if initial_stream:
                first=engine.submit_with_stream('chat',params,events.append,idempotency_key='same')
            else:
                first=engine.submit('chat',params,idempotency_key='same')
            if not completed:
                with pytest.raises(IdempotencyConflictError,match='stream'):
                    engine.submit_with_stream('chat',params,retry_events.append,idempotency_key='same')
        deadline=time.monotonic()+10
        while engine.task_status(first) not in (TaskState.COMPLETED,TaskState.FAILED) and time.monotonic()<deadline:
            time.sleep(.01)
        assert engine.task_status(first)==TaskState.COMPLETED
        if completed:
            with pytest.raises(IdempotencyConflictError,match='stream'):
                engine.submit_with_stream('chat',params,retry_events.append,idempotency_key='same')
        assert engine.submit('chat',params,idempotency_key='same')==first
        assert not retry_events
        if initial_stream:assert any(isinstance(value,Result) for value in events)
        assert len(engine.task_queue.all_tasks())==1
    finally:engine.shutdown(timeout=2)

