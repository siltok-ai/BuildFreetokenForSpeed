"""Show the duplicate response cancelling an unrelated, still-connected original stream."""
import asyncio
import time
from silitok_speed import Engine, EngineConfig, ModelSpec, TaskState
from silitok_speed.api.routes.chat import _sse_chat_response
from silitok_speed.pipelines import ChatPipeline
from tests.support.mocks import MockLLMBackend, make_mock_hw

def test_closing_retry_must_not_cancel_original_request(tmp_path):
    class StreamingBackend(MockLLMBackend):
        def generate(self,inputs,**kwargs):
            while True:
                time.sleep(.01)
                yield dict(type='token',token_id=1,delta_reasoning='',delta_text='x')
    engine=Engine(EngineConfig(models_dir='',result_store_dir=str(tmp_path/'results'),
                               task_retention_seconds=60),hw=make_mock_hw())
    engine.register_pipeline('chat',ChatPipeline)
    engine.register_model(ModelSpec('m1','/fake'),StreamingBackend)
    async def scenario():
        engine.start()
        params=dict(model='m1',messages=[dict(role='user',content='block')])
        async def submit(sink):
            return engine.submit_with_stream('chat',params,sink,idempotency_key='retry'),60
        original=None
        try:
            original=await _sse_chat_response(engine,'m1',submit)
            await anext(original.body_iterator)
            first=engine.find_task_by_idempotency_key('retry')
            deadline=time.monotonic()+10
            while engine.task_status(first.id)!=TaskState.RUNNING and time.monotonic()<deadline:
                await asyncio.sleep(.01)
            assert engine.task_status(first.id)==TaskState.RUNNING
            retry=await _sse_chat_response(engine,'m1',submit)
            await anext(retry.body_iterator)
            await retry.body_iterator.aclose()
            deadline=time.monotonic()+2
            while engine.task_status(first.id) in (TaskState.RUNNING,TaskState.CANCELLING) and time.monotonic()<deadline:
                await asyncio.sleep(.01)
            assert engine.task_status(first.id)==TaskState.RUNNING
        finally:
            if original is not None:await original.body_iterator.aclose()
            engine.shutdown(timeout=2)
    asyncio.run(scenario())
