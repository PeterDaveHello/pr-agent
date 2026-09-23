"""Raise a real openai.APIError when a streaming response arrives empty."""
import asyncio
import copy
from contextvars import ContextVar
from types import SimpleNamespace

import openai
import pytest
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from starlette_context import request_cycle_context

from pr_agent.algo.ai_handlers import litellm_helpers
from pr_agent.algo.ai_handlers.litellm_helpers import _handle_streaming_response
from pr_agent.config_loader import get_settings, global_settings
from pr_agent.git_providers import utils as git_utils


class Chunk:
    def __init__(self, content, finish_reason):
        delta = type("Delta", (), {"content": content})()
        self.choices = [type("Choice", (), {"delta": delta, "finish_reason": finish_reason})()]
        self.usage = None
        self._hidden_params = {}


class Stream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        async def generate():
            for chunk in self.chunks:
                yield chunk
        return generate()


def collect(chunks):
    return asyncio.run(_handle_streaming_response(Stream(chunks), model="some-model"))


def test_collect_a_normal_streaming_response():
    """Keep assembling a streamed answer exactly as before."""
    content, finish_reason, _ = collect([Chunk("hel", None), Chunk("lo", None),
                                         Chunk(None, "stop")])

    assert content == "hello"
    assert finish_reason == "stop"


@pytest.mark.parametrize("chunks, reason", [
    ([Chunk(None, "stop")], "completed with a finish reason but no content"),
    ([Chunk(None, None)], "ended without content or a finish reason"),
])
def test_raise_an_api_error_the_retry_can_catch(chunks, reason):
    """openai.APIError is what @retry(retry_if_exception_type(openai.APIError)) waits for."""
    with pytest.raises(openai.APIError):
        collect(chunks)


def test_the_raised_error_carries_its_message():
    """Keep the diagnostic message that names the finish reason."""
    with pytest.raises(openai.APIError) as excinfo:
        collect([Chunk(None, "content_filter")])

    assert "content_filter" in str(excinfo.value)


@pytest.mark.parametrize("outcome", ["success", "empty", "failure", "cancel"])
async def test_stream_is_closed_on_every_collection_exit(outcome):
    closed = []
    failure = RuntimeError("stream failed")

    class ClosingStream(Stream):
        def __aiter__(self):
            async def generate():
                if outcome == "failure":
                    raise failure
                if outcome == "cancel":
                    raise asyncio.CancelledError
                if outcome == "success":
                    yield Chunk("ping", "stop")
            return generate()

        async def aclose(self):
            closed.append(True)
            raise ValueError("cleanup must not replace the result")

    stream = ClosingStream([])
    if outcome == "success":
        assert (await _handle_streaming_response(stream))[0] == "ping"
    else:
        expected = {"empty": openai.APIError, "failure": RuntimeError, "cancel": asyncio.CancelledError}[outcome]
        with pytest.raises(expected) as caught:
            await _handle_streaming_response(stream)
        if outcome == "failure":
            assert caught.value is failure
    assert closed == [True]


async def test_stream_close_restores_correlation_context_in_the_consuming_task():
    correlation_id = ContextVar("correlation_id", default="outer")
    correlation_id.set("stream")
    consuming_task = asyncio.current_task()
    restored_in = []

    class CorrelatedStream:
        def _restore_consumer_correlation_context(self):
            restored_in.append(asyncio.current_task())
            correlation_id.set("outer")

        async def aclose(self):
            self._restore_consumer_correlation_context()

    await litellm_helpers._close_stream(CorrelatedStream())

    assert len(restored_in) == 2
    assert restored_in[0] is not consuming_task
    assert restored_in[1] is consuming_task
    assert correlation_id.get() == "outer"


async def test_real_litellm_stream_close_contract():
    consuming_task = asyncio.current_task()
    events = []

    class UnderlyingStream:
        async def aclose(self):
            events.append(("close", asyncio.current_task()))

    class Logging:
        def _restore_correlation_context(self):
            events.append(("restore", asyncio.current_task()))

        def _restore_correlation_context_if_unclaimed(self):
            pass

    class RealLiteLLMStream(CustomStreamWrapper):
        def __aiter__(self):
            async def generate():
                yield Chunk("ping", "stop")

            return generate()

    stream = object.__new__(RealLiteLLMStream)
    stream.completion_stream = UnderlyingStream()
    stream.logging_obj = Logging()

    assert (await _handle_streaming_response(stream))[0] == "ping"
    assert stream.completion_stream is None
    assert events[0][0] == "close"
    assert events[0][1] is not consuming_task
    assert events[1] == ("restore", events[0][1])
    assert events[2] == ("restore", consuming_task)


def test_litellm_stream_exposes_consumer_correlation_restore_hook():
    assert callable(getattr(CustomStreamWrapper, "_restore_consumer_correlation_context", None))


async def test_stream_close_is_bounded_and_late_failure_observed(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()
    original_wait = asyncio.wait
    tasks = []
    observed = []

    class ObservedTask(asyncio.Task):
        def exception(self):
            observed.append(self)
            return super().exception()

    def create_task(coroutine):
        task = ObservedTask(coroutine)
        tasks.append(task)
        return task

    class HangingStream:
        async def aclose(self):
            started.set()
            await release.wait()
            raise ValueError("late provider error")

    async def expired_wait(pending, *, timeout):
        assert timeout == litellm_helpers.STREAM_CLOSE_TIMEOUT_SECONDS
        await started.wait()
        return await original_wait(pending, timeout=0)

    monkeypatch.setattr(litellm_helpers, "asyncio", SimpleNamespace(create_task=create_task, wait=expired_wait))
    try:
        async with asyncio.timeout(5):
            await litellm_helpers._close_stream(HangingStream())
            assert tasks[0] in litellm_helpers._stream_close_tasks
            assert not tasks[0].done()
            tasks[0].add_done_callback(lambda _: completed.set())
            release.set()
            await completed.wait()
            assert tasks[0] not in litellm_helpers._stream_close_tasks
            # Check the production callback, before the test retrieves the exception.
            assert observed == tasks
    finally:
        release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_repeated_cancellation_during_close_propagates_without_cancelling_cleanup():
    consuming = asyncio.Event()
    closing = asyncio.Event()
    closed = asyncio.Event()
    release = asyncio.Event()

    class CancelledStream:
        def __aiter__(self):
            async def generate():
                consuming.set()
                await release.wait()
                yield Chunk("unreachable", "stop")
            return generate()

        async def aclose(self):
            closing.set()
            try:
                await release.wait()
            finally:
                closed.set()

    task = asyncio.create_task(_handle_streaming_response(CancelledStream()))
    try:
        async with asyncio.timeout(5):
            await consuming.wait()
            task.cancel()
            await closing.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not closed.is_set()
            release.set()
            await closed.wait()
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_real_litellm_close_continues_after_bounded_wait(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()
    cancelled = asyncio.Event()
    previous_tasks = set(litellm_helpers._stream_close_tasks)

    class UnderlyingStream:
        async def aclose(self):
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            finally:
                closed.set()

    class Logging:
        def _restore_correlation_context(self):
            pass

        def _restore_correlation_context_if_unclaimed(self):
            pass

    stream = object.__new__(CustomStreamWrapper)
    stream.completion_stream = UnderlyingStream()
    stream.logging_obj = Logging()
    monkeypatch.setattr(
        litellm_helpers,
        "get_settings",
        lambda: SimpleNamespace(get=lambda *args: 0.01),
    )

    try:
        await litellm_helpers._close_stream(stream)
        close_tasks = litellm_helpers._stream_close_tasks - previous_tasks
        assert len(close_tasks) == 1
        close_task = close_tasks.pop()
        assert started.is_set()
        assert not close_task.done()
        assert not cancelled.is_set()

        release.set()
        await asyncio.wait_for(closed.wait(), timeout=5)
        await asyncio.wait_for(asyncio.shield(close_task), timeout=5)
        await asyncio.sleep(0)
        assert not cancelled.is_set()
        assert close_task not in litellm_helpers._stream_close_tasks
    finally:
        release.set()
        await asyncio.gather(
            *(litellm_helpers._stream_close_tasks - previous_tasks),
            return_exceptions=True,
        )


@pytest.mark.parametrize("closer", [None, "not-callable", lambda: None])
async def test_optional_or_synchronous_closer_preserves_success(closer):
    stream = Stream([Chunk("ping", "stop")])
    stream.aclose = closer
    assert (await _handle_streaming_response(stream))[0] == "ping"


async def test_stream_close_scheduling_failure_preserves_success(monkeypatch):
    warnings = []
    restored = []

    class SchedulingFailure:
        @staticmethod
        def create_task(_coroutine):
            raise RuntimeError("sensitive-scheduler-detail")

    class RestoringStream(Stream):
        def _restore_consumer_correlation_context(self):
            restored.append(asyncio.current_task())

        async def aclose(self):
            raise AssertionError("unscheduled cleanup must not run")

    monkeypatch.setattr(litellm_helpers, "asyncio", SchedulingFailure)
    monkeypatch.setattr(litellm_helpers, "get_logger", lambda: SimpleNamespace(warning=warnings.append))

    assert (await _handle_streaming_response(RestoringStream([Chunk("ping", "stop")])))[0] == "ping"
    assert restored == [asyncio.current_task()]
    assert warnings == ["Unable to schedule stream cleanup"]


@pytest.mark.parametrize("failure_site", ["settings", "scheduling"])
async def test_cleanup_warning_failure_preserves_success(monkeypatch, failure_site):
    def fail(*args):
        raise RuntimeError("sensitive-cleanup-detail")

    if failure_site == "settings":
        monkeypatch.setattr(litellm_helpers, "get_settings", fail)
    else:
        monkeypatch.setattr(litellm_helpers.asyncio, "create_task", fail)
    monkeypatch.setattr(litellm_helpers, "get_logger", lambda: SimpleNamespace(warning=fail))

    assert (await _handle_streaming_response(Stream([Chunk("ping", "stop")])))[0] == "ping"


@pytest.mark.parametrize("configured", [1, 3, 0.25, None, False, 0, -1, "3", float("nan"), float("inf"), 10 ** 400])
async def test_cleanup_budget_is_configurable_without_losing_cleanup(monkeypatch, configured):
    waits = []
    closed = []
    original_wait = asyncio.wait

    async def wait(tasks, *, timeout):
        waits.append(timeout)
        return await original_wait(tasks, timeout=timeout)

    monkeypatch.setattr(litellm_helpers, "get_settings", lambda: SimpleNamespace(get=lambda *args: configured))
    monkeypatch.setattr(litellm_helpers, "asyncio", SimpleNamespace(create_task=asyncio.create_task, wait=wait))
    stream = Stream([Chunk("ping", "stop")])

    async def close():
        closed.append(True)

    stream.aclose = close
    assert (await _handle_streaming_response(stream))[0] == "ping"
    assert closed == [True]
    assert waits == [configured if configured in (1, 3, 0.25) else 1]


@pytest.mark.parametrize("failure_site", ["settings", "lookup"])
async def test_cleanup_survives_settings_errors_without_logging_details(monkeypatch, failure_site):
    warnings = []
    closed = []

    def fail(*args):
        raise RuntimeError("sensitive-provider-setting")

    settings = fail if failure_site == "settings" else lambda: SimpleNamespace(get=fail)
    monkeypatch.setattr(litellm_helpers, "get_settings", settings)
    monkeypatch.setattr(litellm_helpers, "get_logger", lambda: SimpleNamespace(warning=warnings.append))
    stream = Stream([Chunk("ping", "stop")])

    async def close():
        closed.append(True)

    stream.aclose = close
    assert (await _handle_streaming_response(stream))[0] == "ping"
    assert closed == [True]
    assert warnings == ["Invalid stream cleanup timeout; using the one-second safety fallback"]


@pytest.mark.parametrize("configured", ["5", "0.25"])
@pytest.mark.parametrize("auto_cast", ["true", "false"])
async def test_cleanup_uses_environment_budget_after_repo_settings(monkeypatch, configured, auto_cast):
    monkeypatch.setenv("AUTO_CAST_FOR_DYNACONF", auto_cast)
    settings = copy.deepcopy(global_settings)
    settings.set("CONFIG.EXTRA_CONFIG_URL", None)
    settings.set("CONFIG.USE_REPO_SETTINGS_FILE", True)
    monkeypatch.setenv("LITELLM__STREAM_CLOSE_TIMEOUT_SECONDS", configured)
    provider = SimpleNamespace(get_repo_settings=lambda: b"[litellm]\nstream_close_timeout_seconds = 2\n")
    monkeypatch.setattr(git_utils, "get_git_provider_with_context", lambda url: provider)
    waits = []
    original_wait = asyncio.wait

    async def wait(tasks, *, timeout):
        waits.append(timeout)
        return await original_wait(tasks, timeout=timeout)

    monkeypatch.setattr(litellm_helpers, "asyncio", SimpleNamespace(create_task=asyncio.create_task, wait=wait))
    with request_cycle_context({"settings": settings}):
        git_utils.apply_repo_settings("https://git.example/project/pull/1")
        assert get_settings().get("LITELLM.STREAM_CLOSE_TIMEOUT_SECONDS") == float(configured)
        assert (await _handle_streaming_response(Stream([Chunk("ping", "stop")])))[0] == "ping"
    assert waits == [float(configured)]
