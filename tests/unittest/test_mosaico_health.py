"""HTTP /health route tests.

Exercise build_app() directly through an in-process ASGI transport and monkeypatch
health_check (the no-retry behavior itself was proven in 2c). Verify the route and
200/503 response shape without relying on Starlette's thread-backed TestClient.

Also exercises the REAL health_check() (no stub) to lock in Fix A: the removed
'stop'-param gate must NOT short-circuit /health for models that lack 'stop', since
PR-Agent's LiteLLMAIHandler never sends 'stop'."""
import asyncio
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import httpx
import litellm
import pytest

from pr_agent.algo.ai_handlers import litellm_helpers
from pr_agent.algo.ai_handlers.litellm_ai_handler import LiteLLMAIHandler
from pr_agent.config_loader import get_settings
from pr_agent.mosaico import executor as executor_mod
from pr_agent.mosaico import server as server_mod
from pr_agent.mosaico.executor import health_check
from pr_agent.mosaico.server import build_app
from tests.unittest._settings_helpers import restore_settings, snapshot_settings


def _app(monkeypatch, health_value):
    async def fake_health_check():
        return health_value

    # health_check is imported into server_mod's namespace and called by _HealthApp._health.
    monkeypatch.setattr(server_mod, "health_check", fake_health_check)
    return build_app()


async def _get_health(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.get("/health")


class TestHealthRoute:
    @pytest.mark.asyncio
    async def test_healthy_returns_200(self, monkeypatch):
        resp = await _get_health(_app(monkeypatch, "OK"))
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_healthy"] is True
        assert body["status"] == "OK"

    @pytest.mark.asyncio
    async def test_unhealthy_returns_503(self, monkeypatch):
        resp = await _get_health(_app(monkeypatch, "Unhealthy: connection refused"))
        assert resp.status_code == 503
        body = resp.json()
        assert body["is_healthy"] is False
        assert "Unhealthy" in body["status"]
        assert "Unhealthy" in body["detail"]


# A model id whose litellm-reported supported params genuinely LACK 'stop' (verified
# under the pinned litellm). Under the OLD (removed) gate, health_check() short-circuited
# to "Unhealthy: LLM does not support 'stop' parameter" for exactly such models — so these
# tests would have failed before Fix A. They guard against the gate being reintroduced.
_MODEL_WITHOUT_STOP = "perplexity/sonar"


@pytest.fixture
def restore_config_model():
    """Restore LLM settings exactly, including originally-absent state."""
    snapshot = snapshot_settings(
        ["CONFIG.MODEL", "LITELLM.CUSTOM_LLM_PROVIDER", "OPENAI.KEY", "GROQ.KEY", "MOSAICO.HEALTH_TIMEOUT_SECONDS"]
    )
    yield get_settings()
    restore_settings(snapshot)


class TestHealthCheckGate:
    """Exercise the REAL health_check() (not the monkeypatched stub) to lock in Fix A."""

    @pytest.mark.asyncio
    async def test_health_credentials_remain_request_local(self, monkeypatch, restore_config_model):
        restore_config_model.set("LITELLM.CUSTOM_LLM_PROVIDER", "")
        restore_config_model.set("GROQ.KEY", "groq-request-key")
        restore_config_model.set("OPENAI.KEY", "openai-request-key")
        monkeypatch.setattr(litellm, "api_key", "unrelated-global-key")
        calls = []

        async def fake_acompletion(**kwargs):
            calls.append(kwargs)
            return {"choices": [{"message": {"content": "pong"}}]}

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        for model in ("groq/gemma2-9b-it", "openai/gpt-4o"):
            restore_config_model.set("CONFIG.MODEL", model)
            assert await health_check() == "OK"

        assert [call["api_key"] for call in calls] == ["groq-request-key", "openai-request-key"]
        assert litellm.api_key == "unrelated-global-key"

    @pytest.mark.asyncio
    async def test_model_without_stop_probes_live_and_returns_ok(
        self, monkeypatch, restore_config_model
    ):
        restore_config_model.set("CONFIG.MODEL", _MODEL_WITHOUT_STOP)

        called = {}

        async def fake_acompletion(**kwargs):
            called.update(kwargs)
            return {"choices": [{"message": {"content": "pong"}}]}

        # health_check injects litellm.acompletion into the handler's probe.
        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        result = await health_check()

        # Must NOT short-circuit on the missing 'stop' param; it reaches the live probe.
        assert result == "OK"
        assert called.get("model") == _MODEL_WITHOUT_STOP

    @pytest.mark.asyncio
    async def test_live_probe_failure_returns_unhealthy(
        self, monkeypatch, restore_config_model
    ):
        restore_config_model.set("CONFIG.MODEL", _MODEL_WITHOUT_STOP)

        async def boom_acompletion(**kwargs):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(litellm, "acompletion", boom_acompletion)

        result = await health_check()
        assert result == "Unhealthy: LLM probe failed"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure_stage", ["construction", "completion"])
    async def test_failure_details_do_not_reach_http_body_or_health_log(
        self, monkeypatch, restore_config_model, failure_stage
    ):
        restore_config_model.set("CONFIG.MODEL", _MODEL_WITHOUT_STOP)
        error = RuntimeError("private-provider.example: credential=health-test-secret")
        logger = MagicMock()
        monkeypatch.setattr(executor_mod, "get_logger", lambda: logger)
        completion = AsyncMock(side_effect=error)
        monkeypatch.setattr(litellm, "acompletion", completion)
        if failure_stage == "construction":
            monkeypatch.setattr(LiteLLMAIHandler, "__init__", MagicMock(side_effect=error))

        response = await _get_health(build_app())

        assert response.status_code == 503
        assert response.json() == {
            "is_healthy": False,
            "status": "Unhealthy: LLM probe failed",
            "detail": "Unhealthy: LLM probe failed",
        }
        assert logger.mock_calls == [call.warning("MOSAICO health_check unhealthy: RuntimeError")]
        assert completion.await_count == (failure_stage == "completion")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("wait_stage", ["preparation", "completion"])
    async def test_deadline_cancels_cooperative_probe_work(
        self, monkeypatch, restore_config_model, wait_stage
    ):
        restore_config_model.set("CONFIG.MODEL", _MODEL_WITHOUT_STOP)
        deadline = asyncio.timeout(None)
        seen_timeouts = []

        def controlled_timeout(seconds):
            seen_timeouts.append(seconds)
            return deadline

        monkeypatch.setattr(executor_mod, "asyncio", SimpleNamespace(timeout=controlled_timeout))
        logger = MagicMock()
        monkeypatch.setattr(executor_mod, "get_logger", lambda: logger)
        cancelled = asyncio.Event()

        async def wait_forever(*args, **kwargs):
            # Expire only after reaching the intended stage, without a timing race.
            deadline.reschedule(asyncio.get_running_loop().time())
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        completion = AsyncMock(side_effect=wait_forever)
        monkeypatch.setattr(litellm, "acompletion", completion)
        if wait_stage == "preparation":
            monkeypatch.setattr(LiteLLMAIHandler, "_get_provider_request_params_async", wait_forever)

        result = await asyncio.wait_for(health_check(), timeout=1)

        assert seen_timeouts == [restore_config_model.get("MOSAICO.HEALTH_TIMEOUT_SECONDS")]
        assert result == "Unhealthy: LLM probe failed"
        assert cancelled.is_set()
        logger.warning.assert_called_once_with("MOSAICO health_check unhealthy: TimeoutError")
        assert completion.await_count == (wait_stage == "completion")
        if wait_stage == "completion":
            assert completion.call_args.kwargs["timeout"] == seen_timeouts[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("timeout", [10, 25, 0.5])
    async def test_configured_timeout_applies_to_preparation_and_dispatch(
        self, monkeypatch, restore_config_model, timeout
    ):
        restore_config_model.set("CONFIG.MODEL", _MODEL_WITHOUT_STOP)
        restore_config_model.set("MOSAICO.HEALTH_TIMEOUT_SECONDS", timeout)
        seen_timeouts = []

        def controlled_timeout(seconds):
            seen_timeouts.append(seconds)
            return asyncio.timeout(seconds)

        monkeypatch.setattr(executor_mod, "asyncio", SimpleNamespace(timeout=controlled_timeout))
        completion = AsyncMock()
        monkeypatch.setattr(litellm, "acompletion", completion)

        assert await health_check() == "OK"
        assert seen_timeouts == [timeout]
        completion.assert_awaited_once()
        assert completion.call_args.kwargs["timeout"] == timeout

    @pytest.mark.asyncio
    @pytest.mark.parametrize("timeout", [None, True, False, 0, -1, float("nan"), float("inf"), "secret-timeout", []])
    async def test_invalid_timeout_fails_without_dispatch(
        self, monkeypatch, restore_config_model, timeout
    ):
        restore_config_model.set("CONFIG.MODEL", _MODEL_WITHOUT_STOP)
        restore_config_model.set("MOSAICO.HEALTH_TIMEOUT_SECONDS", timeout)
        logger = MagicMock()
        monkeypatch.setattr(executor_mod, "get_logger", lambda: logger)
        completion = AsyncMock()
        monkeypatch.setattr(litellm, "acompletion", completion)

        assert await health_check() == "Unhealthy: LLM probe failed"
        completion.assert_not_awaited()
        assert logger.mock_calls == [call.warning("MOSAICO health_check unhealthy: ValueError")]

    @pytest.mark.asyncio
    async def test_caller_cancellation_propagates_without_retry_or_failure_log(
        self, monkeypatch, restore_config_model
    ):
        restore_config_model.set("CONFIG.MODEL", _MODEL_WITHOUT_STOP)
        logger = MagicMock()
        monkeypatch.setattr(executor_mod, "get_logger", lambda: logger)
        started = asyncio.Event()

        async def wait_forever(**kwargs):
            started.set()
            await asyncio.Event().wait()

        completion = AsyncMock(side_effect=wait_forever)
        monkeypatch.setattr(litellm, "acompletion", completion)
        task = asyncio.create_task(health_check())
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

        completion.assert_awaited_once()
        logger.warning.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("custom_llm_provider", "expected_model", "expected_provider"),
        [
            ("", "openrouter/openrouter/auto", ""),
            (" OpenRouter ", "openrouter/openrouter/auto", "openrouter"),
            (" OpenAI ", "openrouter/auto", "openai"),
        ],
    )
    async def test_openrouter_router_model_preserves_provider_routing(
        self, monkeypatch, restore_config_model, custom_llm_provider, expected_model, expected_provider
    ):
        restore_config_model.set("CONFIG.MODEL", "openrouter/auto")
        restore_config_model.set("LITELLM.CUSTOM_LLM_PROVIDER", custom_llm_provider)

        called = {}

        async def fake_acompletion(**kwargs):
            called.update(kwargs)
            return {"choices": [{"message": {"content": "pong"}}]}

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

        result = await health_check()

        assert result == "OK"
        assert called.get("model") == expected_model
        if expected_provider:
            assert called.get("custom_llm_provider") == expected_provider
        else:
            assert "custom_llm_provider" not in called

    @pytest.mark.asyncio
    async def test_no_model_configured_returns_unhealthy(
        self, monkeypatch, restore_config_model
    ):
        restore_config_model.set("CONFIG.MODEL", "")

        async def should_not_be_called(**kwargs):
            raise AssertionError("acompletion must not run when no model is configured")

        monkeypatch.setattr(litellm, "acompletion", should_not_be_called)

        result = await health_check()
        assert result == "Unhealthy: no model configured"


@pytest.fixture
def streaming_health_settings(restore_config_model):
    snapshot = snapshot_settings(["LITELLM.STREAM_CLOSE_TIMEOUT_SECONDS"])
    restore_config_model.set("CONFIG.MODEL", "openai/gpt-4o")
    restore_config_model.set("OPENAI.KEY", "test-key")
    restore_config_model.set("LITELLM.CUSTOM_LLM_PROVIDER", "")
    restore_config_model.set("LITELLM.STREAM_CLOSE_TIMEOUT_SECONDS", 0.01)
    try:
        yield restore_config_model
    finally:
        restore_settings(snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["required", "forced", "unexpected"])
@pytest.mark.parametrize("failure", [False, True])
async def test_health_consumes_and_closes_stream_without_text(
    monkeypatch, streaming_health_settings, mode, failure,
):
    monkeypatch.setattr(LiteLLMAIHandler, "_requires_streaming", lambda self, model: mode == "required")
    monkeypatch.setattr(
        LiteLLMAIHandler, "_force_streaming_for_request", lambda self, provider, base: mode == "forced",
    )
    logger = MagicMock()
    monkeypatch.setattr(executor_mod, "get_logger", lambda: logger)
    monkeypatch.setattr(litellm_helpers, "get_logger", lambda: logger)
    events = []
    secret = "fake-key https://private-endpoint.example"

    class Stream:
        async def __aiter__(self):
            yield {"choices": []}
            events.append("consumed")
            if failure:
                raise RuntimeError(secret)

        async def aclose(self):
            events.append("closed")
            raise ValueError(secret)

    completion = AsyncMock(return_value=Stream())
    monkeypatch.setattr(litellm, "acompletion", completion)
    response = await asyncio.wait_for(_get_health(build_app()), timeout=5)

    assert response.status_code == (503 if failure else 200)
    assert response.json()["status"] == ("Unhealthy: LLM probe failed" if failure else "OK")
    assert events == ["consumed", "closed"]
    completion.assert_awaited_once()
    assert completion.call_args.kwargs.get("stream", False) is (mode != "unexpected")
    for private_detail in ("fake-key", "private-endpoint"):
        assert private_detail not in response.text
        assert private_detail not in str(logger.mock_calls)
    if failure:
        logger.warning.assert_called_once_with("MOSAICO health_check unhealthy: RuntimeError")
    else:
        logger.warning.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["required", "forced"])
async def test_health_rejects_non_stream_response_when_streaming_requested(
    monkeypatch, streaming_health_settings, mode,
):
    monkeypatch.setattr(LiteLLMAIHandler, "_requires_streaming", lambda self, model: mode == "required")
    monkeypatch.setattr(
        LiteLLMAIHandler, "_force_streaming_for_request", lambda self, provider, base: mode == "forced",
    )
    logger = MagicMock()
    monkeypatch.setattr(executor_mod, "get_logger", lambda: logger)
    completion = AsyncMock(return_value={"choices": []})
    monkeypatch.setattr(litellm, "acompletion", completion)

    response = await asyncio.wait_for(_get_health(build_app()), timeout=5)

    assert response.status_code == 503
    assert response.json()["status"] == "Unhealthy: LLM probe failed"
    completion.assert_awaited_once()
    assert completion.call_args.kwargs["stream"] is True
    logger.warning.assert_called_once_with("MOSAICO health_check unhealthy: TypeError")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "provider", "force_provider", "base", "expected_model", "streaming"),
    [
        ("openai/qwq-plus", "", "", "", "openai/qwq-plus", True),
        ("openai/gpt-4o", "", "", "", "openai/gpt-4o", False),
        ("hosted-model", "openai", "openai", "https://gateway.example/v1", "hosted-model", True),
        ("hosted-model", "openai", "openai", "https://other.example/v1", "hosted-model", False),
        ("hosted-model", "openai", "anthropic", "https://gateway.example/v1", "hosted-model", False),
        (
            "openrouter/auto", "openrouter", "openrouter", "https://gateway.example/v1",
            "openrouter/openrouter/auto", True,
        ),
    ],
)
async def test_probe_uses_real_streaming_settings(
    model, provider, force_provider, base, expected_model, streaming,
):
    class EmptyStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    snapshot = snapshot_settings([
        "OPENAI.KEY", "OPENAI.API_BASE", "LITELLM.CUSTOM_LLM_PROVIDER",
        "LITELLM.FORCE_STREAMING_CUSTOM_LLM_PROVIDER", "LITELLM.FORCE_STREAMING_API_BASE_SUBSTRINGS",
    ])
    settings = get_settings()
    completion = AsyncMock(return_value=EmptyStream() if streaming else {"choices": []})
    try:
        settings.set("OPENAI.KEY", "test-key")
        settings.set("OPENAI.API_BASE", base)
        settings.set("LITELLM.CUSTOM_LLM_PROVIDER", provider)
        settings.set("LITELLM.FORCE_STREAMING_CUSTOM_LLM_PROVIDER", force_provider)
        settings.set("LITELLM.FORCE_STREAMING_API_BASE_SUBSTRINGS", ["gateway.example"])
        assert await LiteLLMAIHandler().probe_completion(model, _completion=completion) is None
        completion.assert_awaited_once()
        assert completion.call_args.kwargs["model"] == expected_model
        assert completion.call_args.kwargs.get("stream", False) is streaming
        expected_stream_options = {"include_usage": True} if streaming else None
        assert completion.call_args.kwargs.get("stream_options") == expected_stream_options
    finally:
        restore_settings(snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0.5, 25])
async def test_health_dispatch_consumption_and_cleanup_share_configured_deadline(
    monkeypatch, streaming_health_settings, timeout,
):
    streaming_health_settings.set("MOSAICO.HEALTH_TIMEOUT_SECONDS", timeout)
    active = []
    events = []

    @asynccontextmanager
    async def deadline(seconds):
        assert seconds == timeout
        marker = object()
        active.append(marker)
        events.append("enter")
        try:
            yield
        finally:
            assert active.pop() is marker
            events.append("exit")

    monkeypatch.setattr(executor_mod, "asyncio", SimpleNamespace(timeout=deadline))

    def record(phase):
        assert len(active) == 1
        events.append((phase, active[0]))

    class Stream:
        async def __aiter__(self):
            record("consume")
            yield {"choices": []}
            record("exhaust")

        async def aclose(self):
            record("close")

    async def dispatch(**kwargs):
        assert kwargs["timeout"] == timeout
        record("dispatch")
        return Stream()

    completion = AsyncMock(side_effect=dispatch)
    monkeypatch.setattr(litellm, "acompletion", completion)
    assert await asyncio.wait_for(health_check(), timeout=5) == "OK"
    marker = events[1][1]
    assert events == [
        "enter", ("dispatch", marker), ("consume", marker), ("exhaust", marker), ("close", marker), "exit",
    ]
    completion.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_mode", ["caller", "timeout"])
@pytest.mark.parametrize("phase", ["consumption", "cleanup"])
async def test_health_cancellation_bounds_noncooperative_cleanup_and_observes_late_error(
    monkeypatch, streaming_health_settings, cancel_mode, phase,
):
    deadline = asyncio.timeout(None)
    seen_timeouts = []

    def controlled_timeout(seconds):
        seen_timeouts.append(seconds)
        return deadline

    monkeypatch.setattr(executor_mod, "asyncio", SimpleNamespace(timeout=controlled_timeout))
    if phase == "cleanup":
        # Keep the close wait active until cancellation reaches it.
        streaming_health_settings.set("LITELLM.STREAM_CLOSE_TIMEOUT_SECONDS", 3)
    logger = MagicMock()
    monkeypatch.setattr(executor_mod, "get_logger", lambda: logger)
    monkeypatch.setattr(litellm_helpers, "get_logger", lambda: logger)
    consuming, closing, release, finished = (asyncio.Event() for _ in range(4))
    close_tasks = []
    observed = []

    class ObservedTask(asyncio.Task):
        def exception(self):
            observed.append(self)
            return super().exception()

    monkeypatch.setattr(
        litellm_helpers, "asyncio", SimpleNamespace(create_task=ObservedTask, wait=asyncio.wait),
    )

    class Stream:
        async def __aiter__(self):
            consuming.set()
            if phase == "consumption":
                await asyncio.Event().wait()
            yield {"choices": []}

        async def aclose(self):
            close_tasks.append(asyncio.current_task())
            closing.set()
            await release.wait()
            raise RuntimeError("fake-late-secret https://private-endpoint.example")

    completion = AsyncMock(return_value=Stream())
    monkeypatch.setattr(litellm, "acompletion", completion)
    task = asyncio.create_task(health_check())
    try:
        async with asyncio.timeout(5):
            await (consuming if phase == "consumption" else closing).wait()
            if cancel_mode == "caller":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                deadline.reschedule(asyncio.get_running_loop().time())
                assert await task == "Unhealthy: LLM probe failed"
            assert not release.is_set()
            assert len(close_tasks) == 1
            closer = close_tasks[0]
            assert not closer.done()
            # Register after the production observer, and do not retrieve the
            # exception ourselves before checking that the observer consumed it.
            closer.add_done_callback(lambda done: finished.set())
            release.set()
            await finished.wait()
            assert closer.done() and not closer.cancelled()
            assert observed == close_tasks
            assert isinstance(closer.exception(), RuntimeError)

        completion.assert_awaited_once()
        assert seen_timeouts == [streaming_health_settings.get("MOSAICO.HEALTH_TIMEOUT_SECONDS")]
        assert completion.call_args.kwargs["timeout"] == seen_timeouts[0]
        assert "fake-late-secret" not in str(logger.mock_calls)
        assert "private-endpoint" not in str(logger.mock_calls)
        if cancel_mode == "caller":
            logger.warning.assert_not_called()
        else:
            logger.warning.assert_called_once_with("MOSAICO health_check unhealthy: TimeoutError")
    finally:
        release.set()
        task.cancel()
        await asyncio.wait_for(asyncio.gather(task, *close_tasks, return_exceptions=True), timeout=5)
