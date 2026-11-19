"""Hedged generation: a slow model is raced by a backup; a hung model never blocks forever."""
import threading
import time
from types import SimpleNamespace

import pytest

from app.core.config import get_settings
from app.generation import llm
from app.generation.gemini import LLMError
from app.schemas import QueryRewrite

HANG = threading.Event()


class FakeModels:
    def __init__(self, behaviour):
        self.behaviour = behaviour          # model -> "hang" | "fail" | "slow" | "ok"

    def _act(self, model):
        b = self.behaviour[model]
        if b == "hang":
            HANG.wait(30)
        if b == "fail":
            raise TimeoutError("read timed out")
        if b == "slow":
            time.sleep(0.6)
        return model

    def generate_content(self, model, contents, config):
        self._act(model)
        return SimpleNamespace(text='{"standalone_question": "%s", "search_queries": ["q"]}' % model)

    def generate_content_stream(self, model, contents, config):
        self._act(model)
        for part in (f"answer from {model} ", "[1]"):
            yield SimpleNamespace(text=part)


@pytest.fixture
def fake(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "gemini_model", "primary")
    monkeypatch.setattr(s, "gemini_fallback_models", ["backup1", "backup2"])
    monkeypatch.setattr(s, "hedge_after_s", 0.1)
    monkeypatch.setattr(s, "request_timeout_s", 0.3)

    def install(**behaviour):
        monkeypatch.setattr(llm, "get_client", lambda: SimpleNamespace(models=FakeModels(behaviour)))
    yield install
    HANG.set()
    HANG.clear()


def test_slow_primary_loses_to_backup(fake):
    fake(primary="slow", backup1="ok", backup2="ok")
    assert "".join(llm.stream_text("p", "s")) == "answer from backup1 [1]"
    assert llm.generate_structured("p", QueryRewrite, "s").standalone_question == "backup1"


def test_hung_primary_and_failing_backups_raise_instead_of_hanging(fake):
    fake(primary="hang", backup1="fail", backup2="fail")
    t = time.monotonic()
    with pytest.raises(LLMError):
        list(llm.stream_text("p", "s"))
    with pytest.raises(LLMError):
        llm.generate_structured("p", QueryRewrite, "s")
    assert time.monotonic() - t < 4
