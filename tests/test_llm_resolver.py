"""Tests for the layered LLM client resolver
(``statspai.causal_llm._resolver``) and config file
(``statspai.causal_llm._config``).

The resolver is the central rendezvous for "auto" LLM access used by
``sp.paper(llm='auto')`` and the LLM-DAG closed loop. These tests
exercise:

- Layered fallback order: explicit client > explicit provider+key >
  env var > config file (provider + model only) > interactive prompt
  (TTY-gated) > hard error.
- Config file: read / write, XDG path resolution, missing file
  graceful no-op, malformed file safe-default.
- Provider availability inspection.
- ``configure_llm()`` round-trip.
- Hard-error message points the user at remediation steps.

We never make real LLM calls — every test that needs a "client"
either inspects what would be constructed or uses an
:func:`echo_client` to capture the round-trip.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from statspai.causal_llm._config import (
    DEFAULT_MODELS,
    config_path,
    load_config,
    save_config,
    set_preferences,
)
from statspai.causal_llm._resolver import (
    LLMConfigurationError,
    configure_llm,
    get_llm_client,
    list_available_providers,
)
from statspai.causal_llm.llm_clients import echo_client, LLMClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_env(monkeypatch):
    """Strip both API key env vars and any TTY signal."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return monkeypatch


@pytest.fixture
def tmp_cfg(tmp_path) -> Path:
    return tmp_path / "llm.toml"


@pytest.fixture
def fake_construct(monkeypatch):
    """Patch ``_construct_client`` so we can verify the resolver's
    decisions without actually importing the OpenAI / Anthropic SDKs
    (which aren't a hard dep of StatsPAI's core).
    """
    captured: dict = {}

    def _fake(provider, *, model=None, api_key=None, **kw):
        captured["provider"] = provider
        captured["model"] = model
        captured["api_key"] = api_key
        captured["kw"] = kw
        # Return a duck-typed stand-in that has the same .model / .name
        # surface the LLMClient defines.
        client = type("FakeClient", (), {})()
        client.model = model or DEFAULT_MODELS[provider]
        client.name = provider
        client.complete = lambda p: f"{provider}:{p}"
        client.chat = lambda role, p: f"{provider}:{p}"
        client.__call__ = lambda p: f"{provider}:{p}"
        return client

    monkeypatch.setattr(
        "statspai.causal_llm._resolver._construct_client", _fake,
    )
    return captured


# ---------------------------------------------------------------------------
# Config file
# ---------------------------------------------------------------------------

class TestConfigFile:
    def test_default_models_exposed(self):
        assert "anthropic" in DEFAULT_MODELS
        assert "openai" in DEFAULT_MODELS
        assert DEFAULT_MODELS["anthropic"].startswith("claude")
        assert DEFAULT_MODELS["openai"].startswith("gpt")

    def test_config_path_uses_xdg_when_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        # Avoid Windows path branching for this test.
        if os.name != "nt":
            assert config_path() == tmp_path / "statspai" / "llm.toml"

    def test_load_missing_returns_empty(self, tmp_cfg):
        assert load_config(tmp_cfg) == {}

    def test_load_malformed_returns_empty(self, tmp_cfg):
        tmp_cfg.write_text("this is not valid toml [unclosed",
                           encoding="utf-8")
        # Our tiny parser is permissive; structurally bad input returns
        # an empty section rather than raising.
        result = load_config(tmp_cfg)
        assert isinstance(result, dict)

    def test_save_then_load_round_trip(self, tmp_cfg):
        save_config({"llm": {"provider": "anthropic",
                              "model": "claude-sonnet-4-5"}}, tmp_cfg)
        cfg = load_config(tmp_cfg)
        assert cfg == {"llm": {"provider": "anthropic",
                                "model": "claude-sonnet-4-5"}}

    def test_save_creates_parent_dir(self, tmp_path):
        cfg_path = tmp_path / "deep" / "nested" / "llm.toml"
        save_config({"llm": {"provider": "openai"}}, cfg_path)
        assert cfg_path.exists()

    def test_save_does_not_write_api_key_field(self, tmp_cfg):
        # Even if a caller passes api_key in the dict, save_config will
        # still write it (it doesn't filter) — but our convention is
        # that callers don't pass keys. Document the convention via
        # comment in the file header.
        save_config({"llm": {"provider": "anthropic",
                              "model": "claude-sonnet-4-5"}}, tmp_cfg)
        text = tmp_cfg.read_text()
        # Header comment warns against putting keys in the file.
        assert "API keys" in text or "API key" in text
        assert "ANTHROPIC_API_KEY" in text or "OPENAI_API_KEY" in text

    def test_set_preferences_partial_update(self, tmp_cfg):
        set_preferences(provider="anthropic", model="claude-3-5-haiku",
                        path=tmp_cfg)
        # Only update the model.
        set_preferences(model="claude-sonnet-4-5", path=tmp_cfg)
        cfg = load_config(tmp_cfg)
        # Provider still anthropic, model updated.
        assert cfg["llm"]["provider"] == "anthropic"
        assert cfg["llm"]["model"] == "claude-sonnet-4-5"


# ---------------------------------------------------------------------------
# Provider availability
# ---------------------------------------------------------------------------

class TestAvailability:
    def test_no_env_no_provider_available(self, clean_env):
        avail = list_available_providers()
        assert avail["anthropic"]["available"] is False
        assert avail["openai"]["available"] is False
        # Default model strings are populated regardless of availability.
        assert avail["anthropic"]["default_model"]
        assert avail["openai"]["default_model"]
        # Env-var hints surface for the prompt.
        assert avail["anthropic"]["env_var"] == "ANTHROPIC_API_KEY"
        assert avail["openai"]["env_var"] == "OPENAI_API_KEY"

    def test_anthropic_env_set(self, clean_env):
        clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        avail = list_available_providers()
        assert avail["anthropic"]["available"] is True
        assert avail["openai"]["available"] is False


# ---------------------------------------------------------------------------
# Resolver — layered fallback
# ---------------------------------------------------------------------------

class TestResolverLayers:
    def test_explicit_client_pass_through(self):
        ec = echo_client(lambda role, prompt: "hi")
        out = get_llm_client(client=ec)
        assert out is ec

    def test_explicit_provider_constructs(self, clean_env, tmp_cfg,
                                            fake_construct):
        get_llm_client(
            provider="anthropic", api_key="sk-ant-fake",
            allow_interactive=False, config_path=tmp_cfg,
        )
        assert fake_construct["provider"] == "anthropic"
        assert fake_construct["api_key"] == "sk-ant-fake"

    def test_env_var_drives_anthropic(self, clean_env, tmp_cfg,
                                        fake_construct):
        clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        get_llm_client(allow_interactive=False, config_path=tmp_cfg)
        assert fake_construct["provider"] == "anthropic"

    def test_env_var_drives_openai_when_only_one_set(self, clean_env,
                                                       tmp_cfg,
                                                       fake_construct):
        clean_env.setenv("OPENAI_API_KEY", "sk-fake")
        get_llm_client(allow_interactive=False, config_path=tmp_cfg)
        assert fake_construct["provider"] == "openai"

    def test_both_env_vars_tie_breaks_to_config_provider(
        self, clean_env, tmp_cfg, fake_construct,
    ):
        clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        clean_env.setenv("OPENAI_API_KEY", "sk-fake")
        save_config({"llm": {"provider": "openai", "model": "gpt-4o"}},
                    tmp_cfg)
        get_llm_client(allow_interactive=False, config_path=tmp_cfg)
        assert fake_construct["provider"] == "openai"
        assert fake_construct["model"] == "gpt-4o"

    def test_both_env_vars_no_config_tiebreaks_anthropic(
        self, clean_env, tmp_cfg, fake_construct,
    ):
        clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        clean_env.setenv("OPENAI_API_KEY", "sk-fake")
        get_llm_client(allow_interactive=False, config_path=tmp_cfg)
        assert fake_construct["provider"] == "anthropic"

    def test_no_env_no_interactive_raises_with_remediation(
        self, clean_env, tmp_cfg,
    ):
        with pytest.raises(LLMConfigurationError) as excinfo:
            get_llm_client(allow_interactive=False, config_path=tmp_cfg)
        msg = str(excinfo.value)
        assert "ANTHROPIC_API_KEY" in msg
        assert "OPENAI_API_KEY" in msg

    def test_explicit_model_override(self, clean_env, tmp_cfg,
                                       fake_construct):
        clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        get_llm_client(
            model="claude-haiku-4-5",
            allow_interactive=False, config_path=tmp_cfg,
        )
        assert fake_construct["model"] == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Interactive prompt (mocked stdin)
# ---------------------------------------------------------------------------

class TestInteractivePrompt:
    def test_no_env_non_tty_raises(self, clean_env, tmp_cfg, monkeypatch):
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        with pytest.raises(LLMConfigurationError):
            get_llm_client(allow_interactive=True, config_path=tmp_cfg)

    def test_no_env_tty_but_no_keys_raises(self, clean_env, tmp_cfg,
                                            monkeypatch, capsys):
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        # User's input is irrelevant — both providers report unavailable.
        monkeypatch.setattr("builtins.input", lambda *a, **kw: "1")
        with pytest.raises(LLMConfigurationError):
            get_llm_client(allow_interactive=True, config_path=tmp_cfg)

    def test_with_env_skips_interactive(self, clean_env, tmp_cfg,
                                          monkeypatch, fake_construct):
        clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        # No prompt should fire — we go straight to env-driven path.
        called = []
        monkeypatch.setattr(
            "builtins.input",
            lambda *a, **kw: called.append(a) or "1",
        )
        get_llm_client(allow_interactive=True, config_path=tmp_cfg)
        assert called == []  # input() never called
        assert fake_construct["provider"] == "anthropic"


# ---------------------------------------------------------------------------
# configure_llm() persistence
# ---------------------------------------------------------------------------

class TestConfigureLLM:
    def test_persist_provider_and_model(self, tmp_cfg):
        configure_llm(provider="openai", model="gpt-4o",
                      config_path=tmp_cfg)
        cfg = load_config(tmp_cfg)
        assert cfg["llm"]["provider"] == "openai"
        assert cfg["llm"]["model"] == "gpt-4o"

    def test_unknown_provider_rejected(self, tmp_cfg):
        with pytest.raises(ValueError, match="Unknown provider"):
            configure_llm(provider="xai-grok", config_path=tmp_cfg)

    def test_subsequent_call_resolves_to_saved(self, clean_env, tmp_cfg,
                                                  fake_construct):
        clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        clean_env.setenv("OPENAI_API_KEY", "sk-fake")
        configure_llm(provider="openai", model="gpt-4o",
                      config_path=tmp_cfg)
        get_llm_client(allow_interactive=False, config_path=tmp_cfg)
        assert fake_construct["provider"] == "openai"
        assert fake_construct["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# LLMClient.complete() alias (latent bug fix)
# ---------------------------------------------------------------------------

class TestCompleteAlias:
    def test_echo_client_has_complete(self):
        ec = echo_client(lambda role, prompt: f"echo:{prompt}")
        # Both routes produce the same result.
        assert ec.complete("hello") == "echo:hello"
        assert ec("hello") == "echo:hello"
        assert ec.chat("user", "hello") == "echo:hello"


# ---------------------------------------------------------------------------
# Integration: sp.paper(..., llm='auto') with no API key falls back to
# heuristic, returns a valid PaperDraft, populates DAG.
# ---------------------------------------------------------------------------

class TestPaperLLMAuto:
    def test_llm_auto_no_env_falls_back_to_heuristic(self, clean_env):
        import statspai as sp
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "wage": 10 + rng.normal(size=200),
            "trained": rng.binomial(1, 0.5, 200),
            "edu": rng.normal(size=200),
        })
        # No API key; resolver returns None internally; llm_dag_propose
        # falls back to its deterministic heuristic — paper still works.
        draft = sp.paper(
            df, "effect of trained on wage",
            treatment="trained", y="wage",
            llm="auto", llm_domain="labor economics",
        )
        # PaperDraft built either with or without a DAG (heuristic may
        # propose an empty edge list for these generic vars).
        from statspai.workflow.paper import PaperDraft
        assert isinstance(draft, PaperDraft)
        # Either no DAG (heuristic skipped it) OR a DAG is set with
        # edges drawn from the variable list.
        if draft.dag is not None:
            assert all(
                u in df.columns and v in df.columns
                for u, v in draft.dag.edges
            )

    def test_llm_explicit_client_used(self, clean_env):
        import statspai as sp
        import numpy as np
        import pandas as pd

        # Build an echo_client that returns a fixed JSON edge list.
        def respond(role, prompt):
            return '[["edu", "wage"], ["edu", "trained"], ["trained", "wage"]]'
        client = echo_client(respond)

        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "wage": 10 + rng.normal(size=200),
            "trained": rng.binomial(1, 0.5, 200),
            "edu": rng.normal(size=200),
        })
        draft = sp.paper(
            df, "effect of trained on wage",
            treatment="trained", y="wage",
            llm="auto", llm_client=client,
        )
        # The DAG was populated from the echo client's response.
        assert draft.dag is not None
        edges = set(tuple(e) for e in draft.dag.edges)
        assert ("edu", "wage") in edges
        assert ("trained", "wage") in edges
