"""Integration tests for multi-component StatsPAI workflows.

These tests exercise end-to-end pipelines that wire several modules
together — e.g. LLM-driven causal discovery feeding into DAG analysis,
or assimilation on top of per-batch estimation.  Unlike unit tests they
may involve mocked / deterministic external services (the
``echo_client`` LLM, pre-computed data files) but **never** hit real
network endpoints.
"""
