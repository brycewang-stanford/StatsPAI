# LLM-DAG Setup — Credential Resolution & Auto-Propose

> Three layers of fallback so `sp.paper(..., llm='auto')` "just works":
> environment variable → config file → terminal prompt → fail with
> concrete remediation. API keys never live in a plaintext config
> file (industry-standard split).

## TL;DR

```bash
# Anthropic Claude:
export ANTHROPIC_API_KEY=sk-ant-...

# OR OpenAI GPT:
export OPENAI_API_KEY=sk-...
```

Then in Python:

```python
import statspai as sp
draft = sp.paper(df, "effect of trained on wage",
                 treatment="trained", y="wage",
                 llm="auto")
```

That's it. The resolver detects whichever env var is set, picks a
sensible default model, calls `llm_dag_propose`, and attaches the
returned DAG to your draft. See [Resolution order](#resolution-order)
for the full layered fallback.

## Why a layered resolver

StatsPAI's LLM credential handling follows the same pattern as the
Anthropic SDK, OpenAI SDK, AWS CLI, `kubectl`, and `huggingface_hub`:

1. **API keys live in environment variables**, never in a plaintext
   config file. Plaintext keys leak — committed to dotfiles repos
   by accident, synced to cloud backups without encryption, sometimes
   world-readable when users forget `chmod 600`.
2. **Provider + model preferences live in a config file**
   (`~/.config/statspai/llm.toml`), so you don't have to retype them
   every call.
3. **Interactive prompts only on TTY**, never inside a Jupyter kernel
   or agent script (where `input()` blocks indefinitely).

The result: you set `ANTHROPIC_API_KEY` once at machine setup, and
every StatsPAI call that needs an LLM finds it without further
configuration.

## Resolution order

`sp.causal_llm.get_llm_client()` walks the layers below in order;
**first match wins**.

### 1. Explicit `client=`

Pass an already-built `LLMClient` instance:

```python
client = sp.causal_llm.anthropic_client(
    model="claude-sonnet-4-5",
    api_key="sk-ant-...",
)
client = sp.causal_llm.get_llm_client(client=client)  # pass-through
```

Useful for tests (inject a mock / `echo_client`) and for advanced
users who roll their own retry / caching wrapper.

### 2. Explicit `provider=` + `api_key=`

```python
client = sp.causal_llm.get_llm_client(
    provider="anthropic",
    api_key="sk-ant-...",
    model="claude-sonnet-4-5",   # optional; defaults per provider
)
```

Forces a specific provider regardless of env state. Use this when
you want one Python process to talk to one provider while another
talks to another.

### 3. Environment variable auto-detect

```bash
export ANTHROPIC_API_KEY=sk-ant-...    # → Anthropic, default model
# OR
export OPENAI_API_KEY=sk-...           # → OpenAI, default model
```

```python
client = sp.causal_llm.get_llm_client()   # picks whichever is set
```

When **both** are set, the resolver tie-breaks via the config file's
`[llm].provider`. If no config file exists, it tie-breaks to
Anthropic.

### 4. Config file `~/.config/statspai/llm.toml`

Stores **provider** and **model** preferences (XDG-Base-Directory
compliant). API keys are never written here.

```toml
[llm]
provider = "anthropic"
model = "claude-sonnet-4-5"
```

Set via the convenience helper:

```python
sp.causal_llm.configure_llm(provider="openai", model="gpt-4o")
```

Or check current state:

```python
sp.causal_llm.list_available_providers()
# → {'anthropic': {'available': True, 'default_model': 'claude-sonnet-4-5',
#                  'env_var': 'ANTHROPIC_API_KEY'},
#    'openai':    {'available': False, 'default_model': 'gpt-4o-mini',
#                  'env_var': 'OPENAI_API_KEY'}}
```

The path follows the platform convention:

| Platform | Path |
|---|---|
| Linux / macOS | `${XDG_CONFIG_HOME:-~/.config}/statspai/llm.toml` |
| Windows | `%APPDATA%\statspai\llm.toml` |

Find it programmatically:

```python
print(sp.causal_llm.llm_config_path())
```

### 5. Interactive prompt (TTY only)

When all of the above fail and `sys.stdin.isatty()` is True, the
resolver walks the user through a provider + model selection:

```text
StatsPAI: pick an LLM provider for this session.
Available providers (env-var key set):
  1. ✓ anthropic [default] — model=claude-sonnet-4-5
  2. ✗ openai (set OPENAI_API_KEY first) — model=gpt-4o-mini
Choice [1]:
Model [claude-sonnet-4-5]:
```

The prompt **never asks for an API key over stdin** — that path is
deliberately closed. Reasons:

- Shell history (`bash`/`zsh`/`fish`) records `input()` lines.
- Pasted keys leave fingerprints in scrollback / tmux logs.
- No clean integration with system keyrings (macOS Keychain, Linux
  secret-service, Windows Credential Manager).

If you want to provision a key from inside Python (e.g., reading
from a keyring), use **Layer 2** explicitly:

```python
import keyring   # pip install keyring
client = sp.causal_llm.get_llm_client(
    provider="anthropic",
    api_key=keyring.get_password("statspai", "anthropic"),
)
```

Pass `allow_interactive=False` to disable the prompt entirely (agent
/ Jupyter contexts where `input()` would hang the kernel):

```python
client = sp.causal_llm.get_llm_client(allow_interactive=False)
# Raises LLMConfigurationError if no env var is set.
```

### 6. Hard error

When all layers fail (no env var, no config, non-TTY or
`allow_interactive=False`), the resolver raises with concrete
remediation:

```text
LLMConfigurationError: No LLM provider configured. Set one of these env vars:
  export ANTHROPIC_API_KEY=...     # for Claude
  export OPENAI_API_KEY=...        # for GPT-4 / o-series
Then call again. Or pass an explicit `client=` instance / use
`sp.causal_llm.configure_llm(provider=..., model=...)` to save your
provider+model preference. API keys always come from the environment,
never from the config file.
```

## Default models

`sp.causal_llm.DEFAULT_LLM_MODELS`:

| Provider  | Default                |
|---        |---                     |
| anthropic | `claude-sonnet-4-5`    |
| openai    | `gpt-4o-mini`          |

Override per-call:

```python
client = sp.causal_llm.get_llm_client(model="claude-haiku-4-5")
```

Override globally for the machine:

```python
sp.causal_llm.configure_llm(model="claude-haiku-4-5")
```

## Auto-DAG inside `sp.paper`

The resolver is wired into `sp.paper(..., llm='auto')` so the typical
workflow is "set env var, forget":

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

draft = sp.paper(
    df,
    "effect of trained on wage controlling for education",
    treatment="trained", y="wage",
    covariates=["edu"],
    llm="auto",                              # opt in
    llm_domain="labor economics, training programmes",
    fmt="qmd",
)
```

What happens internally:

1. `sp.paper()` parses the question, infers `y` / `treatment`.
2. `dag is None and llm == "auto"` → call `get_llm_client(allow_interactive=False)`.
3. If a client resolves, call `llm_dag_propose(variables=df.columns, domain=llm_domain, client=client)`.
4. Materialise the proposed edges as a `statspai.dag.graph.DAG`.
5. Attach to `PaperDraft.dag` so `to_qmd()` renders the mermaid block
   and `replication_pack` packs the DAG into the archive.

**Failure modes** (all silent — auto-DAG must never break the paper
pipeline):

- No env var set → resolver raises → caught → fall back to no-DAG.
- Network error / rate limit → SDK raises → caught → fall back.
- Malformed JSON in LLM response → `llm_dag_propose` falls back to
  its deterministic heuristic backend.

To force the offline heuristic path (no API call, no network):

```python
draft = sp.paper(..., llm="heuristic")
```

To inject a specific client and skip resolution entirely:

```python
draft = sp.paper(..., llm="auto", llm_client=my_client)
```

## Privacy & cost

- **What gets sent to the LLM**: the variable names + a single
  free-text `llm_domain` string. Nothing else — no row data, no
  parameter values, no PII.
- **What does NOT get sent**: any cell of the DataFrame, any
  estimator output, any column dtypes, any user identifiers.
- **Cost**: a single ~200-token request per paper. At Anthropic's
  Sonnet pricing (~$3/MTok input), each `sp.paper(llm='auto')` costs
  well under $0.01.
- **Caching**: the resolver does not cache. Each call hits the
  provider. Use `result.dag` on the returned draft and pass it to
  subsequent `sp.paper(..., dag=g)` calls if you want zero further
  network round-trips.

## Diagnostics

```python
# What's available right now?
sp.causal_llm.list_available_providers()

# Where is my config?
sp.causal_llm.llm_config_path()

# What's in it?
sp.causal_llm.load_llm_config()

# Wipe it (just delete the file):
sp.causal_llm.llm_config_path().unlink()
```

## See also

- [Replication workflow](replication_workflow.md) — end-to-end paper
  + archive pipeline that consumes auto-DAG.
- [LLM-assisted causal discovery (closed loop)](llm_dag_family.md) —
  `llm_dag_constrained` + `llm_dag_validate`, the next layer that
  actually validates LLM-proposed DAGs against the data.
- [Causal MAS](causal_mas.md) — multi-agent LLM critique pipeline.
