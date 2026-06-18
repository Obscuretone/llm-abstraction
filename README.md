# llm-abstraction

Async LLM backend abstraction with shared `Retry-After` handling.

Only Mistral chat completions are implemented right now, but the package is
structured around a generic async backend interface.

```python
from llm_abstraction import MistralBackend

backend = MistralBackend(api_key="...")
response = await backend.chat(
    [
        {"role": "user", "content": "Hello"},
    ],
)
print(response.content)

embeddings = await backend.embed(["Hello"])
print(embeddings.embeddings[0])
```

## Build

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
mypy
pyright
python -m build
twine check dist/*
```

## Publish

TestPyPI:

```sh
twine upload --repository testpypi dist/*
```

PyPI:

```sh
twine upload dist/*
```

After publishing, downstream projects can depend on:

```txt
llm-abstraction>=0.1.0
```
