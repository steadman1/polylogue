# Polylogue

a REST API, following the OpenAI API schema, that serves self-hosted, open-source ai models of all types

## Running the API

pre-req: this project depends on mlx, so to run 

1. install `uv` and `redis` on your machine and make sure both are setup properly
2. add environment variables for known model paths. these can be empty/mock values but this might cause some tests to fail. you'll need to add the env vars to your .zshrc, .bashrc, etc.
  - [Ornith-1.5-9B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF) @ Q4_K_M is a good GGUF model for testing

```bash
echo 'export GGUF_MODEL_PATH="path/to/gguf/model.gguf"' >> ~/.zshrc # or .bashrc if that's what your machine uses
echo 'export MLX_MODEL_PATH="path/to/mlx/model"' >> ~/.zshrc && source ~/.zshrc
```


3. run the server locally with fastapi and redis
    - start your redis instance 
    - then, start the fastapi instance in another terminal

```bash
redis-server
uv run fastapi dev src/polylogue/__init__.py --reload-dir src/polylogue --port 8080
```

## Running Unit Tests with pytest

unit tests cover fastapi endpoints and inference components

```bash
uv run pytest
```

## Adding ModelRecords to Redis from the CLI

### Get a ModelRecord

```bash
uv run polylogue-cli get -m model_id
```

### Save a ModelRecord

```bash
uv run polylogue-cli save 
    -m model_id                  # OR --model-id
    -p /path/to/model            # OR --path
    -n 128_000                   # OR --n-ctx
    -d "helpful description..."  # OR --description
```

### List all ModelRecords

```bash
uv run polylogue-cli list
```

### Delete a ModelRecord

```bash
uv run polylogue-cli delete -m model_id
```

### Help

```bash
uv run polylogue-cli --help
```

## Known Issues & Fixes

### ModuleNotFoundError: No module named 'polylogue'

sometimes `uv run` will throw `ModuleNotFoundError: No module named 'polylogue'`. 

running `rm -rf .venv uv.lock && uv sync` should resolve it

### MLX lib hangs on mlx_lm.load

sometimes `mlx_lm.load(...)` will hang indefinitely (?). 

rebuilding mlx_lm with `rm -rf .venv uv.lock && uv sync` should resolve it

## TODO / Current Plan

- [ ] AGENTS.md for those who use agents in this repo
- [x] Model interface with mlx/llama.cpp
- [ ] Tool call support
- [ ] Responses API endpoint support (?)
- [x] CLI tool for adding formatted model records to db
- [x] Scalable, multi-host database (prob Redis) to store chat_completions and model_id -> model_path pairs
- [ ] Work queue to handle host machine(s) reponse generation concurrency
- [ ] Model manager to swap models (prob LRU caching) such that required_memory never exceeds available_memory
