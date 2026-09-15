# Polylogue

a REST API, following the OpenAI API schema, that serves self-hosted, open-source ai models of all types

## Running the API

pre-req: this project depends on mlx, so to run 

1. install `uv` and `redis` on your machine and make sure both are setup properly
2. add environment variables for known model paths. these can be empty/mock values but this might cause some tests to fail. you'll need to create a .env file in the root dir and add your variables here
  - [Ornith-1.5-9B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF) @ Q4_K_M is a good GGUF model for testing

```bash
# inside polylogue/.env
GGUF_MODEL_PATH="path/to/gguf/model.gguf"
MLX_MODEL_PATH="path/to/mlx/model"

API_KEY_PEPPER="your-api-key-pepper..."
```

3. run the server locally with fastapi and redis
    - start your redis instance 
    - then, start the fastapi instance in another terminal

```bash
redis-server
```

```bash
# in another terminal
uv run fastapi dev src/polylogue/__init__.py --reload-dir src/polylogue --port 8080
```

## Running Unit Tests with pytest

unit tests cover fastapi endpoints and inference components

```bash
uv run pytest
```

## Managing ModelRecords from the CLI

### Get a ModelRecord

```bash
uv run polylogue-cli db get -m model_id
```

### Save a ModelRecord

```bash
uv run polylogue-cli db save 
    -m model_id                  # OR --model-id
    -p /path/to/model            # OR --path
    -n 128_000                   # OR --n-ctx
    -d "helpful description..."  # OR --description

uv run polylogue-cli save -m model_id -p /path/to/model -n 128_000 -d "helpful description..."
```

### List all ModelRecords

```bash
uv run polylogue-cli db list
```

### Delete a ModelRecord

```bash
uv run polylogue-cli db delete -m model_id
```

### Help

```bash
uv run polylogue-cli --help
```

## Managing API Keys from the CLI

### Create an API Key

```bash
uv run polylogue-cli api create \
    -o owner_id          # OR --owner-id
    -n "key-name"        # OR --name (default: "default")
    -r 100               # OR --rate-limit (default: 100 req/min)

uv run polylogue-cli api create -o user_123 -n "dev-laptop" -r 120
```
The complete secret key (e.g., sk_live_...) is displayed only once upon creation. Store it immediately.

### Get API Key Metadata

```bash
uv run polylogue-cli api get -k sk_live_key_id # OR --key-id
```

### Verify an API Key

```bash
uv run polylogue-cli api verify sk_live_key_id_secret_token
```

### List API Keys by Owner

```bash
uv run polylogue-cli api list -o owner_id # OR --owner-id
```

### List All Owners

```bash
uv run polylogue-cli api list-owners
```

### Delete an API Key

```bash
uv run polylogue-cli api delete -k sk_live_key_id # OR --key-id
```


### Help

```bash 
uv run polylogue-cli api --help
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
- [ ] Set up docker and cloudflared config
