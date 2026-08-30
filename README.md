# Polylogue

a REST API, following the OpenAI API schema, that serves self-hosted, open-source ai models of all types

## Getting Started

pre-req: this project depends on mlx, so to run 

1. install `uv` and `redis` on your machine and make sure both are setup properly
2. add environment variables for known model paths. these can be empty/mock values but this might cause some tests to fail. you'll need to add the env vars to your .zshrc, .bashrc, etc.

```bash
echo 'GGUF_MODEL_PATH="path/to/gguf/model.gguf"' >> ~/.zshrc && source ~/.zshrc # or .bashrc if that's what your machine uses
echo 'MLX_MODEL_PATH="path/to/mlx/model"' >> ~/.zshrc && source ~/.zshrc
```

3. run the unit tests with pytest `uv run pytest`
4. run the server locally with fastapi and redis
    - start your redis instance `redis-server`
    - then, start the fastapi instance in another terminal `uv run fastapi dev src/polylogue/__init__.py`

## TODO / Current Plan

- [ ] AGENTS.md for those who use agents in this repo
- [x] Model interface with mlx/llama.cpp
- [ ] CLI tool for adding formatted model records to db
- [x] Scalable, multi-host database (prob Redis) to store chat_completions and model_id -> model_path pairs
- [ ] Work queue to handle host machine(s) reponse generation concurrency
- [ ] Model manager to swap models (prob LRU caching) such that required_memory never exceeds available_memory
