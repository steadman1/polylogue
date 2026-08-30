# Polylogue

a REST API, following the OpenAI API schema, that serves self-hosted, open-source ai models of all types

## Getting Started

```
```

## TODO / Current Plan

- [ ] AGENTS.md for those who use agents in this repo
- [x] Model interface with mlx/llama.cpp
- [ ] CLI tool for adding formatted model records to db
- [x] Scalable, multi-host database (prob Redis) to store chat_completions and model_id -> model_path pairs
- [ ] Work queue to handle host machine(s) reponse generation concurrency
- [ ] Model manager to swap models (prob LRU caching) such that required_memory never exceeds available_memory
