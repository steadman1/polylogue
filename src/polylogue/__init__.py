from __future__ import annotations

import os

# import all of our endpoints to reduce __init__.py file size
from polylogue.endpoints.chat import create_chat_completion, list_chat_completions
from polylogue.fastapi_app import app


@app.get("/")
async def root():
    print("Hello from polylogue!")
    return {"message": "Hello World"}
