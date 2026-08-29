from __future__ import annotations

# import all of our endpoints to reduce __init__.py file size
from polylogue.endpoints.chat import *
from polylogue.endpoints.models import *
from polylogue.fastapi_app import app


@app.get("/")
async def root():
    print("Hello from polylogue!")
    return {"message": "Hello World"}
