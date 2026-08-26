from __future__ import annotations
from polylogue.fastapi_app import app

# import all of our endpoints to reduce __init__.py file size
from .endpoints.chat import *

@app.get("/")
async def root():
    print("Hello from polylogue!")
    return {"message": "Hello World"}
