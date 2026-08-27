from __future__ import annotations
import os

from polylogue.fastapi_app import app
# import all of our endpoints to reduce __init__.py file size
from polylogue.endpoints.chat import *

@app.get("/")
async def root():
    print("Hello from polylogue!")
    return {"message": "Hello World"}
