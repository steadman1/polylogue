from __future__ import annotations

from polylogue.clients import app

# import all of our endpoints to reduce __init__.py file size
from polylogue.endpoints.chat import *
from polylogue.endpoints.models import *


@app.get("/")
async def root():
    print("Hello from polylogue!")
    return {"message": "Hello World"}
