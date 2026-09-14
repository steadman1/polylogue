from __future__ import annotations

import psutil

from polylogue.clients import app


# debug only endpoint
@app.get("/ram")
async def get_ram_usage() -> dict[str, float]:
    # Get system-wide memory details
    memory_info = psutil.virtual_memory()

    # Convert bytes to Gigabytes (GB) for readability
    total_gb: float = memory_info.total / (1024**3)
    used_gb: float = memory_info.used / (1024**3)
    available_gb: float = memory_info.available / (1024**3)

    return {
        "total": total_gb,
        "used": used_gb,
        "available": available_gb,
    }
