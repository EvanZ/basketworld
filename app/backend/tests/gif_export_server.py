"""Isolated GIF-only integration server; no environment/model is initialized.

python -m app.backend.tests.gif_export_server /tmp/unique-output-directory
"""
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.backend.routes.media_routes import router, episode_exports

app = FastAPI()
app.include_router(router)
stats = {"requests": 0, "total_bytes": 0, "max_request_bytes": 0}


@app.middleware("http")
async def measure_upload(request, call_next):
    if request.url.path.endswith("/frames"):
        size = int(request.headers.get("content-length", 0))
        stats["requests"] += 1
        stats["total_bytes"] += size
        stats["max_request_bytes"] = max(stats["max_request_bytes"], size)
    return await call_next(request)


@app.get("/test/export_stats")
def export_stats():
    return {**stats, "active_exports": len(episode_exports.sessions),
            "files": [str(path.resolve()) for path in Path("episodes").glob("*.gif")]}


@app.get("/test/exported.gif")
def exported_gif():
    path = max(Path("episodes").glob("*.gif"), key=lambda item: item.stat().st_mtime)
    return FileResponse(path, media_type="image/gif")


if __name__ == "__main__":
    os.chdir(sys.argv[1])
    uvicorn.run(app, host="127.0.0.1", port=8081, access_log=False)
