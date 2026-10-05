from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from server.routes import router
from server.auth import router as auth_router


# =====================================================
# FASTAPI APPLICATION
# =====================================================

app = FastAPI(
    title="Driver Verification System"
)


# =====================================================
# API ROUTES
# =====================================================

app.include_router(router)

app.include_router(auth_router)


# =====================================================
# ADMIN WEBSITE
# =====================================================

FRONTEND_DIR = Path("frontend/admin")

app.mount(
    "/static/admin",
    StaticFiles(directory=FRONTEND_DIR),
    name="admin-static"
)


# =====================================================
# ROOT → ADMIN PORTAL
# =====================================================

@app.get("/")
async def root():

    return FileResponse(
        FRONTEND_DIR / "index.html"
    )