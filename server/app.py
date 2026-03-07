"""
MarketForge - FastAPI Server
==========================================
OpenEnv v0.2.1 compatible FastAPI application.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import MarketAction, MarketObservation

# Try OpenEnv's create_app helper
try:
    from openenv_core.env_server import create_app
    from server.market_environment import MarketEnvironment

    app = create_app(
        env=MarketEnvironment,
        action_cls=MarketAction,
        observation_cls=MarketObservation,
        env_name="market_forge",
        max_concurrent_envs=8,
    )
except ImportError:
    # Fallback: build FastAPI app manually (for standalone usage)
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from dataclasses import asdict
    import json

    from server.market_environment import MarketEnvironment

    app = FastAPI(
        title="MarketForge",
        description="OpenEnv environment for multi-agent market simulation",
        version="0.2.1",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    env = MarketEnvironment()

    @app.get("/health")
    async def health():
        return {"status": "healthy", "env": "market_forge", "version": "0.2.1"}

    @app.post("/reset")
    async def reset(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        obs = env.reset(**body)
        return JSONResponse(content=_obs_to_dict(obs))

    @app.post("/step")
    async def step(request: Request):
        body = await request.json()
        action = MarketAction(**{k: v for k, v in body.items()
                                  if k in MarketAction.__dataclass_fields__})
        obs = env.step(action)
        return JSONResponse(content=_obs_to_dict(obs))

    @app.get("/state")
    async def get_state():
        state = env.state
        return JSONResponse(content=asdict(state))

    def _obs_to_dict(obs):
        return asdict(obs)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
