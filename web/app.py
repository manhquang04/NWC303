"""FastAPI server for SDN-DRL-IDS web dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import CFG

log = logging.getLogger(__name__)

app = FastAPI(title="SDN-DRL-IDS Dashboard", version="1.0.0")

_poll_task: Optional[asyncio.Task] = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_bridge = None


def set_bridge(bridge) -> None:
    global _bridge
    _bridge = bridge


def get_bridge():
    return _bridge


async def _poll_loop() -> None:
    """Background task: poll FlowCollector and update bridge state every 500ms."""
    while True:
        try:
            bridge = get_bridge()
            if bridge is not None and bridge.flow_collector is not None:
                snap = bridge.flow_collector.get_latest()
                if snap is not None:
                    bridge.update_state()
        except Exception:
            log.debug("Poll loop iteration failed", exc_info=True)
        await asyncio.sleep(0.5)


@app.on_event("startup")
async def start_poll_loop() -> None:
    global _poll_task
    _poll_task = asyncio.create_task(_poll_loop())
    log.info("Background poll loop started.")


@app.on_event("shutdown")
async def stop_poll_loop() -> None:
    global _poll_task
    if _poll_task is not None:
        _poll_task.cancel()
        try:
            await _poll_task
        except asyncio.CancelledError:
            pass


class ConnectionManager:

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        log.info("WebSocket connected. Total: %d", len(self.active))

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)
        log.info("WebSocket disconnected. Total: %d", len(self.active))

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


ws_manager = ConnectionManager()


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            bridge = get_bridge()
            if bridge is not None:
                state = bridge.get_state()
                await ws.send_json({
                    "type": "state_update",
                    "data": {
                        "timestamp": state.timestamp,
                        "state_vector": state.state_vector,
                        "raw_features": state.raw_features,
                        "feature_names": state.feature_names,
                        "current_action": state.current_action,
                        "current_action_name": state.current_action_name,
                        "ground_truth": state.ground_truth,
                        "attack_type": state.attack_type,
                        "step_reward": state.step_reward,
                        "step_count": state.step_count,
                        "target": state.target,
                        "episode": state.episode,
                        "epsilon": state.epsilon,
                        "cumulative_reward": state.cumulative_reward,
                        "metrics": state.metrics,
                        "recent_events": state.recent_events,
                        "arp_table": state.arp_table,
                        "mac_table": state.mac_table,
                    },
                })
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception:
        ws_manager.disconnect(ws)


@app.get("/api/topology")
async def get_topology():
    bridge = get_bridge()
    if bridge is None:
        return {"error": "Bridge not initialized"}
    return bridge.get_topology()


@app.get("/api/state")
async def get_state():
    bridge = get_bridge()
    if bridge is None:
        return {"error": "Bridge not initialized"}
    state = bridge.get_state()
    return {
        "timestamp": state.timestamp,
        "state_vector": state.state_vector,
        "raw_features": state.raw_features,
        "current_action": state.current_action,
        "current_action_name": state.current_action_name,
        "ground_truth": state.ground_truth,
        "attack_type": state.attack_type,
        "step_reward": state.step_reward,
        "step_count": state.step_count,
        "target": state.target,
        "episode": state.episode,
        "epsilon": state.epsilon,
        "cumulative_reward": state.cumulative_reward,
    }


@app.get("/api/metrics")
async def get_metrics():
    bridge = get_bridge()
    if bridge is None:
        return {"error": "Bridge not initialized"}
    return bridge.get_metrics()


@app.get("/api/history")
async def get_history(limit: int = 100):
    bridge = get_bridge()
    if bridge is None:
        return {"error": "Bridge not initialized"}
    return {"history": bridge.get_action_history(limit)}


@app.get("/api/events")
async def get_events(limit: int = 50):
    bridge = get_bridge()
    if bridge is None:
        return {"error": "Bridge not initialized"}
    return {"events": bridge.get_event_log(limit)}


@app.get("/api/features")
async def get_features():
    bridge = get_bridge()
    if bridge is None:
        return {"error": "Bridge not initialized"}
    return bridge.get_feature_info()


@app.get("/api/snapshot")
async def get_snapshot():
    bridge = get_bridge()
    if bridge is None:
        return {"error": "Bridge not initialized"}
    return bridge.get_network_snapshot()


@app.post("/api/action")
async def manual_action(action: int = 0, ground_truth: str = "attack"):
    bridge = get_bridge()
    if bridge is None:
        return {"error": "Bridge not initialized"}
    bridge.update_state(action=action, ground_truth=ground_truth, reward=0.0)
    return {"status": "ok", "action": action, "ground_truth": ground_truth}


@app.get("/api/config")
async def get_config():
    return {
        "topology": {
            "num_switches": CFG.topology.num_switches,
            "num_hosts": CFG.topology.num_hosts,
            "rogue_host_idx": CFG.topology.rogue_host_idx,
            "spoofer_host_idx": CFG.topology.spoofer_host_idx,
            "link_bw_mbps": CFG.topology.link_bw_mbps,
        },
        "detection": {
            "poll_interval_ms": CFG.detection.poll_interval_ms,
            "state_dim": CFG.detection.state_dim,
        },
        "dqn": {
            "hidden_layers": list(CFG.dqn.hidden_layers),
            "learning_rate": CFG.dqn.learning_rate,
            "gamma": CFG.dqn.gamma,
            "batch_size": CFG.dqn.batch_size,
            "eps_start": CFG.dqn.eps_start,
            "eps_end": CFG.dqn.eps_end,
        },
        "reward": {
            "r_attack_blocked": CFG.reward.r_attack_blocked,
            "r_attack_flagged": CFG.reward.r_attack_flagged,
            "r_attack_ignored": CFG.reward.r_attack_ignored,
            "r_normal_allowed": CFG.reward.r_normal_allowed,
            "r_normal_flagged": CFG.reward.r_normal_flagged,
            "r_normal_blocked": CFG.reward.r_normal_blocked,
        },
        "action_names": ["allow", "flag", "block", "isolate"],
    }


FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return HTMLResponse(
        "<h1>SDN-DRL-IDS Dashboard</h1>"
        "<p>Frontend not built. Run: <code>cd web/frontend && npm run build</code></p>"
    )


if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")
