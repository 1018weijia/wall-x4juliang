#!/usr/bin/env python3
"""
Wall-X WebSocket policy client: connect to Wall-X Policy Server (WebSocket + msgpack).
Protocol: connect -> receive metadata -> send obs -> receive { action: ndarray }.
"""

import threading
import time
from typing import Any, Dict, Optional

import numpy as np

try:
    import msgpack
    import msgpack_numpy as m
    m.patch()
except ImportError:
    raise ImportError("pip install msgpack msgpack-numpy")

try:
    import websockets
except ImportError:
    raise ImportError("pip install websockets")


class WallXWebsocketClientPolicy:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}"
        self._ws = None
        self._loop = None
        self._thread = None
        self._metadata: Optional[Dict] = None
        self._lock = threading.Lock()

    def _start_background_loop(self):
        import asyncio
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _ensure_loop(self):
        import asyncio
        if self._loop is None or not self._loop.is_running():
            self._thread = threading.Thread(target=self._start_background_loop, daemon=True)
            self._thread.start()
            while self._loop is None:
                time.sleep(0.01)

    def _run_async(self, coro):
        import asyncio
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def _connect_async(self):
        self._ws = await websockets.connect(
            self.uri, ping_interval=None, ping_timeout=None, max_size=None
        )
        raw = await self._ws.recv()
        if isinstance(raw, str):
            raise RuntimeError(f"Server sent text on connect:\n{raw}")
        self._metadata = msgpack.unpackb(raw, raw=False)

    def connect(self) -> Dict[str, Any]:
        with self._lock:
            self._run_async(self._connect_async())
            return self._metadata or {}

    def get_server_metadata(self) -> Dict[str, Any]:
        return self._metadata or {}

    async def _infer_async(self, obs: Dict) -> Dict:
        if self._ws is None:
            raise RuntimeError("Not connected. Call connect() first.")
        await self._ws.send(msgpack.packb(obs, use_bin_type=True))
        raw = await self._ws.recv()
        if isinstance(raw, str):
            raise RuntimeError(f"Server error (text):\n{raw}")
        return msgpack.unpackb(raw, raw=False)

    def infer(self, obs: Dict) -> Dict:
        """obs: face_view, left_wrist_view, right_wrist_view (H,W,3), state (7), prompt, dataset_names."""
        with self._lock:
            resp = self._run_async(self._infer_async(obs))

        action_arr = resp.get("action")
        if action_arr is None:
            action_arr = resp.get("predict_action")
        if action_arr is None:
            raise RuntimeError("Server response missing 'action'")

        action_arr = np.asarray(action_arr, dtype=np.float32)
        if action_arr.ndim == 3:
            actions = [action_arr[0, i] for i in range(action_arr.shape[1])]
        elif action_arr.ndim == 2:
            actions = [action_arr[i] for i in range(action_arr.shape[0])]
        else:
            actions = [action_arr]
        resp["actions"] = actions
        return resp

    def close(self):
        import asyncio

        async def _close():
            if self._ws:
                await self._ws.close()
                self._ws = None

        with self._lock:
            self._run_async(_close())
            if self._loop:
                self._loop.call_soon_threadsafe(self._loop.stop)
