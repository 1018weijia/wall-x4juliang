#!/usr/bin/env python3
import argparse
import asyncio
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import msgpack
    import msgpack_numpy as m

    m.patch()
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing deps: msgpack + msgpack-numpy. Install with:\n"
        "  python -m pip install -U msgpack msgpack-numpy\n"
        f"Original error: {e}"
    )

try:
    import websockets
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dep: websockets. Install with:\n"
        "  python -m pip install -U websockets\n"
        f"Original error: {e}"
    )


@dataclass
class Obs:
    face_view: np.ndarray
    left_wrist_view: np.ndarray
    right_wrist_view: np.ndarray
    state: np.ndarray
    prompt: str
    dataset_names: str

    def to_dict(self) -> dict:
        return {
            "face_view": self.face_view,
            "left_wrist_view": self.left_wrist_view,
            "right_wrist_view": self.right_wrist_view,
            "state": self.state,
            "prompt": self.prompt,
            # IMPORTANT: server code expects a STRING here (it will wrap it into a list internally).
            "dataset_names": self.dataset_names,
        }


def _load_image_rgb_u8(path: str) -> np.ndarray:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)  # (H, W, 3)
    return arr


def _parse_state_csv(s: str, expected_dim: int = 7) -> np.ndarray:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    vals = [float(p) for p in parts]
    if len(vals) != expected_dim:
        raise ValueError(f"--state must have {expected_dim} floats, got {len(vals)}")
    return np.asarray(vals, dtype=np.float32)


def _dummy_image(size: int = 256) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


async def run_once(uri: str, obs: Obs, timeout_s: Optional[float] = 30.0) -> dict:
    async with websockets.connect(
        uri,
        max_size=None,
        ping_interval=None,
        ping_timeout=None,
        close_timeout=timeout_s,
    ) as ws:
        # Server sends metadata on connect.
        raw = await ws.recv()
        if isinstance(raw, str):
            raise RuntimeError(f"Server sent text frame on connect:\n{raw}")
        meta = msgpack.unpackb(raw, raw=False)
        print("[client] server metadata:", meta)

        await ws.send(msgpack.packb(obs.to_dict(), use_bin_type=True))
        raw = await ws.recv()
        if isinstance(raw, str):
            print("[client] Server returned a text frame (likely traceback):")
            print(raw)
            raise RuntimeError("Server error during inference; see traceback above.")
        resp = msgpack.unpackb(raw, raw=False)
        return resp


def main() -> None:
    p = argparse.ArgumentParser(
        description="Minimal Wall-X serving client for real-robot inference."
    )
    p.add_argument("--uri", default="ws://127.0.0.1:32157")
    p.add_argument("--dataset-name", default="stack_bowls_rc")
    p.add_argument("--prompt", default="stack the bowls")
    p.add_argument(
        "--state",
        default="0,0,0,0,0,0,0",
        help="7 floats CSV: x,y,z,rx,ry,rz,gripper (must match training proprio).",
    )
    p.add_argument("--face-image", default=None)
    p.add_argument("--left-wrist-image", default=None)
    p.add_argument("--right-wrist-image", default=None)
    p.add_argument(
        "--dummy",
        action="store_true",
        help="Use black dummy images (connectivity smoke test).",
    )
    args = p.parse_args()

    state = _parse_state_csv(args.state, expected_dim=7)

    if args.dummy:
        face = _dummy_image(256)
        left = _dummy_image(256)
        right = _dummy_image(256)
    else:
        missing: List[str] = []
        if args.face_image is None:
            missing.append("--face-image")
        if args.left_wrist_image is None:
            missing.append("--left-wrist-image")
        if args.right_wrist_image is None:
            missing.append("--right-wrist-image")
        if missing:
            raise SystemExit(
                "Missing image args (or use --dummy): " + ", ".join(missing)
            )

        face = _load_image_rgb_u8(args.face_image)
        left = _load_image_rgb_u8(args.left_wrist_image)
        right = _load_image_rgb_u8(args.right_wrist_image)

    obs = Obs(
        face_view=face,
        left_wrist_view=left,
        right_wrist_view=right,
        state=state,
        prompt=args.prompt,
        dataset_names=args.dataset_name,
    )

    resp = asyncio.run(run_once(args.uri, obs))
    print("[client] response keys:", list(resp.keys()))
    if "action" in resp and isinstance(resp["action"], np.ndarray):
        print("[client] action shape:", resp["action"].shape, resp["action"].dtype)
        print("[client] first action:", resp["action"][0, 0].astype(np.float32))
    if "predict_action" in resp and isinstance(resp["predict_action"], np.ndarray):
        print(
            "[client] predict_action shape:",
            resp["predict_action"].shape,
            resp["predict_action"].dtype,
        )


if __name__ == "__main__":
    main()
