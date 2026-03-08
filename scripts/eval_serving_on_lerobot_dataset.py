#!/usr/bin/env python3
"""
Evaluate a running Wall-X serving server by sampling observations from a LeRobot dataset
and comparing predicted action chunks against ground-truth action chunks.

Typical usage (after you started infer_real.sh):
  HF_HOME=/tmp/hf_home_$USER conda run -n wallx python scripts/eval_serving_on_lerobot_dataset.py \
    --uri ws://127.0.0.1:32157 \
    --config workspace/lerobot_example/franka_real.yml \
    --episodes 0 \
    --num-windows 20
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from wall_x.data.utils import KEY_MAPPINGS

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
class WindowResult:
    mae: float
    rmse: float
    max_abs: float


def _tensor_image_to_uint8_chw_or_hwc(img) -> np.ndarray:
    """Convert a torch Tensor image in [0,1] to uint8 HWC for serving."""
    import torch

    if not isinstance(img, torch.Tensor):
        raise TypeError(f"expected torch.Tensor image, got {type(img)}")
    if img.ndim != 3:
        raise ValueError(f"expected image tensor with 3 dims, got shape {tuple(img.shape)}")

    # LeRobot typically returns float32 CHW in [0,1].
    if img.shape[0] in (1, 3):
        img = (img.permute(1, 2, 0) * 255.0).to(torch.uint8).cpu().numpy()
        return img
    # Fallback: assume already HWC.
    img = (img * 255.0).to(torch.uint8).cpu().numpy()
    return img


def _build_server_images(
    item: Dict,
    repo_id: str,
    server_camera_keys: List[str],
) -> Dict[str, np.ndarray]:
    """
    Build {server_cam_key: np.uint8(H,W,3)} from a LeRobot dataset item.

    Uses KEY_MAPPINGS[repo_id]["camera"] to interpret item keys.
    Also supports a common alias: if server requests left_wrist_view but dataset provides wrist_view.
    """
    if repo_id not in KEY_MAPPINGS:
        raise KeyError(f"repo_id {repo_id} missing in KEY_MAPPINGS")
    cam_map = KEY_MAPPINGS[repo_id]["camera"]  # dataset_key -> canonical_key

    canonical_images: Dict[str, np.ndarray] = {}
    for dataset_key, canonical_key in cam_map.items():
        if dataset_key not in item:
            continue
        canonical_images[canonical_key] = _tensor_image_to_uint8_chw_or_hwc(item[dataset_key])

    server_images: Dict[str, np.ndarray] = {}
    for key in server_camera_keys:
        if key in canonical_images:
            server_images[key] = canonical_images[key]
            continue

        # Common compatibility aliases (real datasets sometimes name the single wrist cam as "wrist_view").
        if key == "left_wrist_view" and "wrist_view" in canonical_images:
            server_images[key] = canonical_images["wrist_view"]
            continue
        if key == "wrist_view" and "left_wrist_view" in canonical_images:
            server_images[key] = canonical_images["left_wrist_view"]
            continue

        available = sorted(canonical_images.keys())
        raise KeyError(
            f"cannot satisfy server camera key '{key}'. "
            f"Available canonical cams from dataset: {available}. "
            f"repo_id={repo_id}"
        )

    return server_images


def _get_action_and_state(item: Dict, repo_id: str) -> Tuple[np.ndarray, np.ndarray, str]:
    import torch

    state_key = KEY_MAPPINGS[repo_id]["state"]
    action_key = KEY_MAPPINGS[repo_id]["action"]

    state = item[state_key]
    actions = item[action_key]
    task = item.get("task", "")

    if not isinstance(state, torch.Tensor):
        raise TypeError(f"expected torch.Tensor state, got {type(state)}")
    if not isinstance(actions, torch.Tensor):
        raise TypeError(f"expected torch.Tensor actions, got {type(actions)}")

    state_np = state.detach().cpu().to(torch.float32).numpy()
    actions_np = actions.detach().cpu().to(torch.float32).numpy()
    if state_np.ndim != 1:
        state_np = np.reshape(state_np, (-1,))
    return actions_np, state_np, task


def _metrics(pred: np.ndarray, gt: np.ndarray) -> WindowResult:
    err = pred - gt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    max_abs = float(np.max(np.abs(err)))
    return WindowResult(mae=mae, rmse=rmse, max_abs=max_abs)


async def _connect(uri: str):
    ws = await websockets.connect(
        uri,
        max_size=None,
        ping_interval=None,
        ping_timeout=None,
    )
    raw = await ws.recv()
    if isinstance(raw, str):
        raise RuntimeError(f"Server sent text frame on connect:\n{raw}")
    meta = msgpack.unpackb(raw, raw=False)
    return ws, meta


async def main_async(args: argparse.Namespace) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    # Work around HF cache permission issues on some systems by using a writable cache dir.
    # You can still override these from your shell.
    os.environ.setdefault("HF_HOME", f"/tmp/hf_home_{os.environ.get('USER', 'user')}")
    os.environ.setdefault(
        "HF_DATASETS_CACHE", os.path.join(os.environ["HF_HOME"], "datasets")
    )

    with open(args.config, "r") as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    repo_id = train_config["data"]["lerobot_config"]["repo_id"]
    root = args.root or train_config["data"]["lerobot_config"].get("root")
    if root is None:
        raise SystemExit("Missing dataset root; pass --root or set data.lerobot_config.root")

    horizon = int(args.pred_horizon or train_config["data"].get("action_horizon", 32))

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    fps = float(meta.fps)
    delta_timestamps = {KEY_MAPPINGS[repo_id]["action"]: [t / fps for t in range(horizon)]}

    episodes = args.episodes
    ds = LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes if episodes else None,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )

    print(f"[eval] uri={args.uri}")
    print(f"[eval] repo_id={repo_id} root={root}")
    print(f"[eval] fps={fps} horizon={horizon}")
    if episodes:
        print(f"[eval] episodes={episodes}")
    print(f"[eval] dataset_len={len(ds)}")

    ws, server_meta = await _connect(args.uri)
    try:
        server_camera_keys = server_meta.get("camera_key", None)
        if server_camera_keys is None:
            # Older servers might not include this; fall back to CLI default.
            server_camera_keys = ["face_view", "left_wrist_view", "right_wrist_view"]
        print(f"[eval] server_meta={server_meta}")
        print(f"[eval] server_camera_keys={server_camera_keys}")

        stride = int(args.stride or horizon)
        start_index = int(args.start_index)
        num_windows = int(args.num_windows)
        compare_steps = None if args.compare_steps is None else int(args.compare_steps)
        if compare_steps is not None and compare_steps <= 0:
            raise ValueError("--compare-steps must be >= 1")
        if compare_steps is not None and compare_steps > horizon:
            compare_steps = horizon

        results: List[WindowResult] = []
        checked = 0
        idx = start_index
        while checked < num_windows and idx < len(ds):
            item = ds[idx]
            gt_actions, state, task = _get_action_and_state(item, repo_id)

            # gt_actions should be (H, action_dim)
            if gt_actions.ndim != 2 or gt_actions.shape[0] != horizon:
                raise ValueError(
                    f"expected gt_actions shape ({horizon}, D), got {gt_actions.shape}"
                )

            images = _build_server_images(item, repo_id, list(server_camera_keys))
            obs = {
                **images,
                "prompt": task if args.prompt is None else args.prompt,
                "state": state.astype(np.float32),
                # IMPORTANT: server expects a STRING (it wraps into a list internally).
                "dataset_names": repo_id,
            }

            await ws.send(msgpack.packb(obs, use_bin_type=True))
            raw = await ws.recv()
            if isinstance(raw, str):
                print("[eval] Server returned a text frame (likely traceback):")
                print(raw)
                raise RuntimeError("Server error during inference; see traceback above.")
            resp = msgpack.unpackb(raw, raw=False)

            if "action" not in resp:
                raise KeyError(f"server response missing 'action'. keys={list(resp.keys())}")
            pred = resp["action"]
            if not isinstance(pred, np.ndarray):
                pred = np.asarray(pred)

            # pred is (1, H, D) in our serving implementation.
            if pred.ndim == 3 and pred.shape[0] == 1:
                pred = pred[0]

            if pred.shape != gt_actions.shape:
                raise ValueError(
                    f"shape mismatch: pred={pred.shape} gt={gt_actions.shape}. "
                    f"Check --model-config.action-dim/--model-config.pred-horizon and dataset delta_timestamps."
                )

            pred_f = pred.astype(np.float32)
            gt_f = gt_actions.astype(np.float32)
            if compare_steps is not None:
                pred_f = pred_f[:compare_steps]
                gt_f = gt_f[:compare_steps]
            r = _metrics(pred_f, gt_f)
            results.append(r)

            if args.verbose:
                print(
                    f"[eval] idx={idx} mae={r.mae:.5f} rmse={r.rmse:.5f} max_abs={r.max_abs:.5f}"
                )

            checked += 1
            idx += stride

        if not results:
            raise SystemExit("No windows evaluated (dataset too short or start_index too large).")

        mae = float(np.mean([r.mae for r in results]))
        rmse = float(np.mean([r.rmse for r in results]))
        max_abs = float(np.max([r.max_abs for r in results]))
        print(
            f"[eval] DONE windows={len(results)} stride={stride} start_index={start_index}\n"
            f"[eval] mean_mae={mae:.6f} mean_rmse={rmse:.6f} max_abs={max_abs:.6f}"
        )
    finally:
        await ws.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a running Wall-X websocket server on a LeRobot dataset."
    )
    p.add_argument("--uri", default="ws://127.0.0.1:32157")
    p.add_argument(
        "--config",
        required=True,
        help="Train config path (e.g. workspace/lerobot_example/franka_real.yml or ckpt/.../config.yml).",
    )
    p.add_argument(
        "--root",
        default=None,
        help="Override dataset root; otherwise uses config:data.lerobot_config.root",
    )
    p.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Episode indices to sample from (e.g. --episodes 0 1 2). Default: all episodes.",
    )
    p.add_argument("--num-windows", type=int, default=20)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Index stride between evaluated windows. Default: pred_horizon.",
    )
    p.add_argument(
        "--pred-horizon",
        type=int,
        default=None,
        help="Action chunk length to request/compare. Default: from config:data.action_horizon (fallback 32).",
    )
    p.add_argument(
        "--compare-steps",
        type=int,
        default=None,
        help="Only compare first K steps (e.g. 1 for next-step error). Default: compare full horizon.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override prompt; default uses dataset item['task'].",
    )
    p.add_argument("--video-backend", type=str, default="pyav")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
