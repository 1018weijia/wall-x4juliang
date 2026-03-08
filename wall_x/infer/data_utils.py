"""
Rotation / pose utilities used by inference dataclasses.

Historically this project referenced a `wall_x.infer.data_utils` module, but it
was missing in this repo snapshot. `RobotStateActionData` expects a small set of
helpers for converting between:
  - 3D rotation vectors (axis-angle / exponential coordinates, shape (..., 3))
  - 6D rotation representation (first two columns of rotation matrix, shape (..., 6))

LIBERO / robosuite actions use axis-angle rotation vectors, so the default
implementation here uses rotvecs (not Euler angles) even though some legacy
function names mention "euler/rpy".
"""

from __future__ import annotations

import numpy as np


def _as_batch(x: np.ndarray, last_dim: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Reshape (..., last_dim) to (N, last_dim) and return original leading shape."""
    x = np.asarray(x)
    if x.shape == (last_dim,):
        lead = (1,)
        x = x.reshape(1, last_dim)
        return x, lead
    if x.ndim < 1 or x.shape[-1] != last_dim:
        raise ValueError(f"Expected shape (..., {last_dim}), got {x.shape}")
    lead = x.shape[:-1]
    return x.reshape(-1, last_dim), lead


def _skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix for a batch of vectors v with shape (N, 3)."""
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    z = np.zeros_like(vx)
    return np.stack(
        [
            np.stack([z, -vz, vy], axis=1),
            np.stack([vz, z, -vx], axis=1),
            np.stack([-vy, vx, z], axis=1),
        ],
        axis=1,
    )  # (N, 3, 3)


def rotvec_to_matrix(rotvec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert axis-angle rotation vectors to rotation matrices.

    Args:
        rotvec: (..., 3) rotation vectors.
    Returns:
        (..., 3, 3) rotation matrices.
    """
    v, lead = _as_batch(rotvec, 3)
    theta = np.linalg.norm(v, axis=1, keepdims=True)  # (N, 1)
    # Avoid divide by zero.
    axis = v / np.maximum(theta, eps)
    K = _skew(axis)  # (N, 3, 3)

    theta_flat = theta[:, 0]
    sin_t = np.sin(theta_flat)[:, None, None]
    cos_t = np.cos(theta_flat)[:, None, None]

    I = np.eye(3, dtype=v.dtype)[None, :, :]
    R = I * cos_t + (1.0 - cos_t) * (axis[:, :, None] * axis[:, None, :]) + sin_t * K

    # For very small angles, fall back to first-order approximation: R ≈ I + skew(v)
    small = (theta_flat < 1e-6)
    if np.any(small):
        R_small = I + _skew(v)  # skew of v (not unit axis)
        R[small] = R_small[small]

    return R.reshape(*lead, 3, 3)


def matrix_to_rotvec(R: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert rotation matrices to axis-angle rotation vectors.

    Args:
        R: (..., 3, 3) rotation matrices.
    Returns:
        (..., 3) rotation vectors.
    """
    R = np.asarray(R)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected shape (..., 3, 3), got {R.shape}")
    lead = R.shape[:-2]
    M = R.reshape(-1, 3, 3)

    tr = np.trace(M, axis1=1, axis2=2)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)  # (N,)

    # axis = (1/(2 sin theta)) * [R32-R23, R13-R31, R21-R12]
    w = np.stack(
        [M[:, 2, 1] - M[:, 1, 2], M[:, 0, 2] - M[:, 2, 0], M[:, 1, 0] - M[:, 0, 1]],
        axis=1,
    )  # (N, 3)

    sin_theta = np.sin(theta)
    denom = np.maximum(2.0 * sin_theta, eps)[:, None]
    axis = w / denom
    rotvec = axis * theta[:, None]

    # Near zero: use first-order approx rotvec ≈ 0.5 * vee(R - R^T)
    small = theta < 1e-6
    if np.any(small):
        rotvec_small = 0.5 * w
        rotvec[small] = rotvec_small[small]

    return rotvec.reshape(*lead, 3)


def matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrices (...,3,3) to 6D representation (...,6)."""
    R = np.asarray(R)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"Expected shape (..., 3, 3), got {R.shape}")
    lead = R.shape[:-2]
    a1 = R[..., :, 0]
    a2 = R[..., :, 1]
    six = np.concatenate([a1, a2], axis=-1)
    return six.reshape(*lead, 6)


def sixd_to_matrix(six: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert 6D representation (...,6) to rotation matrices (...,3,3).

    Uses the standard Gram-Schmidt process to produce an orthonormal basis.
    """
    x, lead = _as_batch(six, 6)
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]

    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=1, keepdims=True), eps)
    proj = (np.sum(b1 * a2, axis=1, keepdims=True)) * b1
    b2_ = a2 - proj
    b2 = b2_ / np.maximum(np.linalg.norm(b2_, axis=1, keepdims=True), eps)
    b3 = np.cross(b1, b2)

    R = np.stack([b1, b2, b3], axis=2)  # (N, 3, 3) columns
    return R.reshape(*lead, 3, 3)


# ---------------------------------------------------------------------------
# Legacy API expected by wall_x.infer.base_dataclass
# ---------------------------------------------------------------------------


def euler_to_matrix_zyx_6d_nb(rot: np.ndarray) -> np.ndarray:
    """Legacy name: convert rotation vector (...,3) to 6D (...,6)."""
    return matrix_to_6d(rotvec_to_matrix(rot))


def so3_to_euler_zyx_batch_nb(six: np.ndarray) -> np.ndarray:
    """Legacy name: convert 6D (...,6) to rotation vector (...,3)."""
    return matrix_to_rotvec(sixd_to_matrix(six))


def compose_state_and_delta_to_abs_rpy(delta: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compose a relative rotation with a state rotation.

    Args:
        delta: (..., 3) relative rotation vectors.
        state: (3,) or (..., 3) state rotation vectors.
    Returns:
        (..., 3) absolute rotation vectors.
    """
    delta_b, lead = _as_batch(delta, 3)
    state = np.asarray(state)
    if state.shape == (3,):
        state_b = np.broadcast_to(state[None, :], delta_b.shape)
    else:
        state_b, _ = _as_batch(state, 3)
        if state_b.shape[0] == 1 and delta_b.shape[0] > 1:
            state_b = np.broadcast_to(state_b, delta_b.shape)
        if state_b.shape != delta_b.shape:
            raise ValueError(f"state shape {state.shape} is not broadcastable to delta {delta.shape}")

    R_state = rotvec_to_matrix(state_b)
    R_delta = rotvec_to_matrix(delta_b)
    R_abs = np.matmul(R_state, R_delta)
    rot_abs = matrix_to_rotvec(R_abs)
    return rot_abs.reshape(*lead, 3)

