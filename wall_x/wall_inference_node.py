#!/usr/bin/env python3
"""
Wall-X inference node: subscribe cameras + robot state, connect to Wall-X policy server,
run inference, publish actions to control node. Supports action plots and concatenated trajectory plot.
"""

import copy
import dataclasses
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Bool
import tyro

from wall_x_websocket_client import WallXWebsocketClientPolicy


@dataclasses.dataclass
class Args:
    max_actions_to_publish: int = 16
    use_last_actions: bool = False
    gripper_threshold: float = 0.07
    max_state_age: float = 0.1
    action_publish_interval: float = 0.01
    delay_steps_auto: bool = False
    delay_steps_manual: int = 0
    inference_frequency: float = 1.0
    save_action_plots: bool = True
    save_concatenated_plot: bool = True
    reverse_actions: bool = False
    use_last_action_as_state: bool = False
    state_blend_robot: float = 0.5
    action_smoothing_alpha: float = 0.45

    image_width: int = 640
    image_height: int = 480

    policy_server_host: str = "localhost"
    policy_server_port: int = 8000
    task: str = "stack the bowls"
    dataset_name: str = "stack_bowls_rc"

    cam_high_serial: str = ""
    cam_wrist_serial: str = ""
    cam_side_serial: str = ""

    ee_states_topic: str = "/franka/ee_states"
    action_topic: str = "/franka/action_command"
    queue_status_topic: str = "/franka/queue_status"
    allow_inference_topic: str = "/franka/allow_inference"


class WallInferenceNode(Node):
    def __init__(self, args: Args):
        super().__init__('wall_inference_node')
        self.args = args
        self._setup_file_logging()

        self.policy = WallXWebsocketClientPolicy(
            host=args.policy_server_host,
            port=args.policy_server_port,
        )
        self.policy.connect()
        self.get_logger().info(f"Server metadata: {self.policy.get_server_metadata()}")

        self.latest_images = {'face_view': None, 'left_wrist_view': None, 'right_wrist_view': None}
        self.latest_ee_states = None
        self.last_ee_state_time = None
        self.state_lock = threading.Lock()
        self.queue_is_empty = False
        self.queue_status_lock = threading.Lock()
        self.can_infer = False
        self.allow_inference = True
        self.allow_inference_lock = threading.Lock()

        self.camera_pipelines = {}
        self.camera_frame_queues = {}
        self._init_cameras()
        self._start_camera_threads()

        self.ee_states_sub = self.create_subscription(
            JointState, args.ee_states_topic, self._on_ee_states, 10
        )
        self.queue_status_sub = self.create_subscription(
            Bool, args.queue_status_topic, self._on_queue_status, 10
        )
        self.allow_inference_sub = self.create_subscription(
            Bool, args.allow_inference_topic, self._on_allow_inference, 10
        )
        self.action_pub = self.create_publisher(Float32MultiArray, args.action_topic, 10)

        inference_period = 1.0 / args.inference_frequency
        self.inference_timer = self.create_timer(inference_period, self._inference_callback)

        self.inference_count = 0
        self.last_inference_time = 0.0
        self.all_chunks: list = []

        self._session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._face_view_frames: list = []
        self._max_face_frames = 10

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"[Wall-X] Started | task: {args.task}")
        self.get_logger().info(
            f"[Wall-X] {args.inference_frequency}Hz | {args.policy_server_host}:{args.policy_server_port}"
        )
        self.get_logger().info(
            f"[Wall-X] reverse_actions={args.reverse_actions} | "
            f"state={'blend' if args.use_last_action_as_state else 'robot'} | alpha={args.action_smoothing_alpha}"
        )
        self.get_logger().info("=" * 60)

    def _setup_file_logging(self):
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"inference_{timestamp}.log"
        self.file_logger = logging.getLogger("inference_file")
        self.file_logger.setLevel(logging.INFO)
        if not self.file_logger.handlers:
            handler = logging.FileHandler(log_file, encoding='utf-8')
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.file_logger.addHandler(handler)
        self.get_logger().info(f"Log file: {log_file}")

    def _plot_actions(self, actions: list, save_path: str, title: str = "Action Chunk"):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            arr = np.array(actions)
            if arr.size == 0:
                return
            dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
            fig, axes = plt.subplots(3, 3, figsize=(12, 10))
            axes = axes.flatten()
            for i in range(7):
                axes[i].plot(arr[:, i], 'b-', linewidth=1.5)
                axes[i].set_title(dim_names[i])
                axes[i].set_xlabel('step')
                axes[i].grid(True, alpha=0.3)
            axes[7].axis('off')
            axes[8].axis('off')
            fig.suptitle(title, fontsize=14)
            plt.tight_layout()
            plt.savefig(save_path, dpi=100)
            plt.close()
        except Exception as e:
            self.get_logger().warn(f"Plot save failed: {e}")

    def _plot_concatenated_chunks(self, all_chunks: list, save_path: str):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            if not all_chunks:
                return
            concat = np.vstack(all_chunks)
            total_steps = concat.shape[0]
            dim_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
            fig, axes = plt.subplots(3, 3, figsize=(14, 10))
            axes = axes.flatten()
            for i in range(7):
                axes[i].plot(concat[:, i], 'b-', linewidth=1.0)
                axes[i].set_title(dim_names[i])
                axes[i].set_xlabel('step (all chunks)')
                axes[i].grid(True, alpha=0.3)
                axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[7].text(
                0.5, 0.5, f'Total steps: {total_steps}\nChunks: {len(all_chunks)}',
                ha='center', va='center', fontsize=12, transform=axes[7].transAxes
            )
            axes[7].axis('off')
            axes[8].axis('off')
            fig.suptitle(f'Concatenated Trajectory ({len(all_chunks)} chunks)', fontsize=14)
            plt.tight_layout()
            plt.savefig(save_path, dpi=100)
            plt.close()
        except Exception as e:
            self.get_logger().warn(f"Concatenated plot save failed: {e}")

    def _init_cameras(self):
        camera_serials = {
            'face_view': self.args.cam_wrist_serial,
            'left_wrist_view': self.args.cam_side_serial,
            'right_wrist_view': self.args.cam_high_serial,
        }
        for cam_name, serial in camera_serials.items():
            try:
                pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_device(serial)
                cfg.enable_stream(
                    rs.stream.color,
                    self.args.image_width, self.args.image_height,
                    rs.format.yuyv, 30
                )
                fq = rs.frame_queue(50)
                pipeline.start(cfg, fq)
                self.camera_pipelines[cam_name] = pipeline
                self.camera_frame_queues[cam_name] = fq
                self.get_logger().info(f"[Camera] {cam_name} (SN:{serial}) started")
                time.sleep(1)
            except Exception as e:
                self.get_logger().error(f"[Camera] {cam_name} (SN:{serial}) failed: {e}")

    def _camera_capture_thread(self, cam_name: str, fq: rs.frame_queue):
        H, W = self.args.image_height, self.args.image_width
        while rclpy.ok():
            try:
                frame = fq.wait_for_frame(timeout_ms=2000)
                try:
                    color_frame = frame.as_frameset().get_color_frame()
                except Exception:
                    color_frame = frame
                if not color_frame:
                    continue
                img_yuyv = np.asanyarray(color_frame.get_data())
                img_yuyv = img_yuyv.view(np.uint8).reshape(H, W, 2)
                img_rgb = cv2.cvtColor(img_yuyv, cv2.COLOR_YUV2RGB_YUYV)
                with self.state_lock:
                    self.latest_images[cam_name] = img_rgb
            except Exception as e:
                if rclpy.ok():
                    self.get_logger().warn(f"[Camera] {cam_name} frame failed: {e}")

    def _start_camera_threads(self):
        for cam_name, fq in self.camera_frame_queues.items():
            t = threading.Thread(
                target=self._camera_capture_thread,
                args=(cam_name, fq),
                daemon=True,
                name=f"camera_{cam_name}",
            )
            t.start()

    def _save_face_view_video(self):
        if not self._face_view_frames:
            return
        try:
            log_dir = Path(__file__).resolve().parent / "logs"
            video_dir = log_dir / "face_view_videos"
            video_dir.mkdir(exist_ok=True)
            frames_dir = video_dir / f"face_view_{self._session_start}_frames"
            frames_dir.mkdir(exist_ok=True)
            for i, frame in enumerate(self._face_view_frames):
                cv2.imwrite(str(frames_dir / f"frame_{i:02d}.png"), frame)
            video_path = video_dir / f"face_view_{self._session_start}.mp4"
            import subprocess
            result = subprocess.run(
                ["ffmpeg", "-y", "-framerate", "5", "-i", str(frames_dir / "frame_%02d.png"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video_path)],
                capture_output=True, timeout=10, check=False
            )
            msg = f"Face view: {video_path}" if result.returncode == 0 else f"Face view frames: {frames_dir}"
            msg += f" ({len(self._face_view_frames)} frames)"
            self.get_logger().info(msg)
        except Exception as e:
            self.get_logger().warn(f"Save face view failed: {e}")

    def _save_sent_images(self, obs: dict, inference_num: int):
        log_dir = Path(__file__).resolve().parent / "logs"
        sent_dir = log_dir / "sent_to_server" / f"infer_{inference_num}"
        sent_dir.mkdir(parents=True, exist_ok=True)
        for key in ('face_view', 'left_wrist_view', 'right_wrist_view'):
            img = obs.get(key)
            if img is not None:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(sent_dir / f"{key}.png"), bgr)
        self.get_logger().info(f"Saved sent images: {sent_dir}")

    def destroy_node(self):
        self._save_face_view_video()
        for pipeline in self.camera_pipelines.values():
            try:
                pipeline.stop()
            except Exception:
                pass
        super().destroy_node()

    def _on_ee_states(self, msg: JointState):
        t = time.time()
        with self.state_lock:
            self.latest_ee_states = msg
            self.last_ee_state_time = t

    def _on_queue_status(self, msg: Bool):
        with self.queue_status_lock:
            with self.allow_inference_lock:
                if msg.data:
                    self.queue_is_empty = True
                    self.can_infer = self.allow_inference
                else:
                    self.queue_is_empty = False
                    self.can_infer = False

    def _on_allow_inference(self, msg: Bool):
        with self.queue_status_lock:
            with self.allow_inference_lock:
                self.allow_inference = msg.data
                self.can_infer = msg.data and self.queue_is_empty

    def _state_from_ee(self, ee_snapshot):
        pos = list(ee_snapshot[:3])
        rpy = list(ee_snapshot[3:6])
        gripper = 1.0 if ee_snapshot[6] <= self.args.gripper_threshold else 0.0
        return np.array(pos + rpy + [float(gripper)], dtype=np.float32)

    def _smooth_actions(self, actions: list) -> list:
        arr = np.array(actions, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        alpha = self.args.action_smoothing_alpha
        smoothed = np.zeros_like(arr)
        smoothed[0] = arr[0]
        for i in range(1, len(arr)):
            smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * arr[i]
        return [smoothed[i].tolist() for i in range(len(smoothed))]

    def get_observation(self):
        with self.state_lock:
            if any(img is None for img in self.latest_images.values()):
                return None
            images_snapshot = {k: (v.copy() if v is not None else None) for k, v in self.latest_images.items()}

        with self.state_lock:
            if self.latest_ee_states is None or len(self.latest_ee_states.position) < 7:
                return None
            ee_snapshot = copy.deepcopy(self.latest_ee_states.position)
        robot_state = self._state_from_ee(ee_snapshot)
        rpy_from_robot = robot_state[3:5]

        if self.args.use_last_action_as_state and self.all_chunks:
            last_chunk = self.all_chunks[-1]
            if len(last_chunk) > 0:
                last_action = np.asarray(last_chunk[-1]).flatten()
                if len(last_action) >= 7:
                    gripper_val = 1.0 if last_action[6] <= self.args.gripper_threshold else 0.0
                    last_action_state = np.array(
                        list(last_action[:3]) + list(rpy_from_robot) + [float(last_action[5])] + [float(gripper_val)],
                        dtype=np.float32
                    )
                    blend = np.clip(self.args.state_blend_robot, 0.0, 1.0)
                    observation_state = (1 - blend) * last_action_state + blend * robot_state
                else:
                    observation_state = robot_state
            else:
                observation_state = robot_state
        else:
            observation_state = robot_state

        return {
            "face_view": images_snapshot['face_view'],
            "left_wrist_view": images_snapshot['left_wrist_view'],
            "right_wrist_view": images_snapshot['right_wrist_view'],
            "state": observation_state,
            "prompt": self.args.task,
            "dataset_names": self.args.dataset_name,
        }

    def _inference_callback(self):
        with self.queue_status_lock:
            with self.allow_inference_lock:
                allow_inference = self.allow_inference
                can_infer_now = self.can_infer
        if not allow_inference or not can_infer_now:
            return

        obs = self.get_observation()
        if obs is None:
            return

        with self.state_lock:
            state_age = time.time() - self.last_ee_state_time if self.last_ee_state_time else 0
        if state_age > self.args.max_state_age:
            self.get_logger().warn(f"[Infer] Skip: state_age={state_age:.3f}s > {self.args.max_state_age}s")
            return

        state_sent = obs.get("state")
        if state_sent is not None:
            chunk_idx = len(self.all_chunks)
            src = f"chunk{chunk_idx} blend={self.args.state_blend_robot:.2f}" if (
                self.args.use_last_action_as_state and self.all_chunks
            ) else "robot"
            self.file_logger.info(
                f"[Infer#{chunk_idx+1}] state ({src}): "
                f"[{state_sent[0]:.4f},{state_sent[1]:.4f},{state_sent[2]:.4f},...] state_age={state_age:.3f}s"
            )

        try:
            t0 = time.time()
            result = self.policy.infer(obs)
            inference_time = time.time() - t0
            self.last_inference_time = inference_time

            actions = result['actions']
            self.inference_count += 1
            self._save_sent_images(obs, self.inference_count)

            skip_steps = 0
            if self.args.delay_steps_manual > 0:
                skip_steps = self.args.delay_steps_manual
            elif self.args.delay_steps_auto:
                action_period_s = 0.1
                skip_steps = min(int(round(inference_time / action_period_s)), len(actions) // 3)
            if skip_steps > 0:
                actions = actions[skip_steps:]

            self.get_logger().info(
                f"[Infer#{self.inference_count}] done | {inference_time*1000:.1f}ms | {len(actions)} actions"
            )

            n = self.args.max_actions_to_publish
            if self.args.use_last_actions:
                actions_to_publish = actions[-n:] if len(actions) >= n else actions
                range_desc = f"last {n}"
            else:
                actions_to_publish = actions[:n]
                range_desc = f"first {n}"
            self.get_logger().info(f"[Infer#{self.inference_count}] chunk={n} | {range_desc} | publish {len(actions_to_publish)}")

            if self.args.reverse_actions:
                actions_to_publish = list(reversed(actions_to_publish))
                self.get_logger().info(f"[Infer#{self.inference_count}] Reversed order")

            if self.args.action_smoothing_alpha > 0 and len(actions_to_publish) > 1:
                actions_to_publish = self._smooth_actions(actions_to_publish)

            self.all_chunks.append(np.array(actions_to_publish))

            log_dir = Path(__file__).resolve().parent / "logs"
            log_dir.mkdir(exist_ok=True)
            if self.args.save_action_plots and actions_to_publish:
                plot_path = log_dir / f"actions_infer_{self.inference_count}.png"
                self._plot_actions(actions_to_publish, str(plot_path), title=f"Chunk #{self.inference_count}")
                self.get_logger().info(f"Plot saved: {plot_path}")
            if self.args.save_concatenated_plot and self.all_chunks:
                concat_path = log_dir / f"actions_concatenated_{self.inference_count}.png"
                self._plot_concatenated_chunks(self.all_chunks, str(concat_path))
                self.get_logger().info(f"Concatenated plot: {concat_path}")

            with self.queue_status_lock:
                self.can_infer = False

            for i, action in enumerate(actions_to_publish):
                if len(self._face_view_frames) < self._max_face_frames:
                    with self.state_lock:
                        face_img = self.latest_images.get('face_view')
                        if face_img is not None:
                            self._face_view_frames.append(cv2.cvtColor(face_img.copy(), cv2.COLOR_RGB2BGR))

                try:
                    action_1d = np.asarray(action).flatten()
                    if len(action_1d) != 7:
                        self.get_logger().warn(f"Action dim {len(action_1d)}, expected 7")
                        continue
                    msg = Float32MultiArray()
                    msg.data = action_1d.tolist()
                    self.action_pub.publish(msg)
                except Exception as e:
                    self.get_logger().error(f"Publish action[{i}] failed: {e}")

                if i < len(actions_to_publish) - 1:
                    time.sleep(self.args.action_publish_interval)

            self.get_logger().info(f"[Infer#{self.inference_count}] Published {len(actions_to_publish)} actions")

        except Exception as e:
            self.get_logger().error(f"[Infer] Failed: {e}")
            self.file_logger.error(f"Inference failed: {e}", exc_info=True)


def main(args: Args):
    rclpy.init()
    node = WallInferenceNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nInterrupt, stopping...")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main(tyro.cli(Args))
