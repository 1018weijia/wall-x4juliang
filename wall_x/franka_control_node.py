#!/usr/bin/env python3
"""
Franka control node: connects via franky, subscribes to actions, publishes EE state and queue status.
Run this node first, then the inference node.
"""

import dataclasses
import math
import threading
import time
import logging
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Bool
import tyro

import franky
from franky import Robot, CartesianMotion, Affine, RelativeDynamicsFactor


@dataclasses.dataclass
class Args:
    robot_ip: str = "172.16.0.2"
    ee_states_topic: str = "/franka/ee_states"
    action_topic: str = "/franka/action_command"
    queue_status_topic: str = "/franka/queue_status"
    allow_inference_topic: str = "/franka/allow_inference"

    control_frequency: float = 10.0
    state_publish_frequency: float = 50.0
    dynamics_factor: float = 0.2
    dynamics_velocity: float = 0.0
    dynamics_acceleration: float = 0.0
    dynamics_jerk: float = 0.0
    action_buffer_size: int = 20
    wait_for_motion: bool = True
    queue_low_threshold: int = 0
    execute_stride: int = 1
    settle_delay: float = 0.25
    min_z: float = 0.03

    gripper_speed: float = 0.3
    gripper_force: float = 40.0
    gripper_sleep_after_grasp: float = 0.1


class FrankaControlNode(Node):
    def __init__(self, args: Args):
        super().__init__('franka_control_node')
        self.args = args
        self._setup_file_logging()

        self.get_logger().info(f"Connecting to Franka: {args.robot_ip}")
        self.robot = Robot(args.robot_ip)
        self.robot.recover_from_errors()

        if (args.dynamics_velocity > 0 and args.dynamics_acceleration > 0 and args.dynamics_jerk > 0):
            self.robot.relative_dynamics_factor = RelativeDynamicsFactor(
                velocity=args.dynamics_velocity,
                acceleration=args.dynamics_acceleration,
                jerk=args.dynamics_jerk,
            )
            self.get_logger().info(
                f"Robot connected (dynamics: v={args.dynamics_velocity} a={args.dynamics_acceleration} j={args.dynamics_jerk})"
            )
        else:
            self.robot.relative_dynamics_factor = args.dynamics_factor
            self.get_logger().info(f"Robot connected (dynamics_factor: {args.dynamics_factor})")

        self.get_logger().info(f"Connecting gripper: {args.robot_ip}")
        self.gripper = franky.Gripper(args.robot_ip)
        self.get_logger().info("Gripper connected")

        self.last_gripper_state = None
        self.action_queue = deque(maxlen=args.action_buffer_size)
        self.queue_lock = threading.Lock()
        self.queue_was_empty = True
        self.inference_triggered_for_chunk = False
        self.last_executed_action = None

        self.action_sub = self.create_subscription(
            Float32MultiArray, args.action_topic, self._on_action, 10
        )
        self.ee_states_pub = self.create_publisher(JointState, args.ee_states_topic, 10)
        self.queue_status_pub = self.create_publisher(Bool, args.queue_status_topic, 10)
        self.allow_inference_pub = self.create_publisher(Bool, args.allow_inference_topic, 10)

        state_period = 1.0 / args.state_publish_frequency
        self.state_timer = self.create_timer(state_period, self._publish_robot_state)

        self.control_running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()

        self.actions_received = 0
        self.actions_executed = 0

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"[Control] Started | {args.control_frequency}Hz | IP: {args.robot_ip}")
        dyn_info = (
            f"v={args.dynamics_velocity} a={args.dynamics_acceleration} j={args.dynamics_jerk}"
            if (args.dynamics_velocity > 0 and args.dynamics_acceleration > 0 and args.dynamics_jerk > 0)
            else f"dynamics={args.dynamics_factor}"
        )
        self.get_logger().info(f"[Control] min_z={args.min_z*100:.1f}cm | {dyn_info}")
        self.get_logger().info("=" * 60)

        threading.Thread(target=self._check_topic_connection, daemon=True).start()
        threading.Thread(target=self._publish_initial_status, daemon=True).start()

    def _setup_file_logging(self):
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"control_{timestamp}.log"
        self.file_logger = logging.getLogger("control_file")
        self.file_logger.setLevel(logging.INFO)
        if not self.file_logger.handlers:
            handler = logging.FileHandler(log_file, encoding='utf-8')
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
            ))
            self.file_logger.addHandler(handler)
        self.get_logger().info(f"Log file: {log_file}")

    def _publish_bool(self, pub, value: bool):
        msg = Bool()
        msg.data = value
        pub.publish(msg)

    def _check_topic_connection(self):
        time.sleep(2.0)
        sub_count = self.ee_states_pub.get_subscription_count()
        try:
            pub_count = self.action_sub.get_publisher_count()
        except AttributeError:
            pub_count = -1
        self.get_logger().info(f"Topics: EE subscribers={sub_count}, action publishers={pub_count}")
        if pub_count == 0:
            self.get_logger().warn(f"No action publisher on {self.args.action_topic}")

    def _publish_initial_status(self):
        time.sleep(3.0)
        with self.queue_lock:
            if len(self.action_queue) == 0:
                self.get_logger().info("[Start] Publishing initial ready signal")
                for _ in range(3):
                    self._publish_bool(self.queue_status_pub, True)
                    self._publish_bool(self.allow_inference_pub, True)
                    time.sleep(0.1)
                self.get_logger().info("[Start] Initial allow_inference sent (x3)")

    def _on_action(self, msg: Float32MultiArray):
        try:
            action = np.array(msg.data, dtype=np.float32)
            if len(action) != 7:
                self.get_logger().error(f"Invalid action dim: {len(action)}, expected 7")
                return
            with self.queue_lock:
                was_empty = len(self.action_queue) == 0
                self.actions_received += 1
                if self.args.execute_stride > 1:
                    if (self.actions_received - 1) % self.args.execute_stride != 0:
                        return
                self.action_queue.append(action)
                if was_empty:
                    self.queue_was_empty = False
                    self.inference_triggered_for_chunk = False
                    self._publish_bool(self.allow_inference_pub, False)
                    self.get_logger().info("[Inference] Disabled")
                if self.actions_received == 1:
                    self.get_logger().info(f"[Action] First received | queue: {len(self.action_queue)}")
                elif self.actions_received % 10 == 0:
                    self.get_logger().info(f"[Action] #{self.actions_received} | queue: {len(self.action_queue)}")
        except Exception as e:
            self.get_logger().error(f"Action callback failed: {e}", exc_info=True)

    @staticmethod
    def _quat_to_rpy(qx, qy, qz, qw):
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    @staticmethod
    def _rpy_to_quat(roll, pitch, yaw):
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return qx, qy, qz, qw

    def _publish_robot_state(self):
        try:
            cartesian_state = self.robot.current_cartesian_state
            ee_pose = cartesian_state.pose.end_effector_pose
            try:
                pos = ee_pose.translation
                ee_pos = [float(pos[0]), float(pos[1]), float(pos[2])]
                q = ee_pose.quaternion
                qx, qy, qz, qw = [float(x) for x in q]
                ee_rpy = list(self._quat_to_rpy(qx, qy, qz, qw))
            except (IndexError, TypeError, AttributeError, Exception) as e:
                self.get_logger().error(f"EE state read failed: {e}", exc_info=True)
                return
            try:
                gripper_width = float(self.gripper.width)
            except Exception:
                gripper_width = 0.0
            try:
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "franka_link0"
                msg.name = [
                    "ee_x", "ee_y", "ee_z", "ee_roll", "ee_pitch", "ee_yaw", "gripper_width"
                ]
                msg.position = ee_pos + ee_rpy + [gripper_width]
                self.ee_states_pub.publish(msg)
            except Exception as e:
                self.get_logger().error(f"Publish state failed: {e}", exc_info=True)
        except Exception as e:
            self.get_logger().error(f"Publish EE state failed: {e}", exc_info=True)

    def _execute_action(self, action: np.ndarray):
        try:
            t0 = time.time()
            action_num = self.actions_executed + 1

            target_pos = action[:3].copy()
            current_pose = self.robot.current_cartesian_state.pose.end_effector_pose
            qx, qy, qz, qw = current_pose.quaternion
            current_roll, current_pitch, current_yaw = self._quat_to_rpy(qx, qy, qz, qw)
            target_yaw = action[5]
            target_rpy = np.array([current_roll, current_pitch, target_yaw], dtype=np.float32)

            z_limited = False
            if target_pos[2] < self.args.min_z:
                orig_z = target_pos[2]
                target_pos[2] = self.args.min_z
                z_limited = True
                warn = f"Z clamped: {orig_z:.4f}m -> {self.args.min_z:.4f}m"
                self.get_logger().warn(warn)
                self.file_logger.warning(warn)

            qx, qy, qz, qw = self._rpy_to_quat(target_rpy[0], target_rpy[1], target_rpy[2])
            gripper_value = action[6]
            gripper_state = 0 if gripper_value < 0.5 else 1

            self.get_logger().info(
                f"[Exec#{action_num}] pos=[{target_pos[0]:.4f},{target_pos[1]:.4f},{target_pos[2]:.4f}]m "
                f"rpy=[{target_rpy[0]:.4f},{target_rpy[1]:.4f},{target_rpy[2]:.4f}] gripper={gripper_state}"
            )
            self.file_logger.info(
                f"[Exec#{action_num}] raw={action.tolist()} pos={target_pos.tolist()} "
                f"rpy={target_rpy.tolist()} gripper={gripper_state} z_limited={z_limited}"
            )

            target_affine = Affine(target_pos.tolist(), [qx, qy, qz, qw])
            motion = CartesianMotion(target_affine)
            t_motion = time.time()
            if self.args.wait_for_motion:
                self.robot.move(motion, asynchronous=False)
            else:
                self.robot.move(motion, asynchronous=True)
            motion_ms = (time.time() - t_motion) * 1000

            gripper_ms = 0.0
            if gripper_state != self.last_gripper_state:
                try:
                    t_grip = time.time()
                    if gripper_state == 0:
                        self.get_logger().info(f"[Exec#{action_num}] Gripper open")
                        self.gripper.open(speed=self.args.gripper_speed)
                    else:
                        self.get_logger().info(f"[Exec#{action_num}] Gripper grasp...")
                        success = self.gripper.grasp(
                            width=0.0,
                            speed=self.args.gripper_speed,
                            force=self.args.gripper_force,
                            epsilon_outer=0.05
                        )
                        if success:
                            self.get_logger().info(f"[Exec#{action_num}] Grasp ok width={self.gripper.width*1000:.1f}mm")
                        else:
                            self.get_logger().warn(f"[Exec#{action_num}] Grasp failed")
                        time.sleep(self.args.gripper_sleep_after_grasp)
                    gripper_ms = (time.time() - t_grip) * 1000
                    self.last_gripper_state = gripper_state
                except Exception as e:
                    self.get_logger().error(f"[Exec#{action_num}] Gripper error: {e}")
                    self.file_logger.error(f"[Exec#{action_num}] Gripper: {e}", exc_info=True)

            self.actions_executed += 1
            self.last_executed_action = action.copy()
            total_ms = (time.time() - t0) * 1000
            self.get_logger().info(
                f"[Exec#{action_num}] done | motion={motion_ms:.1f}ms gripper={gripper_ms:.1f}ms total={total_ms:.1f}ms"
            )
            self.file_logger.info(
                f"[Exec#{action_num}] done motion={motion_ms:.1f}ms total={total_ms:.1f}ms"
            )
            if self.actions_executed % 10 == 0:
                self.get_logger().info(f"[Exec] Count: {self.actions_executed} executed, {self.actions_received} received")

        except Exception as e:
            self.get_logger().error(f"Execute failed: {e}")
            self.file_logger.error(f"Execute failed: {e}", exc_info=True)
            if "Reflex" in str(e):
                self.get_logger().warn("[Recovery] Reflex detected, recovering...")
                try:
                    self.robot.recover_from_errors()
                    self.get_logger().info("[Recovery] OK")
                except Exception as err:
                    self.get_logger().error(f"[Recovery] Failed: {err}")
                    self.file_logger.error(f"Recovery failed: {err}", exc_info=True)

    def _control_loop(self):
        period = 1.0 / self.args.control_frequency
        empty_loops = 0

        while self.control_running:
            loop_start = time.time()
            action = None
            queue_len = 0
            with self.queue_lock:
                queue_len = len(self.action_queue)
                if queue_len > 0:
                    action = self.action_queue.popleft()

            if action is not None:
                empty_loops = 0
                was_empty_before = self.queue_was_empty
                self._execute_action(action)

                with self.queue_lock:
                    queue_len_after = len(self.action_queue)
                    queue_empty = queue_len_after == 0
                    self.queue_was_empty = queue_empty

                if (self.args.queue_low_threshold > 0
                        and not queue_empty
                        and queue_len_after <= self.args.queue_low_threshold
                        and not self.inference_triggered_for_chunk):
                    self.inference_triggered_for_chunk = True
                    self._publish_bool(self.queue_status_pub, True)
                    self._publish_bool(self.allow_inference_pub, True)
                    self.get_logger().info(
                        f"[Pre-infer] Triggered | queue={queue_len_after} <= {self.args.queue_low_threshold}"
                    )

                if queue_empty and not was_empty_before:
                    self.get_logger().info("[Queue] Empty, waiting for motion...")
                    max_wait = 5.0
                    check_interval = 0.05
                    t_start = time.time()
                    while self.robot.is_in_control:
                        if time.time() - t_start > max_wait:
                            self.get_logger().warn(f"[Queue] Wait timeout ({max_wait}s)")
                            break
                        time.sleep(check_interval)
                    waited = time.time() - t_start
                    self.get_logger().info(f"[Queue] Motion done | waited {waited:.2f}s")
                    if self.args.settle_delay > 0:
                        time.sleep(self.args.settle_delay)
                    self.inference_triggered_for_chunk = False
                    self._publish_bool(self.queue_status_pub, True)
                    self._publish_bool(self.allow_inference_pub, True)
                    self.get_logger().info("[Queue] Ready + allow_inference published")
            else:
                empty_loops += 1
                if empty_loops % 10 == 0:
                    if self.queue_was_empty:
                        self._publish_bool(self.queue_status_pub, True)
                        self._publish_bool(self.allow_inference_pub, True)
                    else:
                        self.queue_was_empty = True
                        self.get_logger().info("[Control] Queue empty, publishing ready...")
                        self._publish_bool(self.queue_status_pub, True)
                        self._publish_bool(self.allow_inference_pub, True)
                        self.get_logger().info("[Control] Ready published")
                if empty_loops % 50 == 0:
                    self.get_logger().warn(
                        f"[Control] Queue empty | received={self.actions_received} executed={self.actions_executed}"
                    )

            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

    def shutdown(self):
        self.get_logger().info("Stopping...")
        self.file_logger.info(f"Control node stopped, executed {self.actions_executed} actions")
        self.control_running = False
        if self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
        self.robot.stop()
        self.get_logger().info("Stopped")


def main(args: Args):
    rclpy.init()
    node = FrankaControlNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nInterrupt, stopping...")
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main(tyro.cli(Args))
