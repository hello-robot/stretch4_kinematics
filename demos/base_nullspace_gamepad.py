#!/usr/bin/env python3
"""
Base Nullspace Gamepad Teleoperation Demo.

Translates the Stretch 4 mobile base in the ground plane using joystick input
while keeping the end-effector stationary via nullspace projection.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import numpy as np
import pinocchio as pin

try:
    from stretch4_kinematics.demos.gamepad_mapper import GamepadMapper
except ModuleNotFoundError:
    from gamepad_mapper import GamepadMapper
from stretch4_kinematics.kinematic_models.tool_frame_kinematics import ToolFrameKinematics
from stretch4_kinematics.state import (
    StretchJointPositions,
    StretchJointVelocities,
)


def run_base_nullspace_teleop(
    numerical: bool = False,
    speed_scale: float = 0.15,
    rate_hz: float = 50.0,
    kp_ee: float = 2.0,
) -> None:
    """
    Run gamepad teleoperation driving base translation in the ground plane.

    Uses nullspace projection from ToolFrameKinematics to keep the end-effector
    stationary in Cartesian space while the mobile base translates.

    Args:
        numerical: If True, runs in simulation mode without hardware.
        speed_scale: Maximum translational velocity scale in meters per second.
        rate_hz: Control loop frequency in Hertz.
        kp_ee: Proportional feedback gain for end-effector drift correction.
    """
    target_frame = "tool_attachment_site_link"
    dt = 1.0 / rate_hz

    print("Initializing Gamepad Mapper...")
    try:
        gamepad = GamepadMapper()
    except Exception as exc:
        print(f"Failed to initialize gamepad: {exc}")
        sys.exit(1)

    kinematics = ToolFrameKinematics()

    robot = None
    interface = None
    if numerical:
        print("Running in NUMERICAL simulation mode.")
        current_pos = StretchJointPositions(
            base_x=0.0,
            base_y=0.0,
            base_theta=0.0,
            lift=0.5,
            arm=0.25,
            wrist_yaw=0.0,
            wrist_pitch=0.0,
            wrist_roll=0.0,
        )
    else:
        print("Connecting to robot client...")
        from stretch4_body.robot.robot_client import RobotClient
        from stretch4_kinematics.stretch_interface import StretchInterface

        robot = RobotClient()
        if not robot.startup():
            print("Failed to start robot client connection.")
            gamepad.stop()
            sys.exit(1)

        interface = StretchInterface(robot=robot)
        interface.reset_odometry_offset()
        current_pos = interface.get_joint_position()

    initial_ee_pose = kinematics.forward(current_pos, target_frame)
    last_pushed_was_zero = False

    print("==================================================")
    print("Stretch 4 Base Nullspace Teleop Started")
    print("Left Stick: Translate base in ground plane")
    print("End-Effector: Kept fixed in space via nullspace control")
    print("Ctrl+C to Quit")
    print("==================================================")

    rate = time.time()
    try:
        while True:
            raw_cmd = gamepad.get_commands()

            if not numerical and interface is not None:
                current_pos = interface.get_joint_position()

            if not raw_cmd:
                if not last_pushed_was_zero:
                    if not numerical and interface is not None:
                        interface.cmd_zero_velocity()
                    last_pushed_was_zero = True
                time.sleep(dt)
                continue

            # Left stick: v_desired[0] is forward/back (+/- X), v_desired[1] is left/right (+/- Y)
            v_desired_stick = raw_cmd.get("v_desired", [0.0, 0.0, 0.0])
            left_trigger = raw_cmd.get("left_trigger", 0.0)
            speed_mult = 1.0 - (0.5 * left_trigger)

            v_world_x = float(v_desired_stick[0]) * speed_scale * speed_mult
            v_world_y = float(v_desired_stick[1]) * speed_scale * speed_mult
            v_base_world_des = np.array([v_world_x, v_world_y])

            is_active = (abs(v_world_x) > 1e-4 or abs(v_world_y) > 1e-4)

            if is_active:
                current_ee_pose = kinematics.forward(current_pos, target_frame)
                ee_err_local = pin.log(current_ee_pose.actInv(initial_ee_pose)).vector
                v_ee_local_des = kp_ee * ee_err_local

                v_cmd = kinematics.nullspace_base_control(
                    q=current_pos,
                    v_ee_local_des=v_ee_local_des,
                    v_base_world_des=v_base_world_des,
                )

                if numerical:
                    theta = current_pos.base_theta
                    cos_t = np.cos(theta)
                    sin_t = np.sin(theta)
                    world_vx = v_cmd.base_x * cos_t - v_cmd.base_y * sin_t
                    world_vy = v_cmd.base_x * sin_t + v_cmd.base_y * cos_t

                    q_next = current_pos.to_numpy()
                    q_next[0] += world_vx * dt
                    q_next[1] += world_vy * dt
                    q_next[2] += v_cmd.base_theta * dt
                    q_next[3:] += v_cmd.to_numpy()[3:] * dt
                    current_pos = StretchJointPositions.from_numpy(q_next)

                    print(
                        f"Sim Base XY: [{current_pos.base_x:.3f}, {current_pos.base_y:.3f}] | "
                        f"Arm: {current_pos.arm:.3f}m | Lift: {current_pos.lift:.3f}m | "
                        f"EE Error: {np.linalg.norm(ee_err_local[:3]):.4f}m",
                        end="\r",
                    )
                else:
                    if interface is not None:
                        interface.cmd_velocities(v_cmd)

                last_pushed_was_zero = False
            elif not last_pushed_was_zero:
                if not numerical and interface is not None:
                    interface.cmd_zero_velocity()
                # Update initial_ee_pose to prevent stationary integrator drift
                initial_ee_pose = kinematics.forward(current_pos, target_frame)
                last_pushed_was_zero = True

            sleep_time = rate + dt - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            rate = time.time()

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as exc:
        print(f"\nError in teleoperation loop: {exc}")
        traceback.print_exc()
    finally:
        print("Stopping teleoperation...")
        gamepad.stop()
        if not numerical and interface is not None and robot is not None:
            try:
                interface.cmd_zero_velocity()
                robot.stop()
            except Exception as shutdown_err:
                print(f"Error during shutdown: {shutdown_err}")
        print("Shutdown complete.")


def main() -> None:
    """
    Entry point for base nullspace gamepad teleoperation demo.
    """
    parser = argparse.ArgumentParser(
        description="Base translation gamepad teleop keeping EE still with nullspace control."
    )
    parser.add_argument(
        "--numerical",
        "-n",
        action="store_true",
        help="Run numerically in simulation mode without hardware",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.15,
        help="Maximum translational base speed in m/s (default: 0.15)",
    )
    args = parser.parse_args()

    run_base_nullspace_teleop(
        numerical=args.numerical,
        speed_scale=args.speed,
    )


if __name__ == "__main__":
    main()
