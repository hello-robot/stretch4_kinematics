#!/usr/bin/env python3
"""
Flying Gripper Pose Tracking Control Script for Hello Robot Stretch 4.

Runs the task-space FlyingGripperTrackingController to drive the robot's end-effector
to target Cartesian poses in real-time or simulation.

SAFETY NOTICE: This script can command high-speed physical robot motion.
Workspace clearance and operator supervision are required.
"""

import argparse
import sys
import time
import numpy as np
import pinocchio as pin

from stretch4_kinematics.state import StretchJointPositions, StretchJointVelocities
from stretch4_kinematics.controllers import FlyingGripperTrackingController


def confirm_robot_movement(target_xyz: np.ndarray) -> bool:
    print("\n" + "=" * 60)
    print("  [SAFETY WARNING] PHYSICAL ROBOT MOVEMENT REQUESTED")
    print(f"  Target Translation (x, y, z): {target_xyz} meters relative to base")
    print("  Ensure the robot workspace is completely clear of people, cables, and obstacles!")
    print("=" * 60)
    response = input("  Type 'y' or 'yes' to confirm and begin robot movement: ").strip().lower()
    return response in ("y", "yes")


def run_numerical_simulation(target_pose: pin.SE3):
    print("\n--- Running Numerical Simulation (No Robot Motion) ---")
    controller = FlyingGripperTrackingController()
    controller.reset()

    current_pos = StretchJointPositions()
    current_vel = StretchJointVelocities()

    pseudo_dt = 0.1
    n_steps = 200

    for i in range(n_steps):
        v_limited, state = controller.update(pseudo_dt, current_pos, current_vel, target_pose)
        theta = current_pos.base_theta
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        world_base_x_vel = v_limited.base_x * cos_t - v_limited.base_y * sin_t
        world_base_y_vel = v_limited.base_x * sin_t + v_limited.base_y * cos_t

        q_next = current_pos.to_numpy()
        q_next[0] += world_base_x_vel * pseudo_dt
        q_next[1] += world_base_y_vel * pseudo_dt
        q_next[2] += v_limited.base_theta * pseudo_dt
        q_next[3:] += v_limited.to_numpy()[3:] * pseudo_dt

        current_pos = StretchJointPositions.from_numpy(q_next)
        current_vel = v_limited

    print("\nSimulation Complete. Final Joint Positions:")
    current_pos.pretty_print()
    fk_pose = controller._kinematics_solver.forward(current_pos, "tool_attachment_site_link")
    print(f"Final Tool Position: {fk_pose.translation}")
    print(f"Target Tool Position: {target_pose.translation}")


def run_hardware_tracking(target_pose: pin.SE3, rate_hz: float):
    try:
        from stretch4_body.robot.robot_client import RobotClient
        from stretch4_kinematics.stretch_interface import StretchInterface
    except ImportError:
        print("\n[ERROR] stretch4_body is required for physical robot tracking.")
        print("Install with: pip install hello-robot-stretch4-body")
        sys.exit(1)

    if not confirm_robot_movement(target_pose.translation):
        print("Robot movement cancelled by user.")
        sys.exit(0)

    print("\nConnecting to robot client...")
    robot = RobotClient()
    robot.startup()
    interface = StretchInterface(robot=robot)
    controller = FlyingGripperTrackingController()

    dt = 1.0 / rate_hz
    interface.reset_odometry_offset()

    print(f"\nStarting control loop at {rate_hz} Hz. Press Ctrl+C to stop.\n")
    try:
        while True:
            t_start = time.time()
            current_pos = interface.get_joint_position()
            current_vel = interface.get_joint_velocity()

            v_cmd, state = controller.update(dt, current_pos, current_vel, target_pose)
            interface.cmd_velocities(v_cmd)

            elapsed = time.time() - t_start
            if dt > elapsed:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("\nStopping controller and robot...")
    finally:
        robot.stop()
        print("Robot stopped safely.")


def main():
    parser = argparse.ArgumentParser(
        description="Fly gripper tracking controller for Hello Robot Stretch 4.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--mode", choices=["numerical", "hardware"], default="numerical",
                        help="Run mode: 'numerical' (simulation only) or 'hardware' (physical robot)")
    parser.add_argument("--target-xyz", nargs=3, type=float, default=[0.4, 0.0, 0.5],
                        metavar=("X", "Y", "Z"), help="Target Cartesian position (m) in base frame")
    parser.add_argument("--rate", type=float, default=20.0, help="Control loop rate (Hz) for hardware mode")

    args = parser.parse_args()

    target_pose = pin.SE3.Identity()
    target_pose.translation = np.array(args.target_xyz)

    if args.mode == "numerical":
        run_numerical_simulation(target_pose)
    elif args.mode == "hardware":
        run_hardware_tracking(target_pose, rate_hz=args.rate)


if __name__ == "__main__":
    main()
