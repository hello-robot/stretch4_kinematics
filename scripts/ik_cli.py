#!/usr/bin/env python3
"""
Inverse Kinematics Interactive CLI Script for Hello Robot Stretch 4.

Computes 6-DOF numerical inverse kinematics solutions for target Cartesian poses
and provides an option to dispatch commands to the physical robot.

SAFETY NOTICE: Movement dispatches physical hardware motion.
Verify workspace clearance before confirming robot movement.
"""

import argparse
import sys
import time
import numpy as np

from stretch4_kinematics.kinematic_models import StretchKinematics
from stretch4_kinematics.state import StretchJointPositions


def confirm_robot_movement(ik_solution: StretchJointPositions) -> bool:
    print("\n" + "=" * 60)
    print("  [SAFETY WARNING] PHYSICAL ROBOT MOVEMENT REQUESTED")
    print("  Target Joint Solution:")
    ik_solution.pretty_print()
    print("  Ensure the robot workspace is completely clear of people, cables, and obstacles!")
    print("=" * 60)
    response = input("  Type 'y' or 'yes' to confirm and move robot: ").strip().lower()
    return response in ("y", "yes")


def prompt_float(prompt_text: str, default_val: float) -> float:
    while True:
        raw_val = input(f"{prompt_text} [{default_val}]: ").strip()
        if not raw_val:
            return default_val
        try:
            return float(raw_val)
        except ValueError:
            print("Invalid input. Please enter a valid number.")


def main():
    parser = argparse.ArgumentParser(
        description="Solve inverse kinematics for Stretch 4 target pose and optionally move physical robot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--xyz", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"),
                        help="Target position (m) relative to base frame")
    parser.add_argument("--rpy", nargs=3, type=float, default=None, metavar=("ROLL", "PITCH", "YAW"),
                        help="Target orientation RPY (rad) relative to base frame")
    parser.add_argument("--target-frame", type=str, default="grasp_center_link",
                        help="Kinematic target frame name in URDF (default: grasp_center_link)")
    parser.add_argument("--move", action="store_true", default=False,
                        help="Prompt to dispatch motion to physical robot via stretch4_body (default: False)")

    args = parser.parse_args()

    solver = StretchKinematics()
    robot = None
    interface = None

    if args.move:
        try:
            from stretch4_body.robot.robot_client import RobotClient
            from stretch4_kinematics.stretch_interface import StretchInterface
            print("\nConnecting to robot client...")
            robot = RobotClient()
            robot.startup()
            interface = StretchInterface(robot=robot)
            q_guess = interface.get_joint_position().to_pinocchio_q()
        except Exception as e:
            print(f"Could not connect to robot: {e}")
            print("Falling back to neutral joint configuration guess.")
            q_guess = StretchJointPositions().to_pinocchio_q()
    else:
        q_guess = StretchJointPositions().to_pinocchio_q()

    if args.xyz is not None:
        target_xyz = np.array(args.xyz)
    else:
        print("\nEnter target position (X, Y, Z) in meters:")
        target_xyz = np.array([
            prompt_float("  X (m)", 0.8),
            prompt_float("  Y (m)", 0.0),
            prompt_float("  Z (m)", 0.5),
        ])

    if args.rpy is not None:
        target_rpy = np.array(args.rpy)
    else:
        print("\nEnter target orientation (Roll, Pitch, Yaw) in radians:")
        target_rpy = np.array([
            prompt_float("  Roll (rad)", 0.0),
            prompt_float("  Pitch (rad)", 0.0),
            prompt_float("  Yaw (rad)", 0.0),
        ])

    print(f"\nSolving 6-DOF IK for frame '{args.target_frame}'...")
    try:
        ik_solution = solver.inverse_6dof_local(
            target_frame=args.target_frame,
            target_xyz=target_xyz,
            target_rpy=target_rpy,
            q_guess=q_guess
        )
    except ValueError as err:
        radial_dist = np.linalg.norm(target_xyz[:2])
        print(f"\n[IK SOLVER FAILURE] {err}")
        print(f"Requested target position: {target_xyz} (radial distance R = {radial_dist:.3f} m)")
        print(f"Note: '{args.target_frame}' has physical reach boundaries:")
        print("  - 'tool_attachment_site_link': reach R is ~0.27m (retracted) to ~0.77m (extended)")
        print("  - 'grasp_center_link': reach R is ~0.51m (retracted) to ~1.01m (extended)")
        print("Targets with R < minimum reach (or Z out of range) cannot be reached in stationary base mode.")
        sys.exit(1)

    print("\nSolved Joint State:")
    ik_solution.pretty_print()

    if args.move and interface is not None and robot is not None:
        if confirm_robot_movement(ik_solution):
            print("Dispatching goal pose to robot...")
            interface.move_to_local_pose(ik_solution)
            time.sleep(3.0)
            robot.stop()
            print("Movement complete.")
        else:
            print("Robot movement cancelled.")


if __name__ == "__main__":
    main()
