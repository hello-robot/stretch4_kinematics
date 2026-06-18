#!/usr/bin/env python3
import numpy as np
import time

from stretch4_body.robot.robot_client import RobotClient

from stretch4_kinematics.kinematic_models import StretchKinematics
from stretch4_kinematics.state import StretchJointPositions
from stretch4_kinematics.stretch_interface import StretchInterface

def prompt_float(name: str) -> float:
    """Prompts the user for a float value with input validation."""
    while True:
        try:
            val = input(f"Enter target {name}: ")
            return float(val)
        except ValueError:
            print("Invalid input. Please enter a valid float number.")

def main():
    print("=== Stretch 4 Inverse Kinematics CLI ===")
    
    # 1. Initialize kinematics solver
    # StretchKinematics computes the numerical IK using the robot URDF model.
    solver = StretchKinematics()
    
    # 2. Try to connect to the robot interface to get the current joint positions as a guess
    print("\nAttempting to connect to Stretch interface to get initial joint state guess...")
    robot = None
    interface = None
    try:
        robot = RobotClient()
        robot.startup()
        interface = StretchInterface(robot=robot)
        current_joint_state = interface.get_joint_position()
        print("Successfully connected to robot. Using current joint state as guess.")
    except Exception as e:
        print(f"Could not connect to robot: {e}")
        print("Falling back to default neutral joint configuration.")
        current_joint_state = StretchJointPositions()

    # 3. Prompt the user for target X, Y, Z, R, P, Y individually
    print("\nPlease specify the target pose with respect to the robot's base frame:")
    x = prompt_float("X (m)")
    y = prompt_float("Y (m)")
    z = prompt_float("Z (m)")
    r = prompt_float("R (roll in rad)")
    p = prompt_float("P (pitch in rad)")
    yaw = prompt_float("Y (yaw in rad)")

    # 4. Solve using kinematic_models.StretchKinematics.inverse()
    target_frame = "grasp_center_link"
    q_guess = current_joint_state.to_pinocchio_q()
    
    print(f"\nSolving IK for target pose relative to base...")
    ik_solution = solver.inverse_6dof_local(
        target_frame=target_frame,
        target_xyz=np.array([x, y, z]),
        target_rpy=np.array([r, p, yaw]),
        q_guess=q_guess
    )

    # 5. Pretty print the solved joint state
    print("\nSolved Joint State:")
    ik_solution.pretty_print()

    # 6. Prompt to confirm move
    confirm = input("\nEnter y to confirm move to joint state: ").strip().lower()
    if confirm == 'y' and interface is not None and robot is not None:
        interface.move_to_local_pose(ik_solution)
        time.sleep(3.)
        robot.stop()
    else:
        print("\nMove cancelled.")

if __name__ == '__main__':
    main()
