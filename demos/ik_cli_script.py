#!/usr/bin/env python3
import numpy as np
import pinocchio as pin
import time

from stretch4_kinematics.kinematic_models import StretchKinematics, StretchJointPositions
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
    try:
        interface = StretchInterface()
        interface.reset_odometry_offset()
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

    # 4. Construct the target pose as a pinocchio.SE3 object
    R = pin.rpy.rpyToMatrix(r, p, yaw)
    t = np.array([x, y, z])
    target_pose = pin.SE3(R, t)

    # 5. Solve using kinematic_models.StretchKinematics.inverse()
    target_frame = "grasp_center_link"
    q_guess = current_joint_state.to_pinocchio_q()
    
    print(f"\nSolving IK for target pose relative to base...")
    ik_solution = solver.inverse_6dof(
        target_frame=target_frame,
        target_pose=target_pose,
        q_guess=q_guess
    )

    # 6. Pretty print the solved joint state
    print("\nSolved Joint State:")
    ik_solution.pretty_print()

    # 7. Prompt to confirm move
    confirm = input("\nEnter y to confirm move to joint state: ").strip().lower()
    if confirm == 'y':
        interface.reset_odometry_offset()
        interface.move_to_pose(ik_solution)
        time.sleep(3.)
    else:
        print("\nMove cancelled.")

    # Compute and print error
    print("\n=== Error Analysis ===")
    post_move_joint_state = interface.get_joint_position()
    post_move_pose = solver.forward(
        post_move_joint_state,
        target_frame
    )
    error = pin.log(post_move_pose.actInv(target_pose)).vector
    print(f"Position error (m): {error[:3]}")
    print(f"Rotational error (rad): {error[3:]}")

    # Clean up interface if connected
    if 'interface' in locals():
        interface.shutdown()

if __name__ == '__main__':
    main()
