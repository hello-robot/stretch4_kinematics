import numpy as np
import time

from stretch4_body.robot.robot_client import RobotClient

from stretch4_kinematics.stretch_interface import StretchInterface
from stretch4_kinematics.kinematic_models import StretchKinematics

def main():
    # Initialize the robot interface and kinematics solver
    robot = RobotClient()
    robot.startup()
    interface = StretchInterface(robot=robot)
    solver = StretchKinematics()

    # Query the current joint positions to use as the initial guess
    # Zero out odom for correct base rotation
    current_state = interface.get_joint_position(report_zero_odom=True)

    # Solve 6-DOF Inverse Kinematics (base rotation allowed, no base translation)
    solved_pose = solver.inverse_6dof_local(
        target_frame="tool_attachment_site_link",
        target_xyz=np.array([0.4, -0.2, 0.8]),
        target_rpy=np.deg2rad([30, 30, 0]),
        q_guess=current_state
    )

    print("Solved Joint Positions:")
    solved_pose.pretty_print()

    # Command all joints to move to the solved pose simultaneously
    ui = input("Enter y to move the robot to the solved pose.\nInput: ")
    if ui == "y":
        print("Moving!")
        interface.move_to_local_pose(solved_pose)
        time.sleep(3.)
    else:
        print("Move cancelled.")
    
    # Clean up connections
    robot.stop()

if __name__ == '__main__':
    main()
