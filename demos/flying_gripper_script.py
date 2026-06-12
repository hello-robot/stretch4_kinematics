import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import time

from stretch4_kinematics.stretch_interface import StretchInterface

from stretch4_kinematics.kinematic_models import (
    StretchJointPositions,
    StretchJointVelocities,
)
from stretch4_kinematics.controllers import (
    FlyingGripperController,
    FlyingGripperState,
)

class FlyingGripperScript:
    def __init__(self):
        self.robot_interface = StretchInterface()
        self.controller = FlyingGripperController()
    
    def test_numerical_run(self, target_pose: pin.SE3):
        """
        Tests the controller numerically without the robot.
        Plots the end effector position relative to the target pose.

        Args:
            target_pose (pin.SE3): The target pose as a SE3Pose object.
        """
        self.controller.reset()
        self.robot_interface.reset_odometry_offset()
        current_pos = self.robot_interface.get_joint_position()
        current_vel = self.robot_interface.get_joint_velocity()

        pseudo_dt = 0.1
        n_steps = 200
        fk_history = np.zeros((n_steps, 3))
        
        for i in range(n_steps):
            v_limited, state = self.controller.update(pseudo_dt, current_pos, current_vel, target_pose)
            
            # Rotate local base velocities to world frame for integration
            theta = current_pos.base_theta
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            world_base_x_vel = v_limited.base_x * cos_t - v_limited.base_y * sin_t
            world_base_y_vel = v_limited.base_x * sin_t + v_limited.base_y * cos_t

            q_next = current_pos.to_numpy()
            q_next[0] += world_base_x_vel * pseudo_dt
            q_next[1] += world_base_y_vel * pseudo_dt
            q_next[2] += v_limited.base_theta * pseudo_dt
            q_next[3:] += v_limited.to_numpy()[3:] * pseudo_dt

            current_pos = StretchJointPositions.from_numpy(q_next)
            current_vel = v_limited

            fk_history[i] = self.controller._kinematics_solver.forward(current_pos, "tool_attachment_site_link").translation

        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
        ylabels = ["FK - X (m)", "FK - Y (m)", "FK - Z (m)"]
        for i in range(3):
            axs[i].plot(np.arange(n_steps)*pseudo_dt, fk_history[:  , i], "b-")
            axs[i].plot(np.arange(n_steps)*pseudo_dt, np.ones(n_steps)*target_pose.translation[i], "k--")
            axs[i].set_ylabel(ylabels[i])
            axs[i].set_xlabel("Time (s)")
            axs[i].grid(True)
            axs[i].set_axisbelow(True)
        
        plt.suptitle(f"Test Controller Convergence to Target Pose {target_pose.translation}")
        # legend below all axes
        plt.legend(["end effector pose", "target pose"], loc="upper center", bbox_to_anchor=(0.5, 3.7), ncol=2)
        plt.tight_layout()
        
        plt.show()

        print("\nFinal Joint Positions:")
        current_pos.pretty_print()
        
        current_pose = self.controller._kinematics_solver.forward(current_pos, "tool_attachment_site_link")
        print(f"\nFinal Tool Pose Translation: {current_pose.translation}")
        print(f"Target Pose Translation:      {target_pose.translation}")

    def test_move_to_position(self, target_pose: pin.SE3):
        """
        Tests the controller on the real robot. Moves the arm to the target pose relative to the robot's intial base pose.

        Args:
            target_pose (pin.SE3): The target pose as a SE3Pose object.
        """
        self.controller.reset()
        current_pos = StretchJointPositions()
        current_vel = StretchJointVelocities()

        rate_hz = 20.0
        dt = 1.0 / rate_hz

        self.robot_interface.reset_odometry_offset()

        while True:
            try:
                top_time = time.time()

                current_pos = self.robot_interface.get_joint_position()
                # current_pos.pretty_print()

                current_vel = self.robot_interface.get_joint_velocity()
                # current_vel.pretty_print()

                v_limited, state = self.controller.update(dt, current_pos, current_vel, target_pose)
                print(f"Next Velocity: ")
                v_limited.pretty_print()

                GAIN = 0.5
                LOOKAHEAD = 2.
                v_limited.base_x *= GAIN
                v_limited.base_y *= GAIN
                v_limited.base_theta *= GAIN
                v_limited.lift *= GAIN
                v_limited.arm *= GAIN
                v_limited.wrist_yaw *= GAIN * LOOKAHEAD
                v_limited.wrist_pitch *= GAIN * LOOKAHEAD
                v_limited.wrist_roll *= GAIN * LOOKAHEAD

                self.robot_interface.cmd_velocities(v_limited)

                bottom = time.time()
                time.sleep(dt - (bottom - top_time))

            except KeyboardInterrupt:
                break

        self.robot_interface.shutdown()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--numerical', '-n', action='store_true')
    parser.add_argument('--move', '-m', action='store_true')
    args = parser.parse_args()

    target_pose = pin.SE3.Identity()
    target_pose.translation = np.array([1.1, -0.1, 1.0])
    target_pose.rotation = np.eye(3)

    script = FlyingGripperScript()

    if args.numerical:
        script.test_numerical_run(target_pose)
    if args.move:
        script.test_move_to_position(target_pose)

