import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin

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
    
    def test_numerical_run(self):
        target_pose = pin.SE3.Identity()
        target_pose.translation = np.array([0.5, 0.5, 0.5])
        target_pose.rotation = np.eye(3)
        
        current_pos = StretchJointPositions()
        current_vel = StretchJointVelocities()

        pseudo_dt = 0.1
        n_steps = 100
        fk_history = np.zeros((n_steps, 3))
        
        for i in range(n_steps):
            v_limited, state = self.controller.update(pseudo_dt, current_pos, current_vel, target_pose)
            q_next = current_pos.to_numpy() + v_limited.to_numpy() * pseudo_dt
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

    def test_move_to_position(self):
        rate_hz = 20.0
        dt = 1.0 / rate_hz

        while True:
            try:
                # top_time = time.time()

                current_pos = self.robot_interface.get_joint_position()
                current_pos.pretty_print()

                break

            except KeyboardInterrupt:
                break

        self.robot_interface.shutdown()

if __name__ == '__main__':
    script = FlyingGripperScript()
    script.test_move_to_position()
