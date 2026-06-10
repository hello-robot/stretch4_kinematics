import copy
import matplotlib.pyplot as plt
import numpy as np

from stretch4_kinematics.kinematic_models import (
    StretchJointPositions,
    ToolFrameKinematics,
)

def test_toolframe_kinematics():    
    kinematics = ToolFrameKinematics()
    q_init = StretchJointPositions()
    # Set a realistic nominal starting configuration so we're not starting on limits
    q_init.lift = 0.5
    q_init.arm = 0.25
    joint_names = q_init.get_joint_names()

    # number of samples for each sweep
    n_samples = 100

    # 6D desired twist inputs: one component at a time
    velocities = 0.1 * np.eye(6)
    twist_names = ["V_x", "V_y", "V_z", "W_x", "W_y", "W_z"]

    fig, axes = plt.subplots(6, 8, figsize=(24, 18), sharex="col", sharey="row")

    # # only test Vxyz
    # velocities = velocities[:3, :]
    # twist_names = twist_names[:3]
    # fig, axes = plt.subplots(3, 8, figsize=(30, 12), sharex="col", sharey="row")

    for i, v in enumerate(velocities):
        for j, joint_name in enumerate(joint_names):
            ax = axes[i, j]
            
            # Determine joint range to sample based on model limits
            if joint_name == "base_x":
                lim_lower, lim_upper = -1.0, 1.0
            elif joint_name == "base_y":
                lim_lower, lim_upper = -1.0, 1.0
            elif joint_name == "base_theta":
                lim_lower, lim_upper = -np.pi, np.pi
            else:
                pin_name = {
                    "lift": "lift_joint",
                    "arm": "arm_l4_joint",
                    "wrist_yaw": "wrist_yaw_joint",
                    "wrist_pitch": "wrist_pitch_joint",
                    "wrist_roll": "wrist_roll_joint"
                }[joint_name]
                j_id = kinematics.model.getJointId(pin_name)
                idx_q = kinematics.model.joints[j_id].idx_q
                lim_lower = kinematics.model.lowerPositionLimit[idx_q]
                lim_upper = kinematics.model.upperPositionLimit[idx_q]
                
                # Check for infinite limits
                if lim_lower < -1e3:
                    lim_lower = -np.pi
                if lim_upper > 1e3:
                    lim_upper = np.pi
                
                # Extend arm limit slightly to observe the recovery/blending behavior
                if joint_name == "arm":
                    lim_upper = max(lim_upper, 0.55)

            samples = np.linspace(lim_lower, lim_upper, n_samples)
            vel_outputs = np.zeros((n_samples, 8))

            for k, val in enumerate(samples):
                q_temp = copy.deepcopy(q_init)
                setattr(q_temp, joint_name, val)
                
                # Compute IK
                vel_outputs[k, :] = kinematics.differential_ik(
                    q_temp,
                    "tool_attachment_site_link",
                    v
                ).to_numpy()

            # Plot all 8 joint velocities
            for joint_idx, joint_vels in enumerate(vel_outputs.T):
                ax.plot(samples, joint_vels, alpha=0.8)

            ax.grid(True, linestyle="--", alpha=0.5)
            
            # Setup titles & labels to keep the subplots clean
            if i == 0:
                ax.set_title(f"Sweep {joint_name}", fontsize=16, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"Twist {twist_names[i]}\n(m/s or rad/s)", fontsize=16)
            if i == len(velocities) - 1:
                ax.set_xlabel(f"{joint_name} pos", fontsize=16)

    # Place a single legend for all joint velocity lines
    fig.legend(
        joint_names,
        loc="center right",
        # title="Joint Velocities",
        fontsize=18
    )
    fig.suptitle(
        "Differential IK Joint Velocities over Joint Position Sweeps",
        fontsize=30,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    # Save the plot for inspection
    plt.savefig("differential_kinematics_profile.png", dpi=300)
    print("Saved profiling plot to differential_kinematics_profile.png")
    plt.show()
    

def main():
    test_toolframe_kinematics()

if __name__ == "__main__":
    main()
