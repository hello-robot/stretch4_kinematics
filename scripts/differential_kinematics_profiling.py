#!/usr/bin/env python3
"""
Differential Kinematics Joint Sweep Profiler for Hello Robot Stretch 4.

Sweeps joint position ranges across task-space velocity inputs and plots joint velocity
outputs computed by ToolFrameKinematics differential IK.
"""

import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np

from stretch4_kinematics.state import StretchJointPositions
from stretch4_kinematics.kinematic_models import ToolFrameKinematics


def test_toolframe_kinematics(q_init: StretchJointPositions, file_prefix: str = "", save_plot: bool = True, show_plot: bool = False):
    kinematics = ToolFrameKinematics()
    joint_names = q_init.get_joint_names()

    n_samples = 100
    velocities = 0.1 * np.eye(6)
    twist_names = ["V_x", "V_y", "V_z", "W_x", "W_y", "W_z"]

    velocities = velocities[:3, :]
    twist_names = twist_names[:3]
    fig, axes = plt.subplots(3, 8, figsize=(30, 12), sharex="col", sharey="row")

    for i, v in enumerate(velocities):
        for j, joint_name in enumerate(joint_names):
            ax = axes[i, j]

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

                if lim_lower < -1e3:
                    lim_lower = -np.pi
                if lim_upper > 1e3:
                    lim_upper = np.pi

                if joint_name == "arm":
                    lim_upper = max(lim_upper, 0.55)

            samples = np.linspace(lim_lower, lim_upper, n_samples)
            vel_outputs = np.zeros((n_samples, 8))

            for k, val in enumerate(samples):
                q_temp = copy.deepcopy(q_init)
                setattr(q_temp, joint_name, val)

                vel_outputs[k, :] = kinematics.differential_ik(
                    q_temp,
                    "tool_attachment_site_link",
                    v
                ).to_numpy()

            for joint_idx, joint_vels in enumerate(vel_outputs.T):
                ax.plot(samples, joint_vels, alpha=0.8)

            ax.grid(True, linestyle="--", alpha=0.5)

            if i == 0:
                ax.set_title(f"Sweep {joint_name}", fontsize=16, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"Twist {twist_names[i]}\n(m/s or rad/s)", fontsize=16)
            if i == len(velocities) - 1:
                ax.set_xlabel(f"{joint_name} pos", fontsize=16)

    fig.legend(joint_names, loc="center right", fontsize=18)
    fig.suptitle(
        "Differential IK Joint Velocities over Joint Position Sweeps",
        fontsize=30,
        fontweight="bold",
        y=0.98
    )
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])

    if save_plot:
        filename = f"{file_prefix}differential_kinematics_profile.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved profiling plot to {filename}")
    if show_plot:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Profile Stretch 4 differential kinematics joint velocity sweeps across configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--save-plot", action="store_true", default=False, help="Save generated plot images to PNG files")
    parser.add_argument("--show-plot", action="store_true", default=False, help="Display plot interactive window")
    parser.add_argument("--prefix", type=str, default="cfg_", help="Filename prefix for saved plots")
    args = parser.parse_args()

    q_0 = StretchJointPositions(arm=0.2, lift=0.5)
    q_1 = StretchJointPositions(arm=0.2, lift=0.5, wrist_pitch=-np.pi / 4.0)
    q_2 = StretchJointPositions(arm=0.2, lift=0.5, wrist_pitch=-np.pi / 4.0, wrist_roll=np.pi / 3.0)
    q_3 = StretchJointPositions(arm=0.2, lift=0.5, wrist_pitch=-np.pi / 4.0, wrist_roll=np.pi / 3.0, wrist_yaw=np.pi / 4.0)

    for i, q in enumerate([q_0, q_1, q_2, q_3]):
        print(f"Profiling configuration {i}...")
        test_toolframe_kinematics(q, file_prefix=f"{args.prefix}{i}_", save_plot=args.save_plot, show_plot=args.show_plot)


if __name__ == "__main__":
    main()
