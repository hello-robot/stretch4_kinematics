import argparse
import sys
import time
import traceback

import numpy as np

from stretch4_body.core.robot_params import RobotParams
from stretch4_body.robot.robot_client import RobotClient
from stretch4_kinematics.kinematic_models.tool_frame_kinematics import ToolFrameKinematics
from stretch4_kinematics.state import (
    StretchJointPositions,
    StretchJointVelocities,
)
from stretch4_kinematics.stretch_interface import StretchInterface

from gamepad_mapper import GamepadMapper

# Lookahead gain for wrist movements to compensate for communication and motor lag
WRIST_JOG_LOOKAHEAD_GAIN = 2.0


class FlyingGripperTeleop:
    """
    Main class for running gripper-centric teleoperation.
    """

    def __init__(self, speed: str = "medium", numerical: bool = False) -> None:
        """
        Initializes the teleoperation node.

        Args:
            speed: Speed profile ('low', 'medium', 'high', 'max').
            numerical: Run numerically in simulation mode.
        """
        self.numerical = numerical
        self.speed = speed

        # Initialize gamepad
        print("Initializing Gamepad Mapper...")
        try:
            self.gamepad = GamepadMapper()
        except Exception as e:
            print(f"Failed to connect to gamepad: {e}")
            sys.exit(1)

        # Initialize hardware/simulation
        if self.numerical:
            print("Running in NUMERICAL simulation mode.")
            self.robot = None
            self.robot_interface = None
            # Set up starting simulated joint position
            self.current_pos = StretchJointPositions(
                base_x=0.0,
                base_y=0.0,
                base_theta=0.0,
                lift=0.5,
                arm=0.1,
                wrist_yaw=0.0,
                wrist_pitch=0.0,
                wrist_roll=0.0,
            )
        else:
            print("Connecting to robot client...")
            self.robot = RobotClient()
            if not self.robot.startup():
                print("Failed to start robot client connection.")
                self.gamepad.stop()
                sys.exit(1)

            self.robot_interface = StretchInterface(robot=self.robot)
            self.robot_interface.reset_odometry_offset()
            self.current_pos = self.robot_interface.get_joint_position()

        # Initialize kinematics solver
        self.ik_solver = ToolFrameKinematics()

        # Configure velocity scaling profiles
        self._initialize_velocity_profiles(speed)

    def _initialize_velocity_profiles(self, speed: str) -> None:
        """
        Configures the velocity scaling parameters using RobotParams.

        Args:
            speed: The selected speed profile.
        """
        params = RobotParams().get_params()[1]
        
        speed_mapping = {
            "low": "slow",
            "medium": "default",
            "high": "fast",
            "max": "max",
        }
        motion_prof = speed_mapping[speed]

        # Extract gripper velocity/acceleration settings
        self.vel_grip = params["stretch_gripper"]["motion"][motion_prof]["vel"]
        self.acc_grip = params["stretch_gripper"]["motion"][motion_prof]["accel"]

        # Scale joystick commands
        if speed == "low":
            self.gamepad_speed_trans = 0.05
            self.gamepad_speed_rot = 0.4
        elif speed == "medium":
            self.gamepad_speed_trans = 0.15
            self.gamepad_speed_rot = 0.5
        else:
            self.gamepad_speed_trans = 0.25
            self.gamepad_speed_rot = 1.0

    def run(self) -> None:
        """
        Starts the teleoperation loop.
        """
        control_mode = 1
        hz = 30.0
        dt = 1.0 / hz
        rate = time.time()
        last_pushed_was_zero = False

        gripper_close_pct = -60.0
        gripper_open_pct = 60.0
        pitch_sign_mult = -1.0

        print("====================================")
        print("Gripper-Centric Teleop Started")
        print("Press Top Button (Y) to Toggle Modes")
        print("Mode 1: Gripper Frame Relative (IK)")
        print("Mode 3: Joint-Space Direct Control")
        print("Ctrl+C to Quit")
        print("====================================")

        try:
            while True:
                cmd = self.gamepad.get_commands()

                # Get current joint status
                if not self.numerical:
                    self.current_pos = self.robot_interface.get_joint_position()

                if not cmd:
                    if not last_pushed_was_zero:
                        if not self.numerical:
                            self.robot_interface.cmd_zero_velocity()
                        last_pushed_was_zero = True
                    time.sleep(dt)
                    continue

                if cmd["toggle"]:
                    control_mode = 3 if control_mode == 1 else 1
                    mode_names = {
                        1: "Gripper Frame Relative (IK)",
                        3: "Joint-Space Direct Control",
                    }
                    print(f"--> Switched to Mode {control_mode}: {mode_names[control_mode]}")

                # Gripper Command
                if cmd["grip"] == "OPEN":
                    if not self.numerical:
                        self.robot.end_of_arm.move_by(
                            "stretch_gripper",
                            gripper_open_pct,
                            self.vel_grip,
                            self.acc_grip,
                        )
                elif cmd["grip"] == "CLOSE":
                    if not self.numerical:
                        self.robot.end_of_arm.move_by(
                            "stretch_gripper",
                            gripper_close_pct,
                            self.vel_grip,
                            self.acc_grip,
                        )

                # Proportional damping via left trigger
                left_trigger = cmd.get("left_trigger", 0.0)
                speed_multiplier = 1.0 - (0.5 * left_trigger)

                # Initialize empty velocity vector
                v_joint = StretchJointVelocities()

                if control_mode == 3:
                    # Direct joint space mapping
                    v_joint.lift = cmd["v_desired"][2] * self.gamepad_speed_trans * speed_multiplier

                    if cmd.get("right_trigger", 0.0) > 0.1:
                        # Arm Extend/Retract (Left Stick Y)
                        v_joint.arm = cmd["v_desired"][0] * self.gamepad_speed_trans * speed_multiplier
                        
                        # Wrist Roll (Left Stick X)
                        v_joint.wrist_roll = (
                            cmd["v_desired"][1] * self.gamepad_speed_rot * speed_multiplier * -1.0
                        )
                        
                        # Wrist Pitch (Right Stick Y)
                        v_joint.wrist_pitch = (
                            cmd["rot_change"][1] * self.gamepad_speed_rot * speed_multiplier
                        )
                        
                        # Wrist Yaw (Right Stick X)
                        v_joint.wrist_yaw = (
                            cmd["rot_change"][0] * self.gamepad_speed_rot * speed_multiplier
                        )
                    else:
                        # Base Translation Forward/Backward (Left Stick Y)
                        v_joint.base_x = cmd["v_desired"][0] * self.gamepad_speed_trans * speed_multiplier
                        
                        # Base Translation Left/Right (Left Stick X)
                        v_joint.base_y = cmd["v_desired"][1] * self.gamepad_speed_trans * speed_multiplier
                        
                        # Base Rotation (Right Stick X)
                        v_joint.base_theta = (
                            cmd["rot_change"][0] * self.gamepad_speed_rot * speed_multiplier
                        )
                        
                        # Arm (Right Stick Y)
                        v_joint.arm = cmd["rot_change"][1] * self.gamepad_speed_trans * speed_multiplier
                else:
                    # Gripper Frame Relative Control (IK Mode 1)
                    v_desired_lin = (
                        np.array(cmd["v_desired"]) * self.gamepad_speed_trans * speed_multiplier
                    )
                    v_desired_6d = np.zeros(6)
                    v_desired_6d[:3] = v_desired_lin

                    # Compute base, lift, arm velocities via stateless differential IK
                    v_joint = self.ik_solver.differential_ik(
                        self.current_pos,
                        "tool_attachment_site_link",
                        v_desired_6d,
                    )

                    # Map wrist velocities directly from rotation commands
                    v_joint.wrist_yaw += (
                        cmd["rot_change"][0] * self.gamepad_speed_rot * speed_multiplier
                    )
                    v_joint.wrist_pitch = (
                        cmd["rot_change"][1]
                        * self.gamepad_speed_rot
                        * speed_multiplier
                        * pitch_sign_mult
                    )
                    v_joint.wrist_roll = (
                        cmd["rot_change"][2] * self.gamepad_speed_rot * speed_multiplier
                    )

                    # Apply lookahead gain to wrist joints to reduce control lag
                    v_joint.wrist_yaw *= WRIST_JOG_LOOKAHEAD_GAIN
                    v_joint.wrist_pitch *= WRIST_JOG_LOOKAHEAD_GAIN
                    v_joint.wrist_roll *= WRIST_JOG_LOOKAHEAD_GAIN

                # Check if any non-zero velocity is commanded
                is_active = (
                    np.any(v_joint.to_numpy() != 0.0)
                    or (cmd["grip"] is not None)
                )

                if is_active:
                    if self.numerical:
                        # Integrate simulated positions
                        theta = self.current_pos.base_theta
                        cos_t = np.cos(theta)
                        sin_t = np.sin(theta)
                        world_base_x_vel = (
                            v_joint.base_x * cos_t - v_joint.base_y * sin_t
                        )
                        world_base_y_vel = (
                            v_joint.base_x * sin_t + v_joint.base_y * cos_t
                        )

                        q_next = self.current_pos.to_numpy()
                        q_next[0] += world_base_x_vel * dt
                        q_next[1] += world_base_y_vel * dt
                        q_next[2] += v_joint.base_theta * dt
                        q_next[3:] += v_joint.to_numpy()[3:] * dt

                        # Clamp to physical joint limits
                        # q_next = np.clip(
                        #     q_next,
                        #     self.ik_solver.model.lowerPositionLimit,
                        #     self.ik_solver.model.upperPositionLimit,
                        # )
                        self.current_pos = StretchJointPositions.from_numpy(q_next)

                        # Print simulated telemetry
                        print(
                            f"Simulated Pose: Base=[{self.current_pos.base_x:.3f}, {self.current_pos.base_y:.3f}, {self.current_pos.base_theta:.3f}], "
                            f"Lift={self.current_pos.lift:.3f}, Arm={self.current_pos.arm:.3f}, "
                            f"Wrist=[{self.current_pos.wrist_yaw:.3f}, {self.current_pos.wrist_pitch:.3f}, {self.current_pos.wrist_roll:.3f}]",
                            end="\r",
                        )
                    else:
                        self.robot_interface.cmd_velocities(v_joint)
                    last_pushed_was_zero = False
                elif not last_pushed_was_zero:
                    if not self.numerical:
                        self.robot_interface.cmd_zero_velocity()
                    last_pushed_was_zero = True

                # Dynamic sleep to maintain loop frequency
                sleep_time = rate + dt - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                rate = time.time()

        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            print(f"\nError in teleop loop: {e}")
        finally:
            self.stop()

    def stop(self) -> None:
        """
        Safely stops the robot and stops the gamepad mapper thread.
        """
        print("Stopping teleop...")
        if not self.numerical and self.robot_interface is not None:
            try:
                self.robot_interface.cmd_zero_velocity()
            except Exception as e:
                print(f"Failed to stop robot: {e}")
        self.gamepad.stop()
        if not self.numerical and self.robot is not None:
            self.robot.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gripper-centric Teleoperation for Stretch 4"
    )
    parser.add_argument(
        "--speed",
        choices=["low", "medium", "high", "max"],
        default="medium",
        help="Velocity scaling profile for joints",
    )
    parser.add_argument(
        "--numerical",
        "-n",
        action="store_true",
        help="Run numerically in simulation mode without hardware",
    )
    args = parser.parse_args()

    teleop = FlyingGripperTeleop(speed=args.speed, numerical=args.numerical)
    teleop.run()


if __name__ == "__main__":
    main()
