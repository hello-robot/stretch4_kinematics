from dataclasses import dataclass, fields
import numpy as np
from stretch4_kinematics.state.modes import Stretch4IKModes

@dataclass
class StretchJointPositions:
    """
    Represents the 8-element joint positions of Stretch 4.
    Models the omnidirectional base as SE(2): two translations and a rotation.

    Contains helper functions to convert to/from:
        - Pinocchio's 9-element configuration vector q
        - 8-element numpy array
        - 8-element dictionary
    """
    # Mobile Base (physical translations and rotation)
    base_x: float = 0.0
    base_y: float = 0.0
    base_theta: float = 0.0  # Physical angle in radians
    
    # Arm translation
    lift: float = 0.5
    arm: float = 0.0  # Merged arm extension (meters)
    
    # End of Arm (wrist joints in radians)
    wrist_yaw: float = 0.0
    wrist_pitch: float = 0.0
    wrist_roll: float = 0.0

    def get_joint_names(self) -> list[str]:
        """
        Returns the joint names in the order corresponding to the numpy array.
        
        Returns:
            list[str]: The joint names.
        """
        return [
            "base_x",
            "base_y",
            "base_theta",
            "lift",
            "arm",
            "wrist_yaw",
            "wrist_pitch",
            "wrist_roll",
        ]

    def to_pinocchio_q(
        self,
        mode: Stretch4IKModes = Stretch4IKModes.BASE_PLANAR
    ) -> np.ndarray:
        """
        Converts the physical joint state into Pinocchio's configuration vector `q`
        based on the selected IK base mode.

        Args:
            mode (Stretch4IKModes): The base mode of the model.

        Returns:
            np.ndarray: Pinocchio's configuration vector.
        """
        if mode == Stretch4IKModes.BASE_PLANAR:
            return np.array([
                self.base_x,
                self.base_y,
                np.cos(self.base_theta),
                np.sin(self.base_theta),
                self.lift,
                self.arm,
                self.wrist_yaw,
                self.wrist_pitch,
                self.wrist_roll
            ])
        elif mode == Stretch4IKModes.BASE_ROTATE:
            return np.array([
                self.base_theta,
                self.lift,
                self.arm,
                self.wrist_yaw,
                self.wrist_pitch,
                self.wrist_roll
            ])
        elif mode == Stretch4IKModes.BASE_FIXED:
            return np.array([
                self.lift,
                self.arm,
                self.wrist_yaw,
                self.wrist_pitch,
                self.wrist_roll
            ])
        else:
            raise ValueError(f"Unsupported IK mode: {mode}")

    @classmethod
    def from_pinocchio_q(
        cls,
        q: np.ndarray,
        mode: Stretch4IKModes = None
    ) -> "StretchJointPositions":
        """
        Creates a StretchJointPositions instance from a Pinocchio configuration vector `q`.
        Auto-detects the mode if not specified, based on the length of q.

        Args:
            q (np.ndarray): Pinocchio's configuration vector.
            mode (Stretch4IKModes, optional): The base mode of the model (handles variable length)

        Returns:
            StretchJointPositions: Instance of StretchJointPositions.
        """
        if mode is None:
            if len(q) == 9:
                mode = Stretch4IKModes.BASE_PLANAR
            elif len(q) == 6:
                mode = Stretch4IKModes.BASE_ROTATE
            elif len(q) == 5:
                mode = Stretch4IKModes.BASE_FIXED
            else:
                raise ValueError(f"Cannot auto-detect IK mode for configuration vector of length {len(q)}")

        if mode == Stretch4IKModes.BASE_PLANAR:
            # Recover physical theta from cos(theta) (q[2]) and sin(theta) (q[3])
            theta = np.arctan2(q[3], q[2])
            return cls(
                base_x=q[0],
                base_y=q[1],
                base_theta=theta,
                lift=q[4],
                arm=q[5],
                wrist_yaw=q[6],
                wrist_pitch=q[7],
                wrist_roll=q[8]
            )
        elif mode == Stretch4IKModes.BASE_ROTATE:
            return cls(
                base_x=0.0,
                base_y=0.0,
                base_theta=q[0],
                lift=q[1],
                arm=q[2],
                wrist_yaw=q[3],
                wrist_pitch=q[4],
                wrist_roll=q[5]
            )
        elif mode == Stretch4IKModes.BASE_FIXED:
            return cls(
                base_x=0.0,
                base_y=0.0,
                base_theta=0.0,
                lift=q[0],
                arm=q[1],
                wrist_yaw=q[2],
                wrist_pitch=q[3],
                wrist_roll=q[4]
            )
        else:
            raise ValueError(f"Unsupported IK mode: {mode}")

    def to_numpy(self) -> np.ndarray:
        """
        Converts the joint positions to a standard 8-element NumPy array.

        Returns:
            np.ndarray: The 8-element joint configuration vector.
        """
        return np.array([
            self.base_x,
            self.base_y,
            self.base_theta,
            self.lift,
            self.arm,
            self.wrist_yaw,
            self.wrist_pitch,
            self.wrist_roll,
        ])

    @classmethod
    def from_numpy(cls, q: np.ndarray) -> "StretchJointPositions":
        """
        Creates a StretchJointPositions instance from an 8-element joint configuration vector `q`.

        Args:
            q (np.ndarray): The 8-element joint configuration vector.

        Returns:
            StretchJointPositions: Instance of StretchJointPositions.
        """
        return cls(
            base_x=q[0],
            base_y=q[1],
            base_theta=q[2],
            lift=q[3],
            arm=q[4],
            wrist_yaw=q[5],
            wrist_pitch=q[6],
            wrist_roll=q[7],
        )

    def to_dict(self) -> dict:
        """
        Converts the joint positions to an 8-element configuration dictionary.

        Returns:
            dict: The 8-element joint configuration dictionary.
        """
        return {
            "base_x": self.base_x,
            "base_y": self.base_y,
            "base_theta": self.base_theta,
            "lift": self.lift,
            "arm": self.arm,
            "wrist_yaw": self.wrist_yaw,
            "wrist_pitch": self.wrist_pitch,
            "wrist_roll": self.wrist_roll,
        }

    @classmethod
    def from_dict(cls, joint_dict: dict) -> "StretchJointPositions":
        """
        Creates a StretchJointPositions instance from a dictionary of joint configurations.

        Args:
            joint_dict (dict): Dictionary containing the joint configurations.

        Returns:
            StretchJointPositions: Instance of StretchJointPositions.
        """
        return cls(
            base_x=joint_dict["base_x"],
            base_y=joint_dict["base_y"],
            base_theta=joint_dict["base_theta"],
            lift=joint_dict["lift"],
            arm=joint_dict["arm"],
            wrist_yaw=joint_dict["wrist_yaw"],
            wrist_pitch=joint_dict["wrist_pitch"],
            wrist_roll=joint_dict["wrist_roll"],
        )

    def pretty_print(self) -> None:
        """
        Pretty-prints each joint name and value on sequential lines.
        """
        for field in fields(self):
            print(f"{field.name}: {getattr(self, field.name):.4f}")
