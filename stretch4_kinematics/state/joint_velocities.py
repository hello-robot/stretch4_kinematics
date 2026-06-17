from dataclasses import dataclass, fields
import numpy as np

@dataclass
class StretchJointVelocities:
    """
    Represents the 8-element joint velocity vector of the Stretch 4 robot.
    """
    base_x: float = 0.0      # Linear velocity along X (m/s)
    base_y: float = 0.0      # Linear velocity along Y (m/s)
    base_theta: float = 0.0  # Angular velocity (rad/s)
    
    lift: float = 0.0        # Prismatic velocity (m/s)
    arm: float = 0.0         # Prismatic velocity (m/s)
    
    wrist_yaw: float = 0.0   # Rotational velocity (rad/s)
    wrist_pitch: float = 0.0 # Rotational velocity (rad/s)
    wrist_roll: float = 0.0  # Rotational velocity (rad/s)

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

    def to_numpy(self) -> np.ndarray:
        """
        Converts the joint velocities to a standard 8-element NumPy array.

        Returns:
            np.ndarray: The 8-element joint velocity vector.
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
    def from_numpy(cls, v: np.ndarray) -> "StretchJointVelocities":
        """
        Creates a StretchJointVelocities instance from Pinocchio's 8-element velocity vector.

        Args:
            v (np.ndarray): Pinocchio's 8-element velocity vector.

        Returns:
            StretchJointVelocities: Instance of StretchJointVelocities.
        """
        return cls(
            base_x=v[0],
            base_y=v[1],
            base_theta=v[2],
            lift=v[3],
            arm=v[4],
            wrist_yaw=v[5],
            wrist_pitch=v[6],
            wrist_roll=v[7],
        )

    def pretty_print(self) -> None:
        """
        Pretty-prints each joint name and velocity value on sequential lines.
        """
        for field in fields(self):
            print(f"{field.name}_dot: {getattr(self, field.name):.4f}")
