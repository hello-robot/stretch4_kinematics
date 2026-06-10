import numpy as np
import pinocchio as pin

from stretch4_urdf import get_urdf, get_urdf_calibrated

def _load_stretch4_urdf(calibrated: bool = False) -> pin.Model:
    """
    Returns the URDF model for the Stretch 4 robot.
    
    Args:
        calibrated (bool): Whether to use the calibrated URDF model.
    
    Returns:
        pin.Model: The Pinocchio model for the Stretch 4 robot.
    """
    if calibrated:
        urdf_string = get_urdf_calibrated(
            model_name = "SE4",
            batch_name = "francis",
            tool_name = "eoa_wrist_dw4_tool_sg4",
        )
    else:
        urdf_string = get_urdf(
            model_name = "SE4",
            batch_name = "francis",
            tool_name = "eoa_wrist_dw4_tool_sg4",
        )

    # Load the model directly from the XML string
    return pin.buildModelFromXML(urdf_string)


class BaseKinematics:
    def __init__(self, use_calibrated_urdf: bool = False):
        """
        Base class for handling the Pinocchio boilerplate for a robot.

        Args:
            use_calibrated_urdf (bool): Whether to use the calibrated URDF model.
        """
        self.model = _load_stretch4_urdf(use_calibrated_urdf)
        self.data = self.model.createData()

    def forward(self, q: np.ndarray, target_frame: str) -> pin.SE3:
        """
        Computes the forward kinematics for the given joint configuration.

        Args:
            q (np.ndarray): The robot's joint configuration.
            target_frame (str): The name of the frame to compute the forward kinematics for.
        
        Returns:
            pin.SE3: The pose of the target frame in the world frame.
        """
        raise NotImplementedError

    def inverse(self, target_frame: str, target_pose: pin.SE3) -> np.ndarray:
        """
        Computes the inverse kinematics for the given joint configuration.

        Args:
            target_frame (str): The name of the frame to compute the inverse kinematics for.
            target_pose (pin.SE3): The pose of the target frame in the world frame.
        
        Returns:
            np.ndarray: The joint configuration.
        """
        raise NotImplementedError

    def differential_ik(self, q: np.ndarray, target_frame: str, v_desired: np.ndarray) -> np.ndarray:
        """
        Computes the joint velocities required to achieve the desired Cartesian velocity.

        Args:
            q (np.ndarray): The robot's joint configuration.
            target_frame (str): The name of the frame to compute the velocity relationship for.
            v_desired (np.ndarray): The desired velocity of the target frame (translation and rotation).
                                    Vector of len 6, where the first 3 elements are the linear velocity
                                    and the last 3 elements are the angular velocity.
        
        Returns:
            np.ndarray: Joint velocities required to achieve the target velocity.
        """
        raise NotImplementedError


class ToolFrameKinematics(BaseKinematics):
    # old "mode 1"
    def __init__(self, use_calibrated_urdf: bool = False):
        """
        Kinematic model for the local "Tool Frame" relative control mode.
        Provides Cartesian translation relative to the robot's local tool plate / 
        gripper coordinate system (X=forward along the gripper, Y=left, Z=up).

        Args:
            use_calibrated_urdf (bool): Whether to use the calibrated URDF model.
        """
        super().__init__(use_calibrated_urdf)


class PlanarToolFrameKinematics(BaseKinematics):
    # old "mode 2"
    def __init__(self, use_calibrated_urdf: bool = False):
        """
        Kinematic model for the gravity-aligned "Planar Tool Frame" control mode.
        Projects the tool's X (forward) and Y (left) axes onto the horizontal ground plane 
        while aligning the Z axis with gravity. This enables camera-intuitive planar 
        manipulation, preventing the tool from digging into the floor or lifting up when pitched.

        Args:
            use_calibrated_urdf (bool): Whether to use the calibrated URDF model.
        """
        super().__init__(use_calibrated_urdf)


class CylindricalToolFrameKinematics(BaseKinematics):
    # old "mode 4"
    def __init__(self, use_calibrated_urdf: bool = False):
        """
        Kinematic model for the constrained "Cylindrical" control mode.
        Locks base translation (X/Y) and all wrist orientation joints (yaw, pitch, roll). 
        Resolves task-space translation at the grasp center solely using base rotation (theta), 
        lift (Z), and telescoping arm extension (radius), tracking a cylindrical coordinate grid.

        Args:
            use_calibrated_urdf (bool): Whether to use the calibrated URDF model.
        """
        super().__init__(use_calibrated_urdf)


