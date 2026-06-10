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
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        
        frame_id = self.model.getFrameId(target_frame)
        return self.data.oMf[frame_id]

    def inverse(self, target_frame: str, target_pose: pin.SE3, q_guess: np.ndarray = None,
                max_iter: int = 200, eps: float = 1e-4, damp: float = 1e-6) -> np.ndarray:
        """
        Computes the numerical inverse kinematics using the standard Closed-Loop 
        Inverse Kinematics (CLIK) algorithm with Levenberg-Marquardt damping.
        
        Reference:
            https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/devel/doxygen-html/md_doc_b-examples_i-inverse-kinematics.html

        Args:
            target_frame (str): The name of the frame to compute the inverse kinematics for.
            target_pose (pin.SE3): The desired pose of the target frame in the world frame.
            q_guess (np.ndarray, optional): Initial joint configuration guess. Defaults to neutral configuration.
            max_iter (int): Maximum number of iterations.
            eps (float): Convergence tolerance.
            damp (float): Damping factor for pseudo-inverse.
        
        Returns:
            np.ndarray: The joint configuration solving the IK.
        """
        q = pin.neutral(self.model) if q_guess is None else q_guess.copy()
        frame_id = self.model.getFrameId(target_frame)
        
        for i in range(max_iter):
            # compute current fk solution
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            # compute error between target and current pose
            dMi = target_pose.actInv(self.data.oMf[frame_id])
            err = pin.log(dMi).vector
            
            # stop if error is small
            if np.linalg.norm(err) < eps:
                break
            
            # compute Jacobian
            J = pin.computeFrameJacobian(self.model, self.data, q, frame_id, pin.ReferenceFrame.LOCAL)
            
            # compute velocity correction using Levenberg-Marquardt damping
            # This conditions the Jacobian to prevent singularities
            J_JT = J @ J.T + damp * np.eye(6)
            dq = -J.T @ np.linalg.solve(J_JT, err)
            
            # update joint configuration
            q = pin.integrate(self.model, q, dq)

        return q

    def differential_ik(self, q: np.ndarray, target_frame: str, v_desired: np.ndarray) -> np.ndarray:
        """
        Abstract method to be implemented depending on the specific control mode / kinematic structure.
        
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


