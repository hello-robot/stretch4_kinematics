from stretch4_kinematics.kinematic_models.base_kinematic_models import StretchKinematics

class PlanarToolFrameKinematics(StretchKinematics):
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
        raise NotImplementedError(
            "PlanarToolFrameKinematics is under active development and not yet implemented."
        )
