from enum import Enum, auto

class Stretch4IKModes(Enum):
    """
    Represents the different URDF configurations for the Stretch 4 URDF.
    """
    BASE_FIXED = auto()   # base cannot move
    BASE_ROTATE = auto()  # base can only rotate about z-axis
    BASE_PLANAR = auto()  # base can translate and rotate (SE(2))
