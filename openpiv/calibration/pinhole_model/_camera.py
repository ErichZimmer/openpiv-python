import numpy as np
from numpy.typing import DTypeLike

from ._check_params import _check_parameters
from ._distortion import _distort_points_brown, _undistort_points_brown
from ._jacobian import _project_jacobian
from ._minimization import _minimize_params
from ._projection import (
    _get_inverse_vector,
    _normalize_image_points,
    _normalize_world_points,
    _project_points,
    _project_to_z,
)
from ._utils import _get_rotation_matrix, _load_parameters, _save_parameters


__all__ = [
    'camera'
]


class camera:
    """A pinhole camera with Brown distortion and optional sensor tilt.

    Attributes
    ----------
    name : str
        Name of camera.
    resolution : tuple[int, int]
        Resolution in x and y respectively.
    translation : np.ndarray
        Camera center in world coordinates, with shape (3,).
    orientation : np.ndarray
        Euler angles in radians. The camera-to-world rotation is Rx Ry Rz.
    rotation : np.ndarray
        Camera-to-world rotation matrix, with shape (3, 3).
    distortion_model : {'reduced', 'full'}
        Reduced Brown uses k1, k2, p1, p2. Full also uses k3 and k4.
        Can be set directly after construction.
    distortion1 : np.ndarray
        Eight coefficients: k1, k2, p1, p2, k3, k4, t1, t2. Here k4 is
        the radial denominator coefficient and t1, t2 are sensor-tilt
        angles in radians (tau_x, tau_y). Disabled coefficients are zero.
    tilt : bool
        Enable the two sensor-tilt angles. Can be set directly.
    refraction : {'none'}
        Refraction is not currently implemented.
    focal : tuple[float, float]
        Focal lengths in pixels.
    principal : tuple[float, float]
        Principal point in pixels. Defaults to the sensor center.
    dtype : np.dtype
        Projection dtype, either float32 or float64.

    """
    def __init__(
        self,
        name: str,
        resolution: tuple[int, int],
        translation: np.ndarray=None,
        orientation: np.ndarray=None,
        distortion_model: str='reduced',
        distortion1: np.ndarray=None,
        focal: tuple[float, float]=(1.0, 1.0),
        principal: tuple[float, float] | None=None,
        dtype: DTypeLike=np.float64,
        tilt: bool=False,
        refraction: str='none'
    ):
        self.name = name
        self.resolution = resolution
        self.dtype = np.dtype(dtype)
        self.translation = np.array([0, 0, 1] if translation is None else translation, dtype=dtype)
        self.orientation = np.array(np.zeros(3) if orientation is None else orientation, dtype=dtype)
        self.distortion1 = np.array(np.zeros(8) if distortion1 is None else distortion1, dtype=dtype)
        self.distortion_model = distortion_model
        self.tilt = tilt
        self.refraction = refraction
        self.focal = np.array(focal, dtype=dtype)
        self.principal = np.array([0, 0] if principal is None else principal, dtype=dtype)

        camera._check_parameters = _check_parameters
        camera._get_inverse_vector = _get_inverse_vector
        camera._get_rotation_matrix = _get_rotation_matrix
        camera._normalize_world_points = _normalize_world_points
        camera._normalize_image_points = _normalize_image_points
        camera._undistort_points_brown = _undistort_points_brown
        camera._distort_points_brown = _distort_points_brown
        camera.minimize_params = _minimize_params
        camera.project_points = _project_points
        camera.project_jacobian = _project_jacobian
        camera.project_to_z = _project_to_z
        camera.save_parameters = _save_parameters
        camera.load_parameters = _load_parameters

        self._check_parameters()
        if principal is None:
            self.principal = np.array(self.resolution, dtype=dtype) / 2

        self._get_rotation_matrix()

    @property
    def distortion_model(self):
        return self._distortion_model

    @distortion_model.setter
    def distortion_model(self, value):
        if value not in ('reduced', 'full'):
            raise ValueError('Distortion model must be "reduced" or "full"')
        
        self._distortion_model = value
        if value == 'reduced':
            self.distortion1[4:6] = 0

    @property
    def tilt(self):
        return self._tilt

    @tilt.setter
    def tilt(self, value):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError('Tilt must be True or False')
        
        self._tilt = value
        if not value:
            self.distortion1[6:8] = 0