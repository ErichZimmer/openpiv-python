from copy import deepcopy

import numpy as np

from .. import _cal_doc_utils
from ._optimization import (
    _camera_columns,
    _camera_vector,
    _levenberg_marquardt,
    _set_camera_vector,
)


@_cal_doc_utils.docfiller
def _minimize_params(
    self,
    object_points: list,
    image_points: list,
    correct_focal: bool=False,
    correct_distortion: bool=False,
    max_iter: int=1000,
    iterations: int=3
):
    """Refine pinhole camera parameters using NumPy Levenberg-Marquardt.

    Parameters
    ----------
    %(object_points)s
    %(image_points)s
    correct_focal : bool, optional
        Refine both focal lengths. Default is False.
    correct_distortion : bool, optional
        Include distortion in projection and refine the enabled Brown and
        tilt coefficients. When False, fit an ideal pinhole projection,
        ignoring the stored distortion coefficients. Default is False.
    max_iter : int, optional
        Maximum LM steps per refinement pass. Default is 1000.
    iterations : int, optional
        Number of successive refinement passes. Default is 3.

    Returns
    -------
    None
        The camera parameters are updated in place.

    Raises
    ------
    ValueError
        If camera parameters are invalid, object points do not have shape
        (3, N), image points do not have matching shape (2, N), or too few
        calibration points are supplied.
    RuntimeError
        If a Levenberg-Marquardt refinement pass fails.

    """
    self._check_parameters()

    object_points = np.array(object_points, dtype=np.float64)
    image_points = np.array(image_points, dtype=np.float64)

    if object_points.ndim != 2 or object_points.shape[0] != 3:
        raise ValueError('Object points must have shape (3, N)')

    if image_points.shape != (2, object_points.shape[1]):
        raise ValueError('Object point image point size mismatch')

    columns = _camera_columns(
        self,
        correct_focal=correct_focal,
        correct_distortion=correct_distortion
    )
    n_parameters = len(columns)
    min_points = max(9, (n_parameters + 1) // 2)

    if object_points.shape[1] < min_points:
        msg = f'Too few points to calibrate. Need at least {min_points} points'
        raise ValueError(msg)

    def linearize(
        cam
    ):
        projected, projection_jacobian, _ = cam.project_jacobian(
            object_points,
            correct_distortion=correct_distortion
        )

        # Interleave x and y residuals in the same order as Jacobian rows.
        residual = (projected - image_points).T
        residual = residual.reshape(-1)

        parameter_jacobian = projection_jacobian[:, :, columns]
        parameter_jacobian = parameter_jacobian.reshape(-1, n_parameters)

        cost = 0.5 * residual @ residual
        camera_normal = parameter_jacobian.T @ parameter_jacobian
        camera_gradient = parameter_jacobian.T @ residual

        # The Schur LM expects board parameters, which do not exist
        # for single-camera optimizations.
        board_normal = np.zeros((0, 6, 6))
        cross = np.zeros((0, n_parameters, 6))
        board_gradient = np.zeros((0, 6))

        return (
            cost,
            camera_normal,
            board_normal,
            cross,
            camera_gradient,
            board_gradient
        )

    def update(
        cam,
        camera_step: np.ndarray,
        palceholder: np.ndarray
    ):
        trial = deepcopy(cam)
        values = _camera_vector(trial)
        values[columns] += camera_step

        _set_camera_vector(
            trial,
            values
        )

        return trial

    def evaluate(
        cam
    ):
        if np.any(np.asarray(cam.focal) <= 0):
            return np.inf

        projected = cam.project_points(
            object_points,
            correct_distortion=correct_distortion
        )
        residual = projected - image_points
        cost = 0.5 * np.sum(residual**2)

        return cost

    state = deepcopy(self)
    state.dtype = np.dtype(np.float64)

    for _ in range(iterations):
        # TODO: Adapt the Schur LM optimizer for single cameras and multiple cameras
        state, result = _levenberg_marquardt(
            state,
            linearize,
            update,
            evaluate,
            max_iter=max_iter
        )

        if not result['success']:
            msg = f"Pinhole refinement failed: {result['message']}"
            raise RuntimeError(msg)

    fitted_values = _camera_vector(state)
    fitted_values = fitted_values.astype(self.dtype)

    _set_camera_vector(
        self,
        fitted_values
    )