from copy import deepcopy

import numpy as np

from ._global_pose import _estimate_global_pose
from ._optimization import (
    _camera_columns,
    _camera_vector,
    _levenberg_marquardt,
    _set_camera_vector,
)
from ._utils import _rotation, _rotation_angles

__all__ = [
    'align_reference_plate',
    'bundle_adjust'
]


def _board_points(
    object_points: np.ndarray
):
    points = np.asarray(object_points, dtype=np.float64)

    if points.ndim != 2 or points.shape[0] not in (2, 3):
        raise ValueError('Board points must have shape (2, N) or (3, N)')

    if points.shape[0] == 2:
        points = np.vstack([
            points,
            np.zeros(points.shape[1])
        ])

    if not np.all(np.isfinite(points)) or not np.allclose(points[2], 0):
        msg = 'Board points must be finite and have Z=0 in the board frame'
        raise ValueError(msg)

    return points


def _board_world(
    points: np.ndarray,
    pose: np.ndarray
):
    rotation, derivatives = _rotation(pose[3:])
    world = rotation @ points + pose[:3, None]

    jac = np.zeros((points.shape[1], 3, 6))
    jac[:, :, :3] = np.eye(3)

    for i in range(3):
        jac[:, :, 3+i] = (derivatives[i] @ points).T

    return world, jac


def _prepare_observations(
    cameras: list,
    points: np.ndarray,
    image_points: list
):
    if len(cameras) == 0 or len(image_points) != len(cameras):
        raise ValueError('Camera - image size mismatch')

    n_boards = len(image_points[0])

    if (
        n_boards == 0
        or any(len(views) != n_boards for views in image_points)
    ):
        msg = 'Each camera must have one image entry per capture'
        raise ValueError(msg)

    observations = []
    board_counts = np.zeros(n_boards, dtype=int)
    links = [[] for _ in range(len(cameras) + n_boards)]

    for i, (cam, views) in enumerate(zip(cameras, image_points)):
        cam._check_parameters()

        for j, image in enumerate(views):
            if image is None:
                continue

            image = np.asarray(image, dtype=np.float64)

            if image.shape != (2, points.shape[1]):
                msg = 'Image points must have shape (2, number of board corners)'
                raise ValueError(msg)

            valid = np.all(np.isfinite(image), axis=0)

            if not np.any(valid):
                continue

            observations.append((
                i,
                j,
                valid,
                image[:, valid]
            ))

            board_counts[j] += np.count_nonzero(valid)
            links[i].append(len(cameras) + j)
            links[len(cameras) + j].append(i)

    if np.any(board_counts < 4):
        msg = 'Each capture needs at least four observed corners'
        raise ValueError(msg)

    # Camera 0 anchors the common world frame for all connected captures.
    connected = {0}
    pending = [0]

    while pending:
        node = pending.pop()

        for neighbor in links[node]:
            if neighbor not in connected:
                connected.add(neighbor)
                pending.append(neighbor)

    if len(connected) != len(links):
        msg = 'All cameras and captures must be connected to camera 0'
        raise ValueError(msg)

    return observations, n_boards


def _bundle_blocks(
    state: tuple,
    points: np.ndarray,
    observations: list,
    columns: list,
    correct_distortion: bool,
    jacobian: bool=True
):
    cameras, poses = state

    offsets = np.cumsum(
        [0] + [len(active) for active in columns]
    )
    size = offsets[-1]

    A = np.zeros((size, size))
    B = np.zeros((len(poses), 6, 6))
    E = np.zeros((len(poses), size, 6))

    gc = np.zeros(size)
    gb = np.zeros((len(poses), 6))

    cost = 0.0

    for i, j, valid, image in observations:
        world, board_jac = _board_world(
            points[:, valid],
            poses[j]
        )

        cam = cameras[i]

        if not jacobian:
            camera_points = cam.rotation.T @ (
                world - cam.translation[:, None]
            )
            depth = camera_points[2]

            if np.any(depth <= 0) or np.any(np.asarray(cam.focal) <= 0):
                return np.inf

        if jacobian:
            projected, camera_jac, point_jac = cam.project_jacobian(
                world,
                correct_distortion
            )
        else:
            projected = cam.project_points(
                world,
                correct_distortion
            )

        residual = (projected - image).T.ravel()
        cost += 0.5 * residual @ residual

        if jacobian:
            c = camera_jac[:, :, columns[i]].reshape(
                -1,
                len(columns[i])
            )
            b = (point_jac @ board_jac).reshape(-1, 6)

            section = slice(
                offsets[i],
                offsets[i+1]
            )

            A[section, section] += c.T @ c
            B[j] += b.T @ b
            E[j, section] += c.T @ b

            gc[section] += c.T @ residual
            gb[j] += b.T @ residual

    if jacobian:
        return (
            cost,
            A,
            B,
            E,
            gc,
            gb
        )

    return cost


def _change_frame(
    cameras: list,
    poses: np.ndarray,
    origin: np.ndarray,
    rotation: np.ndarray
):
    origin = np.array(origin, copy=True)
    rotation = np.array(rotation, copy=True)

    new_poses = poses.copy()

    for cam in cameras:
        cam._get_rotation_matrix()

        cam.translation = rotation.T @ (cam.translation - origin)
        cam.orientation = _rotation_angles(
            rotation.T @ cam.rotation
        )

        cam._get_rotation_matrix()

    for j, pose in enumerate(poses):
        board_rotation, _ = _rotation(pose[3:])

        new_poses[j, :3] = rotation.T @ (pose[:3] - origin)
        new_poses[j, 3:] = _rotation_angles(
            rotation.T @ board_rotation
        )

    return new_poses


def bundle_adjust(
    cameras: list,
    object_points: np.ndarray,
    image_points: list,
    board_poses: np.ndarray | None=None,
    correct_focal: bool=True,
    correct_distortion: bool=True,
    max_iter: int=100,
    tolerance: float=1e-10
):
    """Calibrate cameras and shared target poses using Schur LM.

    Refine the camera parameters and one six-parameter board pose per
    capture by minimizing pixel reprojection errors. Camera 0 is held
    stationary while fitting to avoid cameras from being freely placed
    during minimization.

    Parameters
    ----------
    cameras : list
        Pinhole cameras, updated in place after optimization. When board
        poses are omitted, initialize the cameras from the target views.
    object_points : np.ndarray
        Fixed board coordinates with shape (2, N) or (3, N), with Z=0.
    image_points : list
        Image coordinates indexed by camera and capture. Each view has
        shape (2, N). Missing views are None and missing corners are NaN.
    board_poses : np.ndarray, optional
        Initial board-to-world poses with shape (captures, 6), containing
        translation followed by Rx Ry Rz Euler angles in radians. If
        omitted, initialize intrinsics with Zhang's method, align the grid
        correspondences geometrically, and estimate camera and board
        poses in camera 0's frame. Each camera needs at least three
        non-degenerate views. Otherwise use the supplied camera and board
        parameters and leave the input correspondences as supplied.
    correct_focal : bool, optional
        Refine both focal lengths. Principal points are always refined.
        Default is True.
    correct_distortion : bool, optional
        Include and refine the enabled Brown and sensor-tilt coefficients.
        If False, use an ideal pinhole projection and leave the stored
        distortion coefficients fixed. Default is True.
    max_iter : int, optional
        Maximum number of attempted LM steps. Default is 100.
    tolerance : float, optional
        Scaled gradient tolerance. The minimization will asl stop once
        the cost function reduction is less tolerance times
        max(1, current cost). Default is 1e-10.

    Returns
    -------
    board_poses : np.ndarray
        Fitted shared poses with shape (captures, 6), in camera 0's frame.
        For supplied initial poses, camera 0's initial center and
        orientation define the change from the supplied world frame.
    result : dict
        Convergence success and message, number of attempted steps,
        initial and final costs, RMS per scalar pixel residual, and
        attempted-step history. If optimization does not converge, the
        last accepted state is returned. Automatic initialization also
        returns result['permutations'][camera][capture] column indices
        mapping original observations to the common board indexing.
        Missing views have a permutation of None.

    """
    points = _board_points(object_points)

    _, n_boards = _prepare_observations(
        cameras,
        points,
        image_points
    )

    if max_iter < 1 or tolerance <= 0:
        raise ValueError('max_iter and tolerance must be positive')

    state_cameras = deepcopy(cameras)

    for cam in state_cameras:
        cam.dtype = np.dtype(np.float64)
        values = _camera_vector(cam).astype(np.float64)

        _set_camera_vector(
            cam,
            values
        )

    permutations = None

    if board_poses is None:
        board_poses, image_points, permutations = _estimate_global_pose(
            state_cameras,
            points,
            image_points,
            correct_focal=correct_focal,
            correct_distortion=correct_distortion
        )

    poses = np.array(board_poses, dtype=np.float64)

    if poses.shape != (n_boards, 6) or not np.all(np.isfinite(poses)):
        msg = 'Board poses must have shape (captures, 6) and be finite'
        raise ValueError(msg)

    observations, _ = _prepare_observations(
        state_cameras,
        points,
        image_points
    )

    # Use camera 0's initial center and orientation as the world frame.
    poses = _change_frame(
        state_cameras,
        poses,
        state_cameras[0].translation,
        state_cameras[0].rotation
    )

    state_cameras[0].translation = np.zeros(3)
    state_cameras[0].orientation = np.zeros(3)
    state_cameras[0]._get_rotation_matrix()

    columns = []

    for i, cam in enumerate(state_cameras):
        active = _camera_columns(
            cam,
            correct_focal=correct_focal,
            correct_distortion=correct_distortion,
            pose=i != 0
        )

        columns.append(active)

    n_residuals = sum(
        2 * image.shape[1]
        for _, _, _, image in observations
    )
    n_parameters = sum(len(active) for active in columns) + 6 * n_boards

    if n_residuals < n_parameters:
        msg = 'Too few observations for the number of fitted parameters'
        raise ValueError(msg)

    def linearize(
        state: tuple
    ):
        return _bundle_blocks(
            state,
            points,
            observations,
            columns,
            correct_distortion
        )

    def evaluate(
        state: tuple
    ):
        return _bundle_blocks(
            state,
            points,
            observations,
            columns,
            correct_distortion,
            jacobian=False
        )

    def update(
        state: tuple,
        dc: np.ndarray,
        db: np.ndarray
    ):
        trial_cameras = deepcopy(state[0])
        offset = 0

        for cam, active in zip(trial_cameras, columns):
            values = _camera_vector(cam)
            values[active] += dc[offset:offset+len(active)]

            _set_camera_vector(
                cam,
                values
            )

            offset += len(active)

        trial_poses = state[1] + db

        return trial_cameras, trial_poses

    state = (state_cameras, poses)

    state, result = _levenberg_marquardt(
        state,
        linearize,
        update,
        evaluate,
        max_iter=max_iter,
        tolerance=tolerance
    )

    for cam, fitted in zip(cameras, state[0]):
        values = _camera_vector(fitted).astype(cam.dtype)

        _set_camera_vector(
            cam,
            values
        )

    result['rms'] = np.sqrt(
        2 * result['cost'] / n_residuals
    )

    if permutations is not None:
        result['permutations'] = permutations

    return state[1], result


def align_reference_plate(
    cameras: list,
    board_poses: np.ndarray,
    reference: int=0
):
    """Align a fitted system with a reference plate.

    Transform the cameras and board poses together so the designated
    capture defines the world origin and axes.

    Parameters
    ----------
    cameras : list
        Fitted cameras, transformed in place.
    board_poses : np.ndarray
        Fitted board poses with shape (captures, 6), containing
        translation followed by Rx Ry Rz Euler angles.
    reference : int, optional
        Reference capture index. Default is 0.

    Returns
    -------
    board_poses : np.ndarray
        Poses in the reference-plate frame, with shape (captures, 6).
        The reference board has identity pose and camera 0 is no longer
        fixed at the origin.

    """
    poses = np.asarray(board_poses, dtype=np.float64)
    pose = poses[reference]
    rotation, _ = _rotation(pose[3:])

    return _change_frame(
        cameras,
        poses,
        pose[:3],
        rotation
    )
