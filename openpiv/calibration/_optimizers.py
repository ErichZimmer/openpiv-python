from collections.abc import Callable

import numpy as np


__all__ = [
    'optimize_general',
    'optimize_board'
]


def _column_scales(
    jacobian: np.ndarray
):
    column_norm = np.linalg.norm(jacobian, axis=0)

    return 1 / np.maximum(column_norm, 1e-6)


def _target_scales(
    camera_jacobians: list,
    board_jacobians: list
):
    camera_norm = np.zeros(camera_jacobians[0].shape[1])

    for jacobian in camera_jacobians:
        camera_norm = np.hypot(
            camera_norm,
            np.linalg.norm(jacobian, axis=0)
        )

    camera_scale = 1 / np.maximum(camera_norm, 1e-6)
    board_scale = [_column_scales(jacobian) for jacobian in board_jacobians]

    return camera_scale, board_scale


def _qr_solve(
    jacobian: np.ndarray,
    rhs: np.ndarray
):
    n_parameters = jacobian.shape[1]
    augmented = np.column_stack([jacobian, rhs])
    factor = np.linalg.qr(augmented, mode='r')

    return np.linalg.solve(
        factor[:n_parameters, :n_parameters],
        factor[:n_parameters, n_parameters]
    )


def _point_step(
    residual: np.ndarray,
    jacobian: np.ndarray,
    damping: float
):
    """Calculate an LM step for fixed world points using damped QR.

    Parameters
    ----------
    residual : np.ndarray
        Pixel residuals with shape (N,).
    jacobian : np.ndarray
        Parameter derivatives with shape (N, M).
    damping : float
        Positive damping in scaled parameter coordinates.

    Returns
    -------
    step : np.ndarray
        Parameter update in the original units, with shape (M,).
    predicted : float
        Predicted reduction of the undamped quadratic cost.

    """
    scale = _column_scales(jacobian)
    n_parameters = jacobian.shape[1]

    augmented = np.vstack([
        jacobian * scale,
        np.sqrt(damping) * np.eye(n_parameters)
    ])
    rhs = np.concatenate([-residual, np.zeros(n_parameters)])

    step = scale * _qr_solve(augmented, rhs)
    change = np.dot(jacobian, step)
    predicted = -np.dot(residual, change) - 0.5 * np.dot(change, change)

    return step, predicted


def _target_step(
    residuals: list,
    camera_jacobians: list,
    board_jacobians: list,
    damping: float
):
    sc, sb = _target_scales(camera_jacobians, board_jacobians)
    n_camera = len(sc)
    reduced_camera = []
    reduced_rhs = []
    recovery = []
    root_damping = np.sqrt(damping)

    for residual, camera, board, scale in zip(
        residuals, camera_jacobians, board_jacobians, sb
    ):
        n_rows = len(residual)
        n_board = board.shape[1]
        block = np.zeros((n_rows+n_board, n_board+n_camera+1))

        block[:n_rows, :n_board] = board * scale
        block[:n_rows, n_board:n_board+n_camera] = camera * sc
        block[:n_rows, -1] = residual
        block[n_rows:, :n_board] = root_damping * np.eye(n_board)

        factor = np.linalg.qr(block, mode='r')
        recovery.append(factor[:n_board])
        reduced_camera.append(factor[n_board:, n_board:n_board+n_camera])
        reduced_rhs.append(-factor[n_board:, -1])

    reduced_camera.append(root_damping * np.eye(n_camera))
    reduced_rhs.append(np.zeros(n_camera))
    camera_step = _qr_solve(
        np.vstack(reduced_camera),
        np.concatenate(reduced_rhs)
    )

    board_steps = []

    for factor, scale in zip(recovery, sb):
        n_board = len(scale)
        rhs = -factor[:, -1] - np.dot(
            factor[:, n_board:n_board+n_camera],
            camera_step
        )
        board_step = np.linalg.solve(factor[:, :n_board], rhs)
        board_steps.append(scale * board_step)

    dc = sc * camera_step
    db = np.array(board_steps)
    predicted = 0.0

    for residual, camera, board, step in zip(
        residuals, camera_jacobians, board_jacobians, db
    ):
        change = np.dot(camera, dc) + np.dot(board, step)
        predicted -= np.dot(residual, change) + 0.5 * np.dot(change, change)

    return dc, db, predicted


def _levenberg_marquardt(
    state,
    linearize: Callable,
    update: Callable,
    evaluate: Callable,
    max_iter: int=100,
    tolerance: float=1e-10,
    mode: str='target'
):
    if mode not in ('target', 'points'):
        raise ValueError('LM mode must be "target" or "points"')

    if max_iter < 1 or tolerance <= 0:
        raise ValueError('max_iter and tolerance must be positive')

    damping = 1e-3
    multiplier = 2.0
    blocks = linearize(state)
    initial_cost = blocks[0]
    history = []
    success = False
    message = 'Maximum iterations reached'

    for _ in range(max_iter):
        if mode == 'target':
            cost, residuals, camera_jacobians, board_jacobians = blocks
            sc, sb = _target_scales(camera_jacobians, board_jacobians)
            camera_gradient = np.zeros(len(sc))
            board_gradient = []

            for residual, camera, board, scale in zip(
                residuals, camera_jacobians, board_jacobians, sb
            ):
                camera_gradient += sc * np.dot(camera.T, residual)
                board_gradient.append(scale * np.dot(board.T, residual))

            gradient = np.concatenate([camera_gradient, *board_gradient])
        else:
            cost, residual, jacobian = blocks
            scale = _column_scales(jacobian)
            gradient = scale * np.dot(jacobian.T, residual)

        if np.max(np.abs(gradient)) < tolerance:
            success = True
            message = 'Gradient tolerance reached'
            break

        try:
            if mode == 'target':
                dc, db, predicted = _target_step(
                    residuals, camera_jacobians, board_jacobians, damping
                )
                trial = update(state, dc, db)
            else:
                step, predicted = _point_step(residual, jacobian, damping)
                trial = update(state, step)

            trial_cost = evaluate(trial)

        except (np.linalg.LinAlgError, FloatingPointError):
            predicted = 0.0
            trial_cost = np.inf

        actual = cost - trial_cost
        if predicted > 0 and np.isfinite(trial_cost):
            ratio = actual / predicted
        else:
            ratio = -np.inf

        accepted = ratio > 0
        history.append({
            'cost': cost,
            'trial_cost': trial_cost,
            'damping': damping,
            'ratio': ratio,
            'accepted': accepted
        })

        if accepted:
            state = trial
            blocks = linearize(state)
            damping *= max(1 / 3, 1 - (2 * ratio - 1)**3)
            damping = max(damping, 1e-15)
            multiplier = 2.0

            if actual < tolerance * max(1.0, cost):
                success = True
                message = 'Cost tolerance reached'
                break
        else:
            damping *= multiplier
            multiplier *= 2

            if damping > 1e16:
                message = 'Unable to find a decreasing step'
                break

    return state, {
        'success': success,
        'message': message,
        'iterations': len(history),
        'initial_cost': initial_cost,
        'cost': blocks[0],
        'history': history
    }


def optimize_general(
    fun: Callable,
    x0: np.ndarray,
    jac: Callable,
    max_iter: int=100,
    tolerance: float=1e-10
):
    """Fit a parameter vector using QR Levenberg-Marquardt.

    Parameters
    ----------
    fun : callable
        Return the residual vector for the supplied parameters.
    x0 : np.ndarray
        Initial parameter vector with shape (M,).
    jac : callable
        Return the residual Jacobian with shape (N, M) for the
        supplied parameters.
    max_iter : int, optional
        Maximum attempted LM steps. Default is 100.
    tolerance : float, optional
        Scaled gradient and cost reduction tolerance. Default is 1e-10.

    Returns
    -------
    parameters : np.ndarray
        Final accepted parameter vector.
    result : dict
        Convergence status, costs, and trial history.

    Raises
    ------
    ValueError
        If max_iter or tolerance is not positive.

    """
    def linearize(parameters):
        residual = np.asarray(fun(parameters), dtype=np.float64)
        cost = 0.5 * np.dot(residual, residual)

        return cost, residual, jac(parameters)

    def update(parameters, step):
        return parameters + step

    def evaluate(parameters):
        residual = np.asarray(fun(parameters), dtype=np.float64)

        return 0.5 * np.dot(residual, residual)

    return _levenberg_marquardt(
        np.array(x0, dtype=np.float64),
        linearize,
        update,
        evaluate,
        max_iter=max_iter,
        tolerance=tolerance,
        mode='points'
    )


def optimize_board(
    fun: Callable,
    initial_camera_parameters: np.ndarray,
    initial_board_poses: np.ndarray,
    jac: Callable,
    max_iter: int=100,
    tolerance: float=1e-10
):
    """Fit shared camera parameters and one board pose per capture.

    Parameters
    ----------
    fun : callable
        Given the camera vector and board poses, return a list of
        residual vectors in capture order.
    initial_camera_parameters : np.ndarray
        Initial shared camera parameter vector with shape (M,).
    initial_board_poses : np.ndarray
        Initial board poses with shape (captures, 6).
    jac : callable
        Given the camera vector and board poses, return two lists of
        Jacobians in capture order. For capture j, the camera Jacobian
        has shape (N_j, M) and the board Jacobian has shape (N_j, 6).
    max_iter : int, optional
        Maximum attempted LM steps. Default is 100.
    tolerance : float, optional
        Scaled gradient and cost reduction tolerance. Default is 1e-10.

    Returns
    -------
    camera_parameters : np.ndarray
        Final accepted shared camera parameters.
    board_poses : np.ndarray
        Final accepted board poses.
    result : dict
        Convergence status, costs, and trial history.

    Raises
    ------
    ValueError
        If max_iter or tolerance is not positive.

    Notes
    -----
    The QR step adapts the square-root elimination in Demmel et al. [1]_
    to six-parameter board poses. Each capture's pose is eliminated
    before solving for the shared camera parameters, similar to how it is
    done in Schur optimization commonly used in bundle adjustment algorithms.
    QR decomposition allows for better handeling of the residuals and
    jacobians since it avoids normal matrices which can amplify noise.

    References
    ----------
    .. [1] Demmel, N., Sommer, C., Cremers, D., and Usenko, V. (2021).
        Square Root Bundle Adjustment for Large-Scale Reconstruction.
        Proceedings of the IEEE/CVF Conference on Computer Vision and
        Pattern Recognition, 11723-11732.
        https://doi.org/10.1109/CVPR46437.2021.01155

    """
    def linearize(state):
        residuals = fun(*state)
        cost = sum(0.5 * np.dot(residual, residual) for residual in residuals)
        camera_jacobians, board_jacobians = jac(*state)

        return cost, residuals, camera_jacobians, board_jacobians

    def update(state, camera_step, board_step):
        camera_parameters, board_poses = state

        return camera_parameters + camera_step, board_poses + board_step

    def evaluate(state):
        residuals = fun(*state)

        return sum(0.5 * np.dot(residual, residual) for residual in residuals)

    initial = (
        np.array(initial_camera_parameters, dtype=np.float64),
        np.array(initial_board_poses, dtype=np.float64)
    )
    state, result = _levenberg_marquardt(
        initial,
        linearize,
        update,
        evaluate,
        max_iter=max_iter,
        tolerance=tolerance,
        mode='target'
    )
    camera_parameters, board_poses = state

    return camera_parameters, board_poses, result