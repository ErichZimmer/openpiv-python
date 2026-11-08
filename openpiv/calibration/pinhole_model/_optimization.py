from collections.abc import Callable

import numpy as np

from ._distortion import _active_distortion


def _camera_columns(
    cam,
    correct_focal: bool=True,
    correct_distortion: bool=True,
    pose: bool=True
):
    if pose:
        columns = list(range(6))
    else:
        columns = []

    if correct_focal:
        columns += [6, 7]

    columns += [8, 9]

    if correct_distortion:
        columns += list(10 + _active_distortion(cam))

    return np.array(columns, dtype=int)


def _camera_vector(
    cam
):
    return np.hstack([
        cam.translation,
        cam.orientation,
        cam.focal,
        cam.principal,
        cam.distortion1
    ])


def _set_camera_vector(
    cam,
    values: np.ndarray
):
    cam.translation = values[:3].copy()
    cam.orientation = values[3:6].copy()

    cam.focal = values[6:8].copy()
    cam.principal = values[8:10].copy()

    cam.distortion1 = values[10:18].copy()

    cam._get_rotation_matrix()


def _block_scales(
    A: np.ndarray,
    B: np.ndarray
):
    camera_diagonal = np.diag(A)
    board_diagonal = np.diagonal(
        B,
        axis1=1,
        axis2=2
    )

    camera = 1 / np.sqrt(
        np.maximum(camera_diagonal, 1e-12)
    )

    board = 1 / np.sqrt(
        np.maximum(board_diagonal, 1e-12)
    )

    return camera, board


def _schur_step(
    A: np.ndarray,
    B: np.ndarray,
    E: np.ndarray,
    camera_gradient: np.ndarray,
    board_gradient: np.ndarray,
    damping: float
):
    """Calculate an LM step using Schur elimination.

    Scale the parameters, damp the camera and board blocks, and solve
    the reduced camera system. Recover each board update through
    back-substitution without forming matrix inverses.

    Parameters
    ----------
    A : np.ndarray
        Camera normal matrix with shape (M, M).
    B : np.ndarray
        Board normal matrices with shape (captures, 6, 6).
    E : np.ndarray
        Camera-board cross matrices with shape (captures, M, 6).
    camera_gradient : np.ndarray
        Camera gradient with shape (M,).
    board_gradient : np.ndarray
        Board gradients with shape (captures, 6).
    damping : float
        LM damping applied in scaled parameter coordinates.

    Returns
    -------
    dc : np.ndarray
        Camera updates in the original units, with shape (M,).
    db : np.ndarray
        Board updates in the original units, with shape (captures, 6).
    predicted : float
        Predicted reduction of the undamped quadratic cost model.

    Notes
    -----
    Damping is applied to both diagonal blocks before elimination.
    Each board block is solved once with multiple right-hand sides.
    The reduced system and back-substitution follow Equations (18)-(22)
    in Lourakis and Argyros [1]_, with six-parameter board poses in place
    of independently fitted 3D points.

    References
    ----------
    .. [1] Lourakis, M. I. A., and Argyros, A. A. (2009). SBA: A software
        package for generic sparse bundle adjustment. ACM Transactions
        on Mathematical Software, 36(1), Article 2.
        https://doi.org/10.1145/1486525.1486527

    """
    sc, sb = _block_scales(
        A,
        B
    )

    # Transform the normal equations into scaled coordinates.
    a = sc[:, None] * A * sc[None, :]
    b = sb[:, :, None] * B * sb[:, None, :]
    e = sc[None, :, None] * E * sb[:, None, :]

    gc = sc * camera_gradient
    gb = sb * board_gradient

    # Apply damping before eliminating the board parameters.
    a += damping * np.eye(a.shape[0])
    b += damping * np.eye(6)[None, :, :]

    reduced = a.copy()
    rhs = -gc.copy()

    solved_cross = []
    solved_gradient = []

    for j in range(len(b)):
        # Solve for B^-1 E.T and B^-1 g together without forming B^-1.
        solved = np.linalg.solve(
            b[j],
            np.column_stack([
                e[j].T,
                gb[j]
            ])
        )

        cross = solved[:, :-1]
        gradient = solved[:, -1]

        reduced -= e[j] @ cross
        rhs += e[j] @ gradient

        solved_cross.append(cross)
        solved_gradient.append(gradient)

    # Solve for the scaled camera updates.
    dc = np.linalg.solve(
        reduced,
        rhs
    )

    # Recover the scaled board updates.
    db = np.empty_like(gb)

    for j in range(len(b)):
        db[j] = (
            -solved_gradient[j]
            - solved_cross[j] @ dc
        )

    # Return the updates to the original parameter units.
    dc *= sc
    db *= sb

    # Evaluate the undamped quadratic model at the proposed step.
    linear = (
        camera_gradient @ dc
        + np.sum(board_gradient * db)
    )

    quadratic = dc @ A @ dc

    for j in range(len(B)):
        quadratic += (
            db[j] @ B[j] @ db[j]
            + 2 * dc @ E[j] @ db[j]
        )

    predicted = -linear - 0.5 * quadratic

    return dc, db, predicted


def _levenberg_marquardt(
    state,
    linearize: Callable,
    update: Callable,
    evaluate: Callable,
    max_iter: int=100,
    tolerance: float=1e-10
):
    """Minimize a least-squares cost using LM with Schur steps.

    Parameters
    ----------
    state : object
        Initial accepted parameter state.
    linearize : callable
        Return cost, A, B, E, camera_gradient, and board_gradient for
        the supplied state.
    update : callable
        Return a separate trial state after applying camera and board
        updates to the supplied state.
    evaluate : callable
        Calculate the cost of a supplied trial state.
    max_iter : int, optional
        Maximum number of attempted LM steps. Default is 100.
    tolerance : float, optional
        Scaled gradient tolerance. Also stop when an accepted cost
        reduction is less than tolerance times max(1, current cost).
        Default is 1e-10.

    Returns
    -------
    state : object
        Final accepted parameter state.
    result : dict
        Convergence success and message, number of attempted steps,
        initial and final costs, and the attempted-step history.

    """
    damping = 1e-3
    multiplier = 2.0

    blocks = linearize(state)
    initial_cost = blocks[0]

    history = []
    success = False
    message = 'Maximum iterations reached'

    for _ in range(max_iter):
        cost, A, B, E, gc, gb = blocks

        sc, sb = _block_scales(
            A,
            B
        )

        gradient = np.concatenate([
            sc * gc,
            (sb * gb).ravel()
        ])

        if np.max(np.abs(gradient)) < tolerance:
            success = True
            message = 'Gradient tolerance reached'
            break

        try:
            dc, db, predicted = _schur_step(
                A,
                B,
                E,
                gc,
                gb,
                damping
            )

            trial = update(
                state,
                dc,
                db
            )

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

            damping *= max(
                1 / 3,
                1 - (2 * ratio - 1)**3
            )

            damping = max(
                damping,
                1e-15
            )

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
