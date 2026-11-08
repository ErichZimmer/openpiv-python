import numpy as np

from .._epipolar_utils import _line_intersect, _multi_line_intersect


__all__ = [
    'line_intersect',
    'multi_line_intersect'
]


def line_intersect(
    cam_1: "pinhole_model.camera",
    cam_2: "pinhole_model.camera",
    img_points_1: np.ndarray,
    img_points_2: np.ndarray
):
    """Calculate where two rays intersect.

    Using two cameras, calculate the world coordinates where two rays
    intersect. This is done through an analytical solution based on
    direction vectors and camera origins/translations.

    Parameters
    ----------
    cam_1, cam_2 : pinhole_model.camera
        An instance of a Pinhole camera.
    img_points_1, img_points_2 : np.ndarray
        Image coordinates stored in a ndarray structured like [x, y]'.

    Returns
    -------
    coords : np.ndarray
        The world coordinate that is nearest to the two rays intersecting.
    dist : float
        The minimum distance between the two rays.

    Raises
    ------
    ValueError
        If camera parameters are invalid, camera dtypes do not match,
        or image coordinates cannot be converted to compatible arrays
        with two rows.
    RuntimeError
        If Brown distortion inversion fails when constructing the rays.
    np.linalg.LinAlgError
        If reversing tilt or solving a Brown inversion step encounters
        a singular matrix.

    """
    cam_1._check_parameters()
    cam_2._check_parameters()

    dtype1 = cam_1.dtype
    dtype2 = cam_2.dtype

    # all cameras should have the same dtype
    if dtype1 != dtype2:
        raise ValueError('Dtypes between camera structures must match')

    img_points_1 = np.array(img_points_1, dtype=dtype1)
    img_points_2 = np.array(img_points_2, dtype=dtype2)

    r1 = cam_1._get_inverse_vector(
        img_points_1
    )

    r2 = cam_2._get_inverse_vector(
        img_points_2
    )

    t1 = cam_1.translation
    t2 = cam_2.translation

    # TODO: move extention of dimensions into _line_intersect
    return _line_intersect(
        t1[:, np.newaxis],
        r1,
        t2[:, np.newaxis],
        r2
    )


def multi_line_intersect(
    cameras: list,
    img_points: list,
    iterations: int=5
):
    """Triangulate world points from multiple camera observations.

    Estimate the intersection of at least two rays using least squares.
    Optionally refine each world point using Gauss-Newton iterations
    that minimize its pixel reprojection residuals across the cameras.

    Parameters
    ----------
    cameras : list of pinhole_model.camera
        Cameras observing the points. All cameras must use the same dtype.
    img_points : list of np.ndarray
        Image coordinates for each camera, with shape (2, N) per array.
        Corresponding columns must observe the same world point.
    iterations : int, optional
        Number of Gauss-Newton refinement steps. Use zero for the initial
        ray-intersection estimate only. Default is 5.

    Returns
    -------
    coords : np.ndarray
        Triangulated world coordinates with shape (3, N).

    Raises
    ------
    ValueError
        If fewer than two cameras are supplied, camera and image counts
        differ, camera parameters are invalid, camera dtypes do not
        match, or image coordinates cannot be reshaped into two rows.
    RuntimeError
        If Brown distortion inversion fails when constructing the rays.
    np.linalg.LinAlgError
        If a least-squares solve fails to converge.

    """
    n_cameras = len(cameras)

    if n_cameras < 2:
        raise ValueError('Triangulation needs at least two cameras')

    dtype = cameras[0].dtype
    images = []

    for image in img_points:
        coordinates = np.asarray(image, dtype=dtype)
        coordinates = coordinates.reshape(2, -1)
        images.append(coordinates)

    n_images = len(images)

    if n_cameras != n_images:
        msg = (
            f'Camera - image size mismatch. Got {n_cameras} cameras and '
            f'{n_images} images'
        )
        raise ValueError(msg)

    for camera in cameras:
        camera._check_parameters()

    for camera in cameras[1:]:
        if camera.dtype != dtype:
            raise ValueError('Dtypes between camera structures must match')

    # Construct the world-space ray for each image observation.
    lines = []

    for camera, image in zip(cameras, images):
        points = np.array(image, dtype=dtype)
        directions = camera._get_inverse_vector(
            points
        )
        origin = camera.translation

        lines.append([origin, directions])

    object_points = _multi_line_intersect(
        lines,
        dtype
    )

    n_points = object_points.shape[1]
    n_residuals = 2 * n_cameras

    # Like in the Soloff model, use Gauss-Newton to minimize remaining errors
    for _ in range(iterations):
        # Each point has two residuals per camera and three unknowns.
        projection_jacobian = np.empty((n_points, n_residuals, 3))
        residual = np.empty((n_points, n_residuals))

        for i, camera in enumerate(cameras):
            projected, _, point_jacobian = camera.project_jacobian(
                object_points
            )
            section = slice(2*i, 2*i+2)
            pixel_residual = projected - images[i]

            projection_jacobian[:, section, :] = point_jacobian
            residual[:, section] = pixel_residual.T

        for j in range(n_points):
            step, _, _, _ = np.linalg.lstsq(
                projection_jacobian[j],
                -residual[j],
                rcond=None
            )
            object_points[:, j] += step

    return object_points