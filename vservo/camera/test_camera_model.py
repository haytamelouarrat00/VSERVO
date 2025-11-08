import numpy as np
import pytest
from vservo.camera.camera import CameraModel  # adjust import path if needed


@pytest.fixture
def cam() -> CameraModel:
    cam = CameraModel()
    cam.set_pose(0.0, 0.0, -1.5, 0.0, 0.0, 0.0)
    return cam


def test_homogeneous_transformation_matrix_shape(cam):
    T = cam.homogenous_transformation_matrix()
    assert T.shape == (4, 4)
    assert np.allclose(T[3], [0, 0, 0, 1])


def test_rotation_matrix_shape(cam):
    R = cam.get_rotation_matrix()
    assert R.shape == (3, 3)
    assert np.allclose(np.dot(R, R.T), np.eye(3), atol=1e-8)


def test_translation_vector_shape(cam):
    cam.homogenous_transformation_matrix()
    assert cam.T.shape == (3, 1)


def test_world_to_camera_single_point(cam):
    T = cam.homogenous_transformation_matrix()
    pt_world = np.array([0.0, 0.0, 0.5])
    pt_cam = cam.world_2_camera(T, pt_world)
    assert pt_cam.shape == (3,)
    assert np.isfinite(pt_cam).all()


def test_world_to_camera_multiple_points(cam):
    T = cam.homogenous_transformation_matrix()
    pts_world = np.array([[0.0, 0.0, 0.5], [0.1, 0.0, 0.5]])
    pts_cam = cam.world_2_camera(T, pts_world)
    assert pts_cam.shape == (2, 3)
    assert np.isfinite(pts_cam).all()


def test_project_single_point(cam):
    pt_cam = np.array([0.0, 0.0, 1.0])
    uv = cam.project(pt_cam)
    assert uv.shape == (2,)
    assert np.isfinite(uv).all()


def test_project_multiple_points(cam):
    pts_cam = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]])
    uvs = cam.project(pts_cam)
    assert uvs.shape == (3, 2)
    assert np.isfinite(uvs).all()


def test_project_raises_on_zero_depth(cam):
    pt_cam = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        cam.project(pt_cam)


def test_project_points_pipeline(cam):
    # Define 4 test points in world coordinates
    pts_world = np.array(
        [
            [-0.1, -0.1, 0.5],
            [0.1, -0.1, 0.5],
            [0.1, 0.1, 0.5],
            [-0.1, 0.1, 0.5],
        ]
    )
    pts_2d, pts_cam = cam.project_points(pts_world)

    assert pts_cam.shape == (4, 3)
    assert pts_2d.shape == (4, 2)
    assert np.isfinite(pts_2d).all()
    assert np.isfinite(pts_cam).all()


def test_invalid_inputs_raise(cam):
    T = np.eye(4)
    with pytest.raises(ValueError):
        cam.world_2_camera(np.eye(3), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        cam.world_2_camera(T, np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        cam.project(np.array([1.0, 2.0]))


def test_set_pose_updates_fields(cam):
    cam.set_pose(1.0, 2.0, 3.0, 0.1, 0.2, 0.3)
    assert np.isclose(cam.X, 1.0)
    assert np.isclose(cam.rZ, 0.3)
