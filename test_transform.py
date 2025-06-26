import pytest
from transform import transform

@pytest.fixture
def points_in_old_object():
    return [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0)
    ]

def test_identity_transformation(points_in_old_object):
    matching_points_in_new_object = [
        (0, (0.0, 0.0, 0.0)),
        (1, (1.0, 0.0, 0.0)),
        (2, (0.0, 1.0, 0.0))
    ]
    result = transform(points_in_old_object, matching_points_in_new_object)
    assert len(result) == len(points_in_old_object)
    for new_point, old_point in zip(result, points_in_old_object):
        assert pytest.approx(new_point[0]) == old_point[0]
        assert pytest.approx(new_point[1]) == old_point[1]
        assert pytest.approx(new_point[2]) == old_point[2]

def test_x_shift_transformation(points_in_old_object):
    matching_points_in_new_object = [
        (0, (1.0, 0.0, 0.0)),
        (1, (2.0, 0.0, 0.0)),
        (2, (1.0, 1.0, 0.0))
    ]
    result = transform(points_in_old_object, matching_points_in_new_object)
    assert len(result) == len(points_in_old_object)
    for new_point, old_point in zip(result, points_in_old_object):
        assert pytest.approx(new_point[0]) == old_point[0] + 1.0
        assert pytest.approx(new_point[1]) == old_point[1]
        assert pytest.approx(new_point[2]) == old_point[2]

def test_rotation_z_90_degrees(points_in_old_object):
    # Rotate the first three points 90 degrees counterclockwise around the z-axis
    # (x, y, z) -> (-y, x, z)
    matching_points_in_new_object = [
        (0, (0.0, 0.0, 0.0)),      # origin stays the same
        (1, (0.0, 1.0, 0.0)),      # (1,0,0) -> (0,1,0)
        (2, (-1.0, 0.0, 0.0))      # (0,1,0) -> (-1,0,0)
    ]
    result = transform(points_in_old_object, matching_points_in_new_object)
    assert len(result) == len(points_in_old_object)
    for new_point, old_point in zip(result, points_in_old_object):
        # Apply the expected rotation to old_point
        expected = (-old_point[1], old_point[0], old_point[2])
        assert pytest.approx(new_point[0]) == expected[0]
        assert pytest.approx(new_point[1]) == expected[1]
        assert pytest.approx(new_point[2]) == expected[2]

def test_rotation_z_90_then_translation(points_in_old_object):
    # First rotate 90 deg CCW around z, then translate by (1,1,1)
    # (x, y, z) -> (-y, x, z) + (1, 1, 1)
    matching_points_in_new_object = [
        (0, (1.0, 1.0, 1.0)),      # (0,0,0) -> (1,1,1)
        (1, (1.0, 2.0, 1.0)),      # (1,0,0) -> (0,1,0) -> (1,2,1)
        (2, (0.0, 1.0, 1.0))       # (0,1,0) -> (-1,0,0) -> (0,1,1)
    ]
    result = transform(points_in_old_object, matching_points_in_new_object)
    assert len(result) == len(points_in_old_object)
    for new_point, old_point in zip(result, points_in_old_object):
        expected = (-old_point[1] + 1, old_point[0] + 1, old_point[2] + 1)
        assert pytest.approx(new_point[0]) == expected[0]
        assert pytest.approx(new_point[1]) == expected[1]
        assert pytest.approx(new_point[2]) == expected[2]

def test_identity_with_noise(points_in_old_object):
    # matching points are close to the originals, but with small noise
    matching_points_in_new_object = [
        (0, (0.01, -0.01, 0.0)),   # small noise
        (1, (1.02, 0.01, -0.01)), # small noise
        (2, (-0.01, 1.01, 0.02))  # small noise
    ]
    result = transform(points_in_old_object, matching_points_in_new_object)
    assert len(result) == len(points_in_old_object)
    for new_point, old_point in zip(result, points_in_old_object):
        # The transformation should still be close to the original points
        assert pytest.approx(new_point[0], abs=0.03) == old_point[0]
        assert pytest.approx(new_point[1], abs=0.03) == old_point[1]
        assert pytest.approx(new_point[2], abs=0.03) == old_point[2]

def test_identity_with_canceling_noise(points_in_old_object):
    # Add noise to matching points, but the centroid remains unchanged
    matching_points_in_new_object = [
        (0, (0.02, -0.02, 0.0)),   # +0.02, -0.02
        (1, (0.98, 0.02, 0.0)),    # -0.02, +0.02
        (2, (0.0, 1.0, 0.0))       # no noise
    ]
    # The centroid of the matching points is still (0.333..., 0.333..., 0.0)
    result = transform(points_in_old_object, matching_points_in_new_object)
    assert len(result) == len(points_in_old_object)
    for new_point, old_point in zip(result, points_in_old_object):
        # The transformation should still be very close to the original points
        assert pytest.approx(new_point[0], abs=0.03) == old_point[0]
        assert pytest.approx(new_point[1], abs=0.03) == old_point[1]
        assert pytest.approx(new_point[2], abs=0.03) == old_point[2]

def test_identity_with_canceling_noise(points_in_old_object):
    # Add noise to matching points. The noise is such, that the optimal solution is the identity transformation.
    # (We do not allow scaling, so the minimum distance is achieved by not changing the points)
    matching_points_in_new_object = [
        (0, (-4, -4, 0.0)),
        (1, (5.0, 0.0, 0.0)),
        (2, (0.0, 5.0, 0.0))
    ]
    result = transform(points_in_old_object, matching_points_in_new_object)
    assert len(result) == len(points_in_old_object)
    for new_point, old_point in zip(result, points_in_old_object):
        assert pytest.approx(new_point[0]) == old_point[0]
        assert pytest.approx(new_point[1]) == old_point[1]
        assert pytest.approx(new_point[2]) == old_point[2]
