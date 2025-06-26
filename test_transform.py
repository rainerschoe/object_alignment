import pytest
from transform import transform

@pytest.fixture
def identity_points():
    points_in_old_object = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 1.0)
    ]
    # Only three matching points required
    matching_points_in_new_object = [
        (0, (0.0, 0.0, 0.0)),
        (1, (1.0, 0.0, 0.0)),
        (2, (0.0, 1.0, 0.0))
    ]
    return points_in_old_object, matching_points_in_new_object

def test_identity_transformation(identity_points):
    points_in_old_object, matching_points_in_new_object = identity_points
    result = transform(points_in_old_object, matching_points_in_new_object)
    assert len(result) == len(points_in_old_object)
    for p, q in zip(result, points_in_old_object):
        assert pytest.approx(p[0], abs=1e-6) == q[0]
        assert pytest.approx(p[1], abs=1e-6) == q[1]
        assert pytest.approx(p[2], abs=1e-6) == q[2]
