import numpy as np
from typing import List, Tuple


def transform(points_in_old_object: List[Tuple[float, float, float]], matching_points_in_new_object: List[Tuple[int, Tuple[float, float, float]]]) -> List[Tuple[float, float, float]]:
    """
    Points_in_old_object will be transformed (translation + rotation) so that the distance matching_points_in_new_object to the points_in_old_object with matching index is minimized.
    Result of the transformation with minimum distance is returned.

    For non-ambiguous results, at least three points in points_in_old_object and three corresponding points in matching_points_in_new_object are required.
    """
    # Extract the indices and corresponding points
    indices = [idx for idx, _ in matching_points_in_new_object]
    old = np.array([points_in_old_object[idx] for idx in indices])
    new = np.array([pt for _, pt in matching_points_in_new_object])

    # Compute centroids
    centroid_old = np.mean(old, axis=0)
    centroid_new = np.mean(new, axis=0)

    # Center the points
    old_centered = old - centroid_old
    new_centered = new - centroid_new

    # Compute optimal rotation using SVD (Kabsch algorithm)
    H = old_centered.T @ new_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Ensure a right-handed coordinate system (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroid_new - R @ centroid_old

    # Apply transformation to all points
    transformed = [tuple((R @ np.array(p) + t).tolist()) for p in points_in_old_object]
    return transformed