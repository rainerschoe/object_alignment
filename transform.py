import numpy as np
from typing import List, Tuple


def transform(points_in_old_object: List[Tuple[float,float, float]], matching_points_in_new_object: List[Tuple[int, Tuple[float,float, float]]]) -> List[Tuple[float, float, float]]:
    """
    points_in_old_object will be transformed (translation + rotation) so that the distance matching_points_in_new_object to the points_in_old_object with matching index is minimized.
    Result of the transformation with minimum distance is returned.

    """
    return points_in_old_object