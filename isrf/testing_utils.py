import numpy as np


def get_example_points(center = np.array([1,0,0])):
    # Example galactic coordinates (longitude, latitude, radius)
    coordinates = [
        (-77.016, -12.033, 1),   # Center of the galaxy
        (110.43, -7.78, 1) # Some point in space
    ]

    # Convert to Cartesian coordinates for plotting
    points = np.array([utils.galactic_to_cartesian(*coord) for coord in coordinates])
    p1 = points[0]+center
    p2 = points[1]+center

    return p1,p2, center

def get_example_stars(single_star=False):
    """
    Get example star positions in galactic coordinates.
    Returns:
    - star_positions: array-like, shape (n_stars, 3), galactic coordinates of the stars.
    """
    # Example star positions in galactic coordinates (l, b, r)
    if single_star:
        star1 = np.array([0, 0, 1])  # Example star position in galactic coordinates
        return np.array([star1])  # Return only the single star position
    # Multiple star positions
    else:

        star1 = np.array([0, 0, 1])

        star2 = np.array([10, 20, 1])

        star3 = np.array([20, 30, 2])
        star_positions = np.array([star1, star2, star3])
        return star_positions



def get_example_ISRF(single_position=False):
    """
    Get example ISRF positions in galactic coordinates.
    Returns:
    - numpy array of shape (n_isrf, 3) with (l, b, r) coordinates.
    """
    # Example ISRF positions in galactic coordinates (l, b, r)

    if single_position:
        return np.array([[30, 10, 0.5]])
    # Multiple ISRF positions
    else:
        return np.array([
            [30, 10, 0.5],  # ISRF at some point in the galaxy
            [-20, -20, 2]  # Another ISRF position    
            ])
