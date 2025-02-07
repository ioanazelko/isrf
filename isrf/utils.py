import numpy as np

# Function to convert galactic coordinates to Cartesian coordinates
def galactic_to_cartesian(l, b, r):
    """
    Convert galactic coordinates to Cartesian coordinates.
    in: l, b in degrees, r can be any units
    out: x, y, z units of r
    
    """
    l, b = np.radians(l), np.radians(b)
    x = r * np.cos(b) * np.cos(l)
    y = r * np.cos(b) * np.sin(l)
    z = r * np.sin(b)
    return x, y, z


