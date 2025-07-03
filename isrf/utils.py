

from astropy.coordinates import SkyCoord, CartesianRepresentation
import astropy.units as u
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
    return np.array([ x, y, z])


def cartesian_to_galactic(x, y, z):
    """
    Convert Cartesian coordinates (x, y, z) to Galactic coordinates (l, b, distance).

    Parameters:
        x, y, z: Cartesian coordinates in parsecs.

    Returns:
        l, b: Galactic longitude and latitude in degrees.
        distance: Distance from Galactic center in parsecs.
    """
    cart_rep = CartesianRepresentation(x*u.pc, y*u.pc, z*u.pc)
    c = SkyCoord(cart_rep, frame='galactic')
    galactic = c.spherical

    l = galactic.lon.deg
    b = galactic.lat.deg
    distance = galactic.distance.value

    return l, b, distance
