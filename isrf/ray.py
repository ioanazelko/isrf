import numpy as np

def direction_star_to_ISRF_point(isrf_pos, star_pos):
    """
    Calculate the direction from the star to the ISRF point.
    input:
        isrf_pos - 3D cartesian vector, the position of the ISRF point
        star_pos - 3D cartesian vector, the position of the star
    output:
        direction - 3D cartesian vector, the direction from the star to the ISRF point
    """
    isrf_pos = np.asarray(isrf_pos)  # Shape: (M, 3)
    star_pos = np.asarray(star_pos)  # Shape: (N, 3)

    #direction = isrf_pos - star_pos  
    # All pairwise directions (broadcasting)
    direction = isrf_pos[:, np.newaxis, :] - star_pos[np.newaxis, :, :] # Result shape: (M, N, 3) - M ISRF positions × N stars × 3 coordinates
    return direction

def direction_abs(direction):
    """ This function returns the absolute value of the direction vector.
    
    input:
        direction - 3D cartesian vector, the direction of the ray,(M, N, 3)
    output:
        direction_abs - scalar, the absolute value of the direction vector
    """
    return np.linalg.norm(direction, axis=2)  # Shape: (M, N) - absolute value of the direction vector


def stars_dot_directions(stars, directions):
    """
    Calculate dot product between stars and directions.
    
    Parameters:
    -----------
    stars : array_like, shape (N, 3)
        Star position vectors
    directions : array_like, shape (M, N, 3)
        Direction vectors from stars to ISRF positions
    
    Returns:
    --------
    dot_products : ndarray, shape (M, N)
        Dot products for each ISRF position and star combination
    """
    stars = np.asarray(stars)
    directions = np.asarray(directions)
    
    # Validate shapes
    if stars.ndim != 2 or stars.shape[1] != 3:
        raise ValueError(f"Stars must have shape (N, 3), got {stars.shape}")
    
    if directions.ndim != 3 or directions.shape[2] != 3:
        raise ValueError(f"Directions must have shape (M, N, 3), got {directions.shape}")
    
    if stars.shape[0] != directions.shape[1]:
        raise ValueError(f"Number of stars must match: stars={stars.shape[0]}, directions={directions.shape[1]}")
    
    # Method 1: Using einsum (most efficient)
    dot_products = np.einsum('ij,mij->mi', stars, directions)
    
    return dot_products


def ray_linear_parameterization(star_pos,direction, t):
    """ This function returns the position of a point on the ray defined by star_pos and direction at parameter t.
        The ray is defined as the set of points {star_pos + t*direction | t in R}.

    input:
        star_pos - 3D cartesian vector, the position of the star
        direction -  3D vector, the direction of the ray
        t - scalar, the parameter
    output:
        position - 3D cartesian vector, the position of the point on the ray at parameter t
    
    """
    return star_pos + t*direction


def distance_on_ray(star_pos,direction, t):
    """    This function returns the distance from the star to the point on the ray defined by star_pos and direction at parameter t.
    The ray is defined as the set of points {star_pos + t*direction | t in R}.
    input: 
        star_pos - 3D cartesian vector, the position of the star
        direction -  3D vector, the direction of the ray
        t - scalar, the parameter of the ray parameterization
    output:
        distance - euclidean distance from the star to the point on the ray at parameter t, scalar
    """
    x0,y0,z0 = star_pos
    a,b,c = direction
    distance = ((x0 + t*a)**2 + (y0 + t*b)**2 + (z0 + t*c)**2)**0.5  
    return distance



##### sample the ray of light, get the angles for each point, and then use ang2pix to get the pixel index
def sample_ray_of_light(star_pos, direction, t_min, t_max, num_points):
    """
    Sample the ray of light from star_pos in the direction of direction.
    """
    t_values = np.linspace(t_min, t_max, num_points)
    points = np.array([ray_linear_parameterization(star_pos, direction, t) for t in t_values])
    return points



def get_pixel_indices(points, nside):
    """
    Get the pixel indices for the given points.
    """
    theta, phi, distance = utils.cartesian_to_galactic(points[:,0], points[:,1], points[:,2])
    pixel_indices = hp.ang2pix(nside, theta, phi)
    return pixel_indices
def get_pixel_indices_from_ray(star_pos, direction, t_min, t_max, num_points, nside):
    """
    Sample the ray of light and get the pixel indices.
    """
    points = sample_ray_of_light(star_pos, direction, t_min, t_max, num_points)
    pixel_indices = get_pixel_indices(points, nside)
    return pixel_indices


#### check that the ray is not in the plane defined by any of the healpix edges
#### Find the crossing positions of the ray with a sphere of radius d

def find_crossing_positions(star_r,star_dot_direction,direction_abs,d):
    """ 
    
    This function calculates the crossing positions of a ray defined by star_r and star_dot_direction
    with a sphere of radius d. It returns the two possible intersection parameters t1 and t2.
    input:
        star_r - scalar, the distance of the star from the origin
        star_dot_direction - scalar, the dot product of the star position and the direction vector
        direction_abs - scalar, the absolute value of the direction vector
        d - scalar, the radius of the sphere
    output:
        t1, t2 - scalars, the two possible intersection parameters of the star with the sphere defined by the radius d.
        If there is no intersection, t1 and t2 will be NaN.
    """

      # Convert inputs to numpy arrays for vectorized operations
    star_r = np.asarray(star_r)
    star_dot_direction = np.asarray(star_dot_direction)
    direction_abs = np.asarray(direction_abs)
    d = np.asarray(d)
    

    star_r = star_r[..., np.newaxis]
    star_dot_direction = star_dot_direction[..., np.newaxis]
    direction_abs = direction_abs[..., np.newaxis]

    a = direction_abs**2
    b = 2*star_dot_direction
    c = star_r**2-d**2
    t1 = (-b + (b**2 - 4*a*c)**0.5)/(2*a)
    # We may not need to calculate t2 because it may end up mostly negative. need to investigate all cases
    t2 = (-b - (b**2 - 4*a*c)**0.5)/(2*a)
    mask1 = (t1 >= 0) & (t1 <= 1)
    mask2 = (t2 >= 0) & (t2 <= 1)
    t1 = np.where(mask1, t1, np.nan)  # Set t1 to NaN where the condition is not met
    t2 = np.where(mask2, t2, np.nan)  # Set t2 to NaN where the condition is not met 
    return t1.squeeze(), t2.squeeze()






