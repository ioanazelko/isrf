import numpy as np

def direction_star_to_ISRF_point(isrf_pos, star_pos):
    """
    Calculate the direction from the star to the ISRF point.
    input:
        isrf_pos - 3D cartesian vector, the position of the ISRF point
        star_pos - 3D cartesian vector, the position of the star
    output:
        direction - 3D cartesian vector, the direction from the star to the ISRF point
        direction_abs - scalar, the absolute value of the direction vector
        star_dot_direction - scalar, the dot product of the star position and the direction vector
    """
    direction = isrf_pos - star_pos
    direction_abs = np.dot(direction, direction) ** 0.5
    star_dot_direction = np.dot(star_pos, direction)
    return direction, direction_abs, star_dot_direction

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