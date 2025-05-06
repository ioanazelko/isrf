
import numpy as np

def sample_circle(center, radius, q1, q2, theta):

    """
    Parameterize circles with given radii and centers, using 3D basis vectors spanning the circle plane.
    
    Parameters:
        center: array-like shape (3,) or (N,3), center(s) of the circle(s)
        radius: float or array-like shape (N,), radius/radii of the circle(s)
        q1: array-like shape (3,) or (N,3), normalized basis vector(s), perpendicular to circle normal
        q2: array-like shape (3,) or (N,3), normalized basis vector(s), perpendicular to circle normal and q1
        theta: float, numpy array shape (M,), or numpy array shape (N, M), angle(s) in radians to parameterize the circle

    Returns:
        np.array shape of (3,), (M,3) or (N,M,3) : Array of points on each circle parameterized by theta
    """

    #This ensures consistent handling of inputs (single 3D points or arrays of 3D points) without separate cases
    radius = np.asanyarray(radius)
    theta = np.asanyarray(theta)

    # Ensure proper broadcasting
    # The ... indicates "as many : as needed" to fully expand across remaining dimensions.
    center = center[..., np.newaxis, :]  # Shape (N,1,3)
    q1 = q1[..., np.newaxis, :]          # Shape (N,1,3)
    q2 = q2[..., np.newaxis, :]          # Shape (N,1,3)
    radius = radius[..., np.newaxis, np.newaxis]  # Shape (N,1,1)
    theta = theta[..., np.newaxis]    # Shape (1,M,1)
   
    # Parameterize circle(s)
    points = center + radius * (np.cos(theta)*q1 +
                                np.sin(theta)*q2)

    return points




def normalise_vectors(v):
    """Return unit vectors for a single 3-vector or an array of 3-vectors."""
    v = np.asarray(v)
    lengths = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, lengths, out=np.zeros_like(v), where=lengths>0)


def get_the_basis_vectors(p1,p2,center):
    """ 
    Parameters:
        p1: array-like shape (3,) or (N,3), point(s) on the circle(s)
        p2: array-like shape (3,) or (N,3), another point(s) on the circle(s)
        theta: float, numpy array shape (M,), or numpy array shape (N, M), angle(s) in radians to parameterize the circle
        center: array-like shape (3,), center(s) of the circle(s)
    Returns:
        radius: float or array-like shape (N,), radius/radii of the circle(s)
        q1: array-like shape (3,) or (N,3), normalized basis vector(s), perpendicular to circle normal
        q2: array-like shape (3,) or (N,3), normalized basis vector(s), perpendicular to circle normal and q1
        
    """
    p1 = np.asanyarray(p1)
    p2 = np.asanyarray(p2)
    center = np.asanyarray(center)
    ### Calculate the radius of the center of the circle
    radius =  np.linalg.norm(p1 - center,axis=-1)
    ### Calculate the normal vector of the plane defined by the circle
    n = np.cross(p1 - center,p2 - center,axis=-1)
    ### Normalize the normal vector
    n0 = normalise_vectors(n)
    q1 = normalise_vectors(p1-center)
    q2 = np.cross(n0,q1,axis=-1)

    #This ensures consistent handling of inputs (single 3D points or arrays of 3D points) without separate cases
    radius = np.asanyarray(radius)
    #theta2 = np.arccos (np.einsum('ij,ij->i', p1 - center, p2 - center)/radius**2)
    theta2 = np.arccos(np.sum((p1 - center)* (p2 - center),axis=-1)/radius**2)
    return radius, q1, q2, theta2


def parameterize_circle(p1,p2,theta,center = np.array([0,0,0])):
    """
    Parameterize circle(s) with given center(s) and 2D points on each circle
    Parameters:
        p1: array-like shape (3,) or (N,3), point(s) on the circle(s)
        p2: array-like shape (3,) or (N,3), another point(s) on the circle(s)
        theta: float, numpy array shape (M,), or numpy array shape (N, M), angle(s) in radians to parameterize the circle
        center: array-like shape (3,), center(s) of the circle(s)
    Returns:
        np.array shape of (3,), (M,3) or (N,M,3) : Array of points on each circle parameterized by theta
    """
   
    radius, q1, q2, theta2 = get_the_basis_vectors(p1,p2,center)
    circle_samples =sample_circle(center,radius, q1,q2,theta) 
    return circle_samples



def parameterize_arch(p1,p2,center = np.array([0,0,0]), npoints = 10):
    """
    Parameterize circle(s) with given center(s) and 2D points on each circle
    Parameters:
        p1: array-like shape (3,) or (N,3), point(s) on the circle(s)
        p2: array-like shape (3,) or (N,3), another point(s) on the circle(s)
        center: array-like shape (3,), center(s) of the circle(s)
    Returns:
        np.array shape of (3,), (M,3) or (N,M,3) : Array of points on each circle parameterized by theta
    """
   
    radius, q1, q2, theta2 = get_the_basis_vectors(p1,p2,center)
    print()
    theta_array = np.linspace(0,theta2,npoints).T
    circle_samples = sample_circle(center,radius, q1,q2,theta_array) 
    return circle_samples
    
