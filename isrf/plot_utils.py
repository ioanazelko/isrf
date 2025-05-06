import healpy as hp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import isrf.circle
import isrf.utils

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


def get_verticies_pairs(pixel_index_array, radius, nside, nest):
    """
    Get the vertices of the HEALPix pixels in Cartesian coordinates.
    Parameters:
    -----------
    pixel_index_array: array-like
        Array of pixel indices
    radius: float
        Radius of the sphere
    nside: int
        HEALPix nside parameter
    nest: bool, default=False
        Nested pixel ordering if True, otherwise ring ordering
    Returns:
    --------
    p1_array: array
        Array of the first points of the edges
    p2_array: array
        Array of the second points of the edges
    """
    # Initialize empty arrays to store the points
    p1_array =  np.empty((0, 3), dtype=float)     # shape (0, 3)
    p2_array =  np.empty((0, 3), dtype=float)     # shape (0, 3)


    # Done with a for loop because healpy.boundaries doesn't take an array as input

    for pix in pixel_index_array:

        # Get the boundaries of the pixel, returns vertices in x, y, z cartesian coordinates, (3, N)
        vertices = radius*hp.boundaries(nside, pix, step=1, nest=nest) 

        # Transpose vertices to make it easier to work with individual points
        vertices_t = vertices.T  # shape (N, 3)
        
        # Create the edge pairs for this pixel

        p1 = np.array([vertices_t[0], vertices_t[1], vertices_t[2], vertices_t[3]])
        p2 = np.array([vertices_t[1], vertices_t[2], vertices_t[3], vertices_t[0]]) 

        p1_array = np.vstack((p1_array, p1))
        p2_array = np.vstack((p2_array, p2))
    return p1_array, p2_array


def add_spherical_shell(ax,radius =1):
    """
    Adds a wireframe along long lat at a given radius
    radius : float
        radius of the sphere, default 1
    ax : matplotlib axis to add the wireframe to

    """
    # Add spherical shell at radius 1
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius*np.cos(u)*np.sin(v)
    y = radius*np.sin(u)*np.sin(v)
    z = radius*np.cos(v)
    ax.plot_wireframe(x, y, z, color="k", alpha=0.5)

def add_xyz_labels(ax, unit = "[kpc]"):
    """
    Adds x,y,z labels to the axis
    unit : str, default "[kpc]"
    """

    ax.set_xlabel('X '+unit)
    ax.set_ylabel('Y '+unit)
    ax.set_zlabel('Z '+unit)



def add_ray(ax, origin, destination, color = 'blue'):
    """
    Add a ray to the plot
    ax: the axis to add the ray to
    origin: the origin of the ray, in the form of a 3D vector
    destination: the destination of the ray, in the form of a 3D vector
    color: the color of the ray, default is blue

    """

    x = [origin[0], destination[0]]
    y = [origin[1], destination[1]]
    z = [origin[2], destination[2]]

    ax.plot(x, y, z, color = color)



def add_fullsky_healpix_centers(ax, nside, color = 'r'):
    """
    Add fullsky HEALPix pixel centers to the given 3D plot for the given nside
    ax: 3D plot axis
    nside: int HEALPix nside parameter
    color: str, optional, default='r', color of the points
    """
    npix = hp.nside2npix(nside)

    # Get the coordinates of the centers of the HEALPix pixels
    l_c, b_c = hp.pix2ang(nside, np.arange(npix), lonlat=True)

    # Convert Galactic coordinates to Cartesian coordinates for 3D plotting
    x_c, y_c, z_c = utils.galactic_to_cartesian( l_c,b_c, 1)

    # Scatter plot for point visualization
    ax.scatter(x_c, y_c, z_c, c=color, s=3)  



def add_healpix_centers(ax, pixel_index_array, nside,nest=False, color = 'r'):
    """
    Add HEALPix pixel centers to the given 3D plot for the given nside
    ax: 3D plot axis
    pixel_index_array: array of healpixel indices
    nside: int HEALPix nside parameter
    nest: bool, optional, default=False, if True, assume NESTED pixel ordering, otherwise RING pixel ordering
    color: str, optional, default='r', color of the points
    """

    # Get the coordinates of the centers of the HEALPix pixels
    l_c, b_c = hp.pix2ang(nside, pixel_index_array, lonlat=True, nest=nest)

    # Convert Galactic coordinates to Cartesian coordinates for 3D plotting
    x_c, y_c, z_c = utils.galactic_to_cartesian( l_c,b_c, 1)

    # Scatter plot for point visualization
    ax.scatter(x_c, y_c, z_c, c=color, s=3) 


def add_scatter_points_array(ax, points, color='r', s=1 ):
    """ 
    Adds scatter points to the 3D plot
    points : array of shape (n,3)
    color : str, default 'r'
    s : float, default 1
    """
    ax.scatter(points[0], points[1], points[2], color=color, s= s)



def add_healpix_boundaries(ax, pixel_index_array, nside,nest=False, color = 'b' ):
    """
    Add the boundaries of a set of HEALPix pixels to a 3D plot.
    pixel_index_array: array of pixel indices
    nside: HEALPix nside parameter
    nest: nested pixel ordering if True, otherwise ring ordering
    color: color of the boundaries

    """

    # Done with a for loop because healpy.boundaries doesn't take an array as input
    for pix in pixel_index_array:

        # Get the boundaries of the pixel, returns vertices in x, y, z cartesian coordinates, (3, N)
        vertices = hp.boundaries(nside, pix, step=1, nest=nest)  

        add_scatter_points_array(ax, vertices, color, s=5) # Plot the vertices of the pixel with a given point size
        ### Plot the edges of the pixel
        for i in range(3):
            add_ray(ax, vertices[:,i], vertices[:,(i+1)], color=color)
        add_ray(ax, vertices[:,3], vertices[:,0], color=color)



def add_healpixels(ax, pixel_index_array, nside,nest=False, centers=True):
    """ 
    Add HEALPix pixels to a 3D plot.
    ax: Matplotlib axis object
    pixel_index_array: Array of HEALPix pixel indices
    nside: HEALPix nside parameter
    nest: True for NESTED pixel ordering, False for RING pixel ordering
    centers: True to plot pixel centers, False to leave them out
    """

    add_healpix_boundaries(ax, pixel_index_array=pixel_index_array, nside=nside, nest=nest)
    if centers == True:
        # Add the HEALPix pixel centers
        add_healpix_centers(ax, pixel_index_array=pixel_index_array, nside=nside, nest=nest)



######## Functions for plotting the arches of the HEALPix pixels ########



def plot_test_circle():
    # Example points
    p1, p2, center = get_example_points()
    theta = np.linspace(0,2*np.pi,10)

    circle_samples = circle.parameterize_circle(p1, p2, theta, center = center)


    # Plot points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([p1[0], p2[0]],[p1[1], p2[1]], [p1[2],p2[2]], color='r', s=100)
    ax.scatter(center[0],center[1],center[2], color='b')
    ax.scatter(circle_samples[:,0],circle_samples[:,1],circle_samples[:,2], color='g')
    # ax.scatter(circle_samples2[:,0],circle_samples2[:,1],circle_samples2[:,2], color='orchid')
    # ax.scatter(points[:,0], points[:,1], points[:,2], color='r')

    add_xyz_labels(ax)
    ax.set_title('3D Galactic Visualization')

    # circle_samples = circle.parameterize_circle(np.array([p1,p1]), np.array([p2,p2]), theta, center = np.array([center,center]))

def plot_test_arch():
    # Example points
    p1, p2, center = get_example_points()

    arch_samples = circle.parameterize_arch(p1, p2,  center = center)

    arch_samples2 = circle.parameterize_arch(p2,p1,center = center, npoints = 10)

      # Plot points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([p1[0], p2[0]],[p1[1], p2[1]], [p1[2],p2[2]], color='r', s=100)
    ax.scatter(center[0],center[1],center[2], color='b')
    ax.scatter(arch_samples[:,0],arch_samples[:,1],arch_samples[:,2], color='g', s=10,alpha=1)
    ax.scatter(arch_samples2[:,0],arch_samples2[:,1],arch_samples2[:,2], color='orchid', s=30,alpha=0.5)

    add_xyz_labels(ax)
    ax.set_title('3D Galactic Visualization')
    plt.show()


def add_healpix_arches_optimized(ax, pixel_index_array, nside, nest=False, radius = 1, color='orchid', npoints=10, s=5):
    """
    Add the arches for the boundaries of a set of HEALPix pixels to a 3D plot.
    Note - this doubles the arches, as the sides are double counted
    Parameters:
    -----------
    ax: matplotlib 3D axis
        The axis to plot on
    pixel_index_array: array-like
        Array of pixel indices
    nside: int
        HEALPix nside parameter
    nest: bool, default=False
        Nested pixel ordering if True, otherwise ring ordering
    color: str, default='orchid'
        Color of the boundaries
    npoints: int, default=10
        Number of points to sample along each arch
    s: float, default=5
        Size of the scatter points
    """
    p1_array, p2_array = get_verticies_pairs(pixel_index_array, radius, nside, nest)

    arch_samples = circle.parameterize_arch(p1_array, p2_array,npoints=npoints)

    arch_samples = arch_samples.reshape(-1, 3)
    ax.scatter(arch_samples[:,0],arch_samples[:,1],arch_samples[:,2], color=color, s=s)
