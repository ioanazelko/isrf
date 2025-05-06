from numpy.testing import assert_allclose
import numpy as np
from isrf.utils import galactic_to_cartesian
from isrf.example_mod import primes
import isrf.circle as circle
import isrf.utils as utils

def test_primes():
    assert primes(10) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def test_coordinates():
    print("Hello World")
    print(galactic_to_cartesian(0, 0, 1))
    coords = galactic_to_cartesian(0.0, 0.0, 1.0)
    assert_allclose( coords, np.array([1.0, 0.0, 0.0]), rtol=1e-7, atol=1e-7)


def test_circle_implementation():

    """ This needs to be added some comparison statements"""

    # Example galactic coordinates (longitude, latitude, radius)
    coordinates = [
    (-77.016, -12.033, 1),   # Center of the galaxy
    (110.43, -7.78, 1) # Some point in space
    ]

    # Convert to Cartesian coordinates for plotting
    points = np.array([utils.galactic_to_cartesian(*coord) for coord in coordinates])

    center = np.array([1,0,0])
    p1 = points[0]+center
    p2 = points[1]+center

    theta_array = np.linspace(0,2*np.pi,10)
    different_theta_for_each_circle = np.array([np.linspace(0,2*np.pi,10),np.linspace(0,2*np.pi,10)])

    theta = 0.

    # test a single circle with a scalar theta

    circle_samples = circle.parameterize_circle(p1, p2, theta, center = center)
    print( circle_samples)
    # test a single circle with an array of thetas

    circle_samples = circle.parameterize_circle(p1, p2, theta, center = center)
    print( circle_samples)

    # test multiple circles with a scalar theta

    circle_samples = circle.parameterize_circle(np.array([p1,p1]), np.array([p2,p2]), theta, center = center)
    print( circle_samples)

    # test multiple circles with different centers with a single theta
    print( circle_samples)

    circle_samples = circle.parameterize_circle(np.array([p1,p1]), np.array([p2,p2]), theta, center = np.array([center,center]))

    # test multiple circles with different centers with an array of thetas
    print( circle_samples)

    circle_samples = circle.parameterize_circle(np.array([p1,p1]), np.array([p2,p2]), theta_array, center = np.array([center,center]))
    print( circle_samples)


    # test multiple circles with different centers with different arrays of thetas

    circle_samples = circle.parameterize_circle(np.array([p1,p1]), np.array([p2,p2]), different_theta_for_each_circle, center = np.array([center,center]))
    print( circle_samples)


def test_arches_implementation():

    """ This needs to be added some comparison statements"""

    # Example galactic coordinates (longitude, latitude, radius)
    coordinates = [
    (-77.016, -12.033, 1),   # Center of the galaxy
    (110.43, -7.78, 1) # Some point in space
    ]

    # Convert to Cartesian coordinates for plotting
    points = np.array([utils.galactic_to_cartesian(*coord) for coord in coordinates])

    center = np.array([1,0,0])
    p1 = points[0]+center
    p2 = points[1]+center

    theta_array = np.linspace(0,2*np.pi,10)
    different_theta_for_each_circle = np.array([np.linspace(0,2*np.pi,10),np.linspace(0,2*np.pi,10)])

    theta = 0.

    # test a single arch
    print(p1)
    radius =  np.linalg.norm(p1 - center)

    arch_samples = circle.parameterize_arch(p1, p2,  center = center)
    print(arch_samples)
    # test multiple arches

    arch_samples = circle.parameterize_arch(np.array([p1,p1]), np.array([p2,p2]), center = center)
    print(arch_samples)

    # # test multiple arches with different centers with different arrays of thetas

    circle_samples = circle.parameterize_arch(np.array([p1,p1]), np.array([p2,p2]), center = np.array([center,center]))

