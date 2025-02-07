from numpy.testing import assert_allclose

from isrf.utils import galactic_to_cartesian
from isrf.example_mod import primes


def test_primes():
    assert primes(10) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def test_coordinates():
    print("Hello World")
    print(galactic_to_cartesian(0, 0, 1))
    x,y,z = galactic_to_cartesian(0.0, 0.0, 1.0)
    assert_allclose( [x,y,z], [1.0, 0.0, 0.0], rtol=1e-7, atol=1e-7)