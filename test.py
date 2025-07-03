import healpy as hp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import isrf.circle as circle
import isrf.plot_utils as plot_utils
import isrf.ray as ray
import isrf.stars as stars
import isrf.testing_utils as testing_utils
import isrf.utils as utils




def create_test_data():

    star_positions_galactic =  testing_utils.get_example_stars(single_star=True)
    isrf_positions_galactic = testing_utils.get_example_ISRF(single_position=True)

    star_field = stars.StarField(star_positions_galactic)
    ax = star_field.plot_stars()
    star_field.get_isrf_locations(isrf_positions_galactic)
    star_field.plot_isrf(ax)

    star_field.get_isrf_directions()
    # print("Direction:", star_field.directions)
    star_positions = star_field.stars_cartesian
    isrf_positions = star_field.isrf_cartesian
    print(star_positions_galactic, star_positions_galactic + star_field.directions[0] - isrf_positions_galactic[0])
    print("Directions to ISRF 0", star_field.directions[0])


    print("Star +plus direction:", star_positions + star_field.directions[0])
    print("ISRF position 0:", isrf_positions[0])

    star_field.get_the_shell_crossections()

    return star_positions, star_field.directions

# Test the functions
if __name__ == "__main__":
    # Create test data
    stars, dirs = create_test_data()
    
   