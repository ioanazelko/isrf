import matplotlib.pyplot as plt
import numpy as np


import isrf.dust_maps as dust_maps

import isrf.plot_utils as plot_utils
import isrf.ray as ray
import isrf.testing_utils as testing_utils
import isrf.utils as utils



class StarField:
    def __init__(self, positions):
        """
        Initialize a field of stars.

        Parameters:
        - positions: array-like, shape (3,) or (n_stars, 3), galactic coordinates of the star(s), 
          where each star is represented by (l, b, r).
        """
        self.stars_galactic = np.asanyarray(positions)
        self.stars_cartesian = utils.galactic_to_cartesian(self.stars_galactic[:, 0], self.stars_galactic[:, 1], self.stars_galactic[:, 2]).T #     Shape: (n_stars, 3)

    def plot_stars(self):

        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.set_box_aspect([1, 1, 1])
        plot_utils.add_stars(ax, self.stars_cartesian)
        ax.legend()
        return ax

    def get_isrf_locations(self, isrf):
        """
        Get the locations where to calculate the ISRF.
        This method converts the galactic coordinates of the ISRF to Cartesian coordinates.

        Parameters:
        - isrf: numpy array, ISRF object containing the positions of ISRF, 
          where each position is represented by (l, b, r).
        """
        isrf = np.asanyarray(isrf)
        isrf_cartesian = utils.galactic_to_cartesian(isrf[:, 0], isrf[:, 1], isrf[:, 2]).T  # Shape: (n_isrf, 3)
        self.isrf_cartesian = isrf_cartesian
        
    
    def plot_isrf(self, ax, alpha=0.5):
        """
        Plot the ISRF locations in the star field.
        This method assumes that the ISRF positions have been set using `get_isrf_locations`.
        """
        if not hasattr(self, 'isrf_cartesian'):
            raise ValueError("ISRF positions have not been set. Call get_ISRF_locations first.")
        plot_utils.add_scatter_points_array(ax, self.isrf_cartesian.T, label='ISRF', color='green', s=10, alpha=alpha )
        return ax

    def get_isrf_directions(self):
        """
        Calculate the direction from each star to the ISRF point.
        This method computes the direction vector, its absolute value, and the dot product of the star position and the direction vector.

        Returns:
        - directions: array-like, shape (n_stars, 3), direction vectors from stars to ISRF points.
        - directions_abs: array-like, shape (n_stars,), absolute values of the direction vectors.
        - star_dot_directions: array-like, shape (n_stars,), dot products of star positions and direction vectors.
        """
        if not hasattr(self, 'isrf_cartesian'):
            raise ValueError("ISRF positions have not been set. Call get_ISRF_locations first.")
        
        self.directions = ray.direction_star_to_ISRF_point(self.isrf_cartesian, self.stars_cartesian)
    
    def get_directions_abs(self):
        """
        Calculate the absolute values of the direction vectors.
        
        Returns:
        - directions_abs: array-like, shape (n_stars,), absolute values of the direction vectors.
        """
        if not hasattr(self, 'directions'):
            raise ValueError("Directions have not been calculated. Call get_isrf_directions first.")
        
        self.directions_abs = ray.direction_abs(self.directions)
    def get_stars_dot_directions(self):
        """
        Calculate the dot product of the star positions and the direction vectors.
        
        Returns:
        - stars_dot_directions: array-like, shape (n_stars,), dot products of star positions and direction vectors.
        """
        if not hasattr(self, 'directions'):
            raise ValueError("Directions have not been calculated. Call get_isrf_directions first.")
        
        self.stars_dot_directions =  ray.stars_dot_directions(self.stars_cartesian, self.directions)

    def trace_ray(star, isrf,num_points=100):
        """
        Trace a ray from the star to the ISRF point.

        Parameters
        ----------
        star : tuple
            The coordinates of the star in galactic coordinates (l, b, r).
        isrf : tuple
            The coordinates of the ISRF point in galactic coordinates (l, b, r).
        num_points : int
            The number of points to sample along the ray.
        Returns
        -------
        -------
        points : array
            The sampled points along the ray in cartesian coordinates.

        """
        star_pos = utils.galactic_to_cartesian(star[0], star[1], star[2])
        isrf_pos = utils.galactic_to_cartesian(isrf[0], isrf[1], isrf[2])
        
        # Calculate the direction from the star to the ISRF point
        direction, direction_abs, star_dot_direction = ray.direction_star_to_ISRF_point(isrf_pos, star_pos)
        
        points = ray.sample_ray_of_light(star_pos, direction, t_min=0, t_max=1, num_points=num_points)

        return points

    def get_the_dust_map_distances(self):
        dm = dust_maps.DustMaps()
        self.dust_map_distances = dm.get_dust_map_distances()

    def get_the_shell_crossections(self):

        self.get_stars_dot_directions()
        self.get_directions_abs()
        self.get_the_dust_map_distances()

        print(self.dust_map_distances)

        star_r = self.stars_galactic[:,2]
        d = self.dust_map_distances
        t1, t2 = ray.find_crossing_positions(star_r, self.stars_dot_directions, self.directions_abs, d)
        print("t1 ", t1)
        print("t2 ", t2)
        print("t1 shape", t1.shape)
        print("t2 shape", t2.shape)

    

