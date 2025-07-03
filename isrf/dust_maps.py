import matplotlib.pyplot as plt
import numpy as np


import isrf.plot_utils as plot_utils
import isrf.ray as ray
import isrf.testing_utils as testing_utils
import isrf.utils as utils



class DustMaps:
    def __init__(self,dust_map_model="Bayestar2019"):
        """
        Initialize the DustMaps class.
        This class is responsible for handling dust maps and their properties.
        """
        self.dust_map_model = dust_map_model

        self.load_dust_map( )
        
    def load_dust_map(self ):
        """
        Load a dust map from a specified file.
        
        Parameters
        ----------
        file_path : str
            The path to the dust map file.
        """

        self.dust_map_distances = np.linspace(0, 15, 30)  # Example distances in kpc

        #
        #self.dust_map = utils.load_data(file_path)
       
    def get_dust_map_distances(self):
        """
        Get the distances associated with the dust map.
        
        Returns
        -------
        numpy.ndarray
            An array of distances in kpc.
        """
        return self.dust_map_distances