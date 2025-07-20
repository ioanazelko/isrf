# base_spectrum.py
from abc import ABC, abstractmethod
import numpy as np
from astropy import units as u
from astropy.modeling import models

class BaseSpectrum(ABC):
    """Abstract base class for all spectral objects"""
    
    def __init__(self, wavelength, flux, units='erg/s/cm2/A'):
        """
        Parameters
        ----------
        wavelength : array_like
            Wavelength array with units
        flux : array_like  
            Flux array with units
        units : str
            Flux units (erg/s/cm2/A, Jy, etc.)
        """
        self.wavelength = self._validate_wavelength(wavelength)
        self.flux = self._validate_flux(flux, units)
        self._interpolator = None
        
    @property
    def frequency(self):
        """Convert wavelength to frequency"""
        return (2.998e8 * u.m/u.s / self.wavelength).to(u.Hz)
    
    @abstractmethod
    def integrate_filter(self, filter_response):
        """Integrate spectrum through photometric filter"""
        pass
    
    @abstractmethod
    def interpolate(self, new_wavelength):
        """Interpolate spectrum to new wavelength grid"""
        pass
    
    def normalize(self, wavelength_norm=None, method='total'):
        """Normalize spectrum by total luminosity or at specific wavelength"""
        if method == 'total':
            total_lum = self.total_luminosity()
            self.flux = self.flux / total_lum
        elif method == 'wavelength' and wavelength_norm is not None:
            norm_flux = self.interpolate(wavelength_norm)
            self.flux = self.flux / norm_flux
    
    def total_luminosity(self):
        """Calculate total luminosity by integrating over wavelength"""
        return np.trapz(self.flux, self.wavelength)
    
    def effective_temperature(self):
        """Calculate effective temperature from spectrum"""
        # Stefan-Boltzmann law implementation
        sigma_sb = 5.67e-8 * u.W / (u.m**2 * u.K**4)
        L_total = self.total_luminosity()
        # Assume stellar radius for temperature calculation
        return (L_total / (4 * np.pi * sigma_sb))**(1/4)