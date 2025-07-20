"""
Blackbody radiation calculations for 3D ISRF code.

This module provides comprehensive blackbody and modified blackbody
radiation calculations for stellar sources and dust emission modeling.
Includes support for temperature distributions, spectral index variations,
and observational fitting routines.
"""

import numpy as np
import warnings
from scipy import integrate, optimize
from scipy.special import zeta
from astropy import units as u
from astropy import constants as const
from astropy.modeling import models, fitting
from astropy.utils.exceptions import AstropyWarning

from .base_spectrum import BaseSpectrum
from .units import SpectralUnits


class BlackbodySpectrum(BaseSpectrum):
    """
    Pure blackbody spectrum implementation.
    
    Provides Planck function calculations with proper unit handling
    and various output formats commonly used in astrophysics.
    """
    
    def __init__(self, temperature, wavelength=None, normalize=False):
        """
        Initialize blackbody spectrum.
        
        Parameters
        ----------
        temperature : float or Quantity
            Blackbody temperature in Kelvin
        wavelength : array_like, optional
            Wavelength grid. If None, creates default grid
        normalize : bool, optional
            Whether to normalize to unit total luminosity
        """
        self.temperature = self._validate_temperature(temperature)
        
        if wavelength is None:
            wavelength = self._default_wavelength_grid()
        
        # Calculate Planck function
        flux = self._planck_function(wavelength, self.temperature)
        
        if normalize:
            flux = flux / self.stefan_boltzmann_luminosity()
        
        super().__init__(wavelength, flux, units='erg/s/cm2/Hz/sr')
        
    def _validate_temperature(self, temperature):
        """Validate and convert temperature to Kelvin."""
        if hasattr(temperature, 'unit'):
            return temperature.to(u.K)
        else:
            return temperature * u.K
    
    def _default_wavelength_grid(self):
        """Create default wavelength grid optimized for blackbody."""
        # Logarithmic grid from 0.1 to 1000 microns
        return np.logspace(-1, 3, 2000) * u.micron
    
    def _planck_function(self, wavelength, temperature):
        """
        Calculate Planck function B_lambda(T).
        
        Parameters
        ----------
        wavelength : array_like
            Wavelength array with units
        temperature : Quantity
            Temperature with units
            
        Returns
        -------
        flux : array_like
            Planck function in erg/s/cm2/Hz/sr
        """
        # Constants
        h = const.h  # Planck constant
        c = const.c  # Speed of light
        k_B = const.k_B  # Boltzmann constant
        
        # Convert wavelength to frequency for calculation
        frequency = (c / wavelength).to(u.Hz)
        
        # Planck function in frequency space
        # B_nu = (2 h nu^3 / c^2) / (exp(h nu / k T) - 1)
        
        # Calculate exponential term carefully to avoid overflow
        x = (h * frequency / (k_B * temperature)).decompose()
        
        # Handle overflow in exponential
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            exp_term = np.exp(x.value)
            
        # Replace infinite values with large finite values
        exp_term = np.where(np.isinf(exp_term), 1e100, exp_term)
        
        # Calculate Planck function
        numerator = 2 * h * frequency**3 / c**2
        denominator = exp_term - 1
        
        # Avoid division by zero
        denominator = np.where(denominator == 0, 1e-100, denominator)
        
        planck_nu = (numerator / denominator).to(u.erg / u.s / u.cm**2 / u.Hz / u.sr)
        
        # Convert to wavelength space if needed
        # B_lambda = B_nu * c / lambda^2
        planck_lambda = (planck_nu * c / wavelength**2).to(
            u.erg / u.s / u.cm**2 / u.AA / u.sr
        )
        
        return planck_lambda
    
    def stefan_boltzmann_luminosity(self):
        """Calculate total luminosity using Stefan-Boltzmann law."""
        sigma_sb = const.sigma_sb
        return sigma_sb * self.temperature**4
    
    def peak_wavelength(self):
        """Calculate peak wavelength using Wien's displacement law."""
        wien_constant = 2.898e-3 * u.m * u.K
        return (wien_constant / self.temperature).to(u.micron)
    
    def peak_frequency(self):
        """Calculate peak frequency."""
        return (const.c / self.peak_wavelength()).to(u.Hz)
    
    def rayleigh_jeans_flux(self, wavelength):
        """Calculate Rayleigh-Jeans approximation (long wavelength limit)."""
        return (2 * const.c * const.k_B * self.temperature / wavelength**4).to(
            u.erg / u.s / u.cm**2 / u.AA / u.sr
        )
    
    def wien_flux(self, wavelength):
        """Calculate Wien approximation (short wavelength limit)."""
        h = const.h
        c = const.c
        k_B = const.k_B
        
        frequency = c / wavelength
        exp_term = np.exp(-h * frequency / (k_B * self.temperature))
        
        return (2 * h * frequency**3 / c**2 * exp_term).to(
            u.erg / u.s / u.cm**2 / u.Hz / u.sr
        )


class ModifiedBlackbodySpectrum(BlackbodySpectrum):
    """
    Modified blackbody spectrum with power-law opacity.
    
    Commonly used for dust emission modeling where the opacity
    follows a power law: kappa(lambda) = kappa_0 * (lambda/lambda_0)^(-beta)
    """
    
    def __init__(self, temperature, beta=2.0, lambda_ref=250*u.micron, 
                 kappa_ref=1.0, wavelength=None, normalize=False):
        """
        Initialize modified blackbody spectrum.
        
        Parameters
        ----------
        temperature : float or Quantity
            Dust temperature in Kelvin
        beta : float
            Spectral index (typically 1-3 for dust)
        lambda_ref : Quantity
            Reference wavelength for opacity normalization
        kappa_ref : float
            Opacity at reference wavelength (cm^2/g)
        wavelength : array_like, optional
            Wavelength grid
        normalize : bool, optional
            Whether to normalize total luminosity
        """
        self.beta = beta
        self.lambda_ref = lambda_ref
        self.kappa_ref = kappa_ref
        
        # Initialize parent class first
        super().__init__(temperature, wavelength, normalize=False)
        
        # Apply opacity modification
        self._apply_opacity_law()
        
        if normalize:
            self.flux = self.flux / self.total_luminosity()
    
    def _apply_opacity_law(self):
        """Apply power-law opacity to blackbody spectrum."""
        # Calculate opacity scaling
        opacity_scaling = (self.wavelength / self.lambda_ref)**(-self.beta)
        
        # Apply to flux
        self.flux = self.flux * opacity_scaling * self.kappa_ref
        
    def effective_temperature(self, wavelength_range=(100*u.micron, 500*u.micron)):
        """
        Calculate effective temperature from modified blackbody fit.
        
        Parameters
        ----------
        wavelength_range : tuple
            Wavelength range for fitting (lambda_min, lambda_max)
            
        Returns
        -------
        T_eff : Quantity
            Effective temperature
        """
        lambda_min, lambda_max = wavelength_range
        mask = (self.wavelength >= lambda_min) & (self.wavelength <= lambda_max)
        
        if not np.any(mask):
            raise ValueError("No data points in specified wavelength range")
        
        # Fit modified blackbody to data
        fitter = ModifiedBlackbodyFitter()
        fit_result = fitter.fit(
            self.wavelength[mask], 
            self.flux[mask], 
            beta_fixed=self.beta
        )
        
        return fit_result['temperature']


class GreybodySpectrum(ModifiedBlackbodySpectrum):
    """
    Greybody spectrum with wavelength-independent emissivity.
    
    Special case of modified blackbody with beta=0.
    """
    
    def __init__(self, temperature, emissivity=1.0, wavelength=None, normalize=False):
        """
        Initialize greybody spectrum.
        
        Parameters
        ----------
        temperature : float or Quantity
            Temperature in Kelvin
        emissivity : float
            Wavelength-independent emissivity (0-1)
        wavelength : array_like, optional
            Wavelength grid
        normalize : bool, optional
            Whether to normalize total luminosity
        """
        super().__init__(
            temperature=temperature,
            beta=0.0,
            lambda_ref=100*u.micron,  # Arbitrary since beta=0
            kappa_ref=emissivity,
            wavelength=wavelength,
            normalize=normalize
        )


class MultiTemperatureBlackbody(BaseSpectrum):
    """
    Superposition of multiple blackbody components.
    
    Useful for modeling complex dust temperature distributions
    or multi-component stellar populations.
    """
    
    def __init__(self, temperatures, weights=None, wavelength=None, 
                 beta_values=None, normalize=False):
        """
        Initialize multi-temperature blackbody.
        
        Parameters
        ----------
        temperatures : array_like
            Array of temperatures in Kelvin
        weights : array_like, optional
            Relative weights for each component
        wavelength : array_like, optional
            Wavelength grid
        beta_values : array_like, optional
            Spectral indices for each component (for modified BB)
        normalize : bool, optional
            Whether to normalize total luminosity
        """
        self.temperatures = np.atleast_1d(temperatures) * u.K
        
        if weights is None:
            weights = np.ones(len(self.temperatures))
        self.weights = np.atleast_1d(weights)
        
        if len(self.weights) != len(self.temperatures):
            raise ValueError("Weights and temperatures must have same length")
        
        if wavelength is None:
            wavelength = self._default_wavelength_grid()
        
        if beta_values is not None:
            self.beta_values = np.atleast_1d(beta_values)
            if len(self.beta_values) != len(self.temperatures):
                raise ValueError("Beta values and temperatures must have same length")
        else:
            self.beta_values = None
        
        # Calculate combined spectrum
        flux = self._calculate_combined_spectrum(wavelength)
        
        if normalize:
            flux = flux / np.trapz(flux, wavelength)
        
        super().__init__(wavelength, flux)
    
    def _calculate_combined_spectrum(self, wavelength):
        """Calculate weighted sum of blackbody components."""
        total_flux = np.zeros(len(wavelength)) * u.erg/u.s/u.cm**2/u.AA/u.sr
        
        for i, (temp, weight) in enumerate(zip(self.temperatures, self.weights)):
            if self.beta_values is not None:
                # Modified blackbody
                bb = ModifiedBlackbodySpectrum(
                    temperature=temp,
                    beta=self.beta_values[i],
                    wavelength=wavelength
                )
            else:
                # Pure blackbody
                bb = BlackbodySpectrum(temperature=temp, wavelength=wavelength)
            
            total_flux += weight * bb.flux
        
        return total_flux
    
    def average_temperature(self, method='luminosity_weighted'):
        """
        Calculate average temperature.
        
        Parameters
        ----------
        method : str
            Averaging method: 'arithmetic', 'luminosity_weighted', 'rms'
        """
        if method == 'arithmetic':
            return np.average(self.temperatures, weights=self.weights)
        elif method == 'luminosity_weighted':
            # Weight by T^4 (Stefan-Boltzmann law)
            luminosity_weights = self.weights * self.temperatures**4
            return np.average(self.temperatures, weights=luminosity_weights)
        elif method == 'rms':
            return np.sqrt(np.average(self.temperatures**2, weights=self.weights))
        else:
            raise ValueError(f"Unknown averaging method: {method}")


class TemperatureDistribution:
    """
    Theoretical temperature distribution functions for dust.
    
    Provides various analytical models for dust temperature
    distributions in different astrophysical environments.
    """
    
    @staticmethod
    def power_law_distribution(t_min, t_max, alpha=-2.0, n_points=50):
        """
        Power-law temperature distribution.
        
        dN/dT ∝ T^alpha
        """
        temperatures = np.logspace(np.log10(t_min), np.log10(t_max), n_points)
        weights = temperatures**alpha
        weights = weights / np.sum(weights)  # Normalize
        
        return temperatures * u.K, weights
    
    @staticmethod
    def log_normal_distribution(t_mean, sigma, n_points=50):
        """
        Log-normal temperature distribution.
        
        Common for turbulent heating environments.
        """
        log_t_mean = np.log(t_mean)
        
        # Generate log-space grid
        log_t_min = log_t_mean - 4*sigma
        log_t_max = log_t_mean + 4*sigma
        log_temperatures = np.linspace(log_t_min, log_t_max, n_points)
        temperatures = np.exp(log_temperatures)
        
        # Calculate log-normal weights
        weights = np.exp(-(log_temperatures - log_t_mean)**2 / (2*sigma**2))
        weights = weights / (sigma * np.sqrt(2*np.pi))
        weights = weights / np.sum(weights)  # Normalize
        
        return temperatures * u.K, weights
    
    @staticmethod
    def delta_function(temperature):
        """Single temperature (delta function)."""
        return np.array([temperature]) * u.K, np.array([1.0])


class ModifiedBlackbodyFitter:
    """
    Fit modified blackbody models to observational data.
    
    Provides robust fitting routines for determining dust
    temperatures and spectral indices from photometric data.
    """
    
    def __init__(self):
        """Initialize fitter with default parameters."""
        self.last_fit_result = None
        self.last_fit_errors = None
    
    def fit(self, wavelength, flux, flux_errors=None, beta_fixed=None,
            temperature_bounds=(5, 100), beta_bounds=(0.5, 3.0)):
        """
        Fit modified blackbody to data.
        
        Parameters
        ----------
        wavelength : array_like
            Wavelength values with units
        flux : array_like  
            Flux values with units
        flux_errors : array_like, optional
            Flux uncertainties
        beta_fixed : float, optional
            Fix beta to this value
        temperature_bounds : tuple
            (T_min, T_max) bounds in Kelvin
        beta_bounds : tuple
            (beta_min, beta_max) bounds
            
        Returns
        -------
        fit_result : dict
            Dictionary with fitted parameters and uncertainties
        """
        # Convert to consistent units
        wavelength = wavelength.to(u.micron).value
        flux = flux.to(u.erg/u.s/u.cm**2/u.Hz).value
        
        if flux_errors is not None:
            flux_errors = flux_errors.to(u.erg/u.s/u.cm**2/u.Hz).value
        else:
            flux_errors = np.ones_like(flux) * 0.1 * np.mean(flux)
        
        # Define fitting function
        if beta_fixed is not None:
            def model_func(params):
                temperature, amplitude = params
                model_flux = self._modified_blackbody_model(
                    wavelength, temperature, beta_fixed, amplitude
                )
                return model_flux
            
            # Initial guess
            initial_params = [20.0, np.max(flux)]
            bounds = [temperature_bounds, (0, 10*np.max(flux))]
            
        else:
            def model_func(params):
                temperature, beta, amplitude = params
                model_flux = self._modified_blackbody_model(
                    wavelength, temperature, beta, amplitude
                )
                return model_flux
            
            # Initial guess
            initial_params = [20.0, 2.0, np.max(flux)]
            bounds = [temperature_bounds, beta_bounds, (0, 10*np.max(flux))]
        
        # Chi-squared function
        def chi_squared(params):
            model_flux = model_func(params)
            chi2 = np.sum(((flux - model_flux) / flux_errors)**2)
            return chi2
        
        # Perform fit
        try:
            result = optimize.minimize(
                chi_squared, 
                initial_params, 
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if not result.success:
                warnings.warn("Fit did not converge", AstropyWarning)
            
            # Extract results
            if beta_fixed is not None:
                fit_params = {
                    'temperature': result.x[0] * u.K,
                    'beta': beta_fixed,
                    'amplitude': result.x[1],
                    'chi_squared': result.fun,
                    'reduced_chi_squared': result.fun / (len(flux) - 2)
                }
            else:
                fit_params = {
                    'temperature': result.x[0] * u.K,
                    'beta': result.x[1],
                    'amplitude': result.x[2],
                    'chi_squared': result.fun,
                    'reduced_chi_squared': result.fun / (len(flux) - 3)
                }
            
            # Calculate parameter uncertainties (approximate)
            try:
                # Hessian approximation for error estimation
                hess_inv = result.hess_inv
                if hasattr(hess_inv, 'todense'):
                    hess_inv = hess_inv.todense()
                
                param_errors = np.sqrt(np.diag(hess_inv))
                
                if beta_fixed is not None:
                    fit_params['temperature_error'] = param_errors[0] * u.K
                    fit_params['amplitude_error'] = param_errors[1]
                else:
                    fit_params['temperature_error'] = param_errors[0] * u.K
                    fit_params['beta_error'] = param_errors[1]
                    fit_params['amplitude_error'] = param_errors[2]
                    
            except Exception:
                warnings.warn("Could not calculate parameter uncertainties", 
                            AstropyWarning)
            
            self.last_fit_result = fit_params
            return fit_params
            
        except Exception as e:
            raise RuntimeError(f"Fitting failed: {str(e)}")
    
    def _modified_blackbody_model(self, wavelength, temperature, beta, amplitude):
        """Calculate modified blackbody model flux."""
        # Physical constants (in CGS)
        h = 6.62607015e-27  # erg*s
        c = 2.99792458e10   # cm/s
        k_B = 1.380649e-16  # erg/K
        
        # Convert wavelength to frequency
        frequency = c / (wavelength * 1e-4)  # Hz (wavelength in microns)
        
        # Planck function
        x = h * frequency / (k_B * temperature)
        planck_nu = 2 * h * frequency**3 / c**2 / (np.exp(x) - 1)
        
        # Opacity scaling (relative to some reference)
        lambda_ref = 250.0  # microns
        opacity_scaling = (wavelength / lambda_ref)**(-beta)
        
        # Modified blackbody
        model_flux = amplitude * opacity_scaling * planck_nu
        
        return model_flux
    
    def plot_fit(self, wavelength, flux, flux_errors=None, 
                 show_components=False):
        """
        Plot fitted model with data.
        
        Parameters
        ----------
        wavelength : array_like
            Wavelength values
        flux : array_like
            Flux values
        flux_errors : array_like, optional
            Flux uncertainties
        show_components : bool
            Whether to show individual temperature components
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        
        if self.last_fit_result is None:
            raise ValueError("No fit result available. Run fit() first.")
        
        # Create model curve
        wave_model = np.logspace(
            np.log10(wavelength.min().value), 
            np.log10(wavelength.max().value), 
            1000
        ) * wavelength.unit
        
        # Generate model spectrum
        model_spec = ModifiedBlackbodySpectrum(
            temperature=self.last_fit_result['temperature'],
            beta=self.last_fit_result['beta'],
            wavelength=wave_model
        )
        
        # Scale by fitted amplitude
        model_flux = (model_spec.flux * 
                     self.last_fit_result['amplitude'] / 
                     model_spec.flux.max())
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot data
        if flux_errors is not None:
            ax.errorbar(wavelength, flux, yerr=flux_errors, 
                       fmt='o', label='Data', capsize=3)
        else:
            ax.scatter(wavelength, flux, label='Data', s=50)
        
        # Plot model
        ax.plot(wave_model, model_flux, 'r-', 
               label=f'Modified BB (T={self.last_fit_result["temperature"]:.1f}, '
                     f'β={self.last_fit_result["beta"]:.2f})')
        
        ax.set_xlabel('Wavelength (μm)')
        ax.set_ylabel('Flux')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax


# Convenience functions for common use cases
def planck_function(wavelength, temperature):
    """
    Simple Planck function calculation.
    
    Parameters
    ----------
    wavelength : array_like
        Wavelength with units
    temperature : float or Quantity
        Temperature in Kelvin
        
    Returns
    -------
    flux : array_like
        Planck function values
    """
    bb = BlackbodySpectrum(temperature, wavelength)
    return bb.flux


def wien_displacement_law(temperature):
    """
    Calculate peak wavelength using Wien's displacement law.
    
    Parameters
    ----------
    temperature : float or Quantity
        Temperature in Kelvin
        
    Returns
    -------
    lambda_max : Quantity
        Peak wavelength
    """
    wien_constant = 2.898e-3 * u.m * u.K
    if hasattr(temperature, 'unit'):
        return (wien_constant / temperature).to(u.micron)
    else:
        return (wien_constant / (temperature * u.K)).to(u.micron)


def stefan_boltzmann_law(temperature, radius=None):
    """
    Calculate total luminosity using Stefan-Boltzmann law.
    
    Parameters
    ----------
    temperature : float or Quantity
        Temperature in Kelvin
    radius : Quantity, optional
        Radius for luminosity calculation
        
    Returns
    -------
    luminosity : Quantity
        Total luminosity
    """
    sigma_sb = const.sigma_sb
    if hasattr(temperature, 'unit'):
        temp = temperature
    else:
        temp = temperature * u.K
    
    flux = sigma_sb * temp**4
    
    if radius is not None:
        return 4 * np.pi * radius**2 * flux
    else:
        return flux