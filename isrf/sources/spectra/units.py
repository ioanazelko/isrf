"""
Unit conversion and handling for 3D ISRF code.

This module provides comprehensive unit management for spectral calculations,
radiation transport, and astrophysical quantities. Built on astropy.units
with extensions for specialized ISRF calculations.
"""

import numpy as np
import warnings
from functools import wraps
from astropy import units as u
from astropy import constants as const
from astropy.units import Quantity
from astropy.units.equivalencies import spectral, spectral_density, brightness_temperature
from astropy.utils.exceptions import AstropyWarning


# ============================================================================
# Custom Unit Definitions for ISRF Calculations
# ============================================================================

# Habing unit for FUV radiation field strength
# G_0 = 1.6 × 10^(-3) erg cm^(-2) s^(-1) (6-13.6 eV)
habing_unit = u.def_unit(
    'G0', 1.6e-3 * u.erg / u.cm**2 / u.s,
    doc="Habing unit for FUV radiation field strength"
)

# Draine unit for ISRF (alternative to Habing)
# Used in some literature as 2.3 × G_0
draine_unit = u.def_unit(
    'Draine', 2.3 * habing_unit,
    doc="Draine unit for interstellar radiation field"
)

# Solar luminosity per unit solid angle (for stellar sources)
solar_luminosity_per_steradian = u.def_unit(
    'L_sun_sr', const.L_sun / u.sr,
    doc="Solar luminosity per steradian"
)

# Jansky per square arcsecond (for surface brightness)
jy_per_arcsec2 = u.def_unit(
    'Jy_arcsec2', u.Jy / u.arcsec**2,
    doc="Jansky per square arcsecond"
)

# MJy per steradian (common in far-IR astronomy)
mjy_per_sr = u.def_unit(
    'MJy_sr', u.MJy / u.sr,
    doc="Mega-Jansky per steradian"
)

# Register custom units with astropy
u.add_enabled_units([habing_unit, draine_unit, solar_luminosity_per_steradian,
                     jy_per_arcsec2, mjy_per_sr])


# ============================================================================
# Unit Validation and Conversion Utilities
# ============================================================================

class UnitError(ValueError):
    """Custom exception for unit-related errors."""
    pass


def require_units(expected_unit):
    """
    Decorator to ensure function arguments have correct units.
    
    Parameters
    ----------
    expected_unit : astropy.units.Unit
        Required unit for the decorated function's first argument
    """
    def decorator(func):
        @wraps(func)
        def wrapper(value, *args, **kwargs):
            if not hasattr(value, 'unit'):
                raise UnitError(f"Function {func.__name__} requires input with units. "
                              f"Expected: {expected_unit}")
            
            try:
                # Attempt conversion to check compatibility
                value.to(expected_unit)
            except u.UnitConversionError:
                raise UnitError(f"Function {func.__name__} requires units compatible "
                              f"with {expected_unit}. Got: {value.unit}")
            
            return func(value, *args, **kwargs)
        return wrapper
    return decorator


def validate_spectral_units(wavelength=None, frequency=None, flux=None, 
                           flux_units=None):
    """
    Validate and standardize spectral quantity units.
    
    Parameters
    ----------
    wavelength : Quantity, optional
        Wavelength array
    frequency : Quantity, optional
        Frequency array
    flux : Quantity, optional
        Flux array
    flux_units : str, optional
        Expected flux units for validation
        
    Returns
    -------
    validated_quantities : dict
        Dictionary with validated and standardized quantities
    """
    result = {}
    
    # Wavelength validation
    if wavelength is not None:
        if not hasattr(wavelength, 'unit'):
            raise UnitError("Wavelength must have units")
        
        if not wavelength.unit.is_equivalent(u.m):
            raise UnitError(f"Wavelength units must be length units. "
                          f"Got: {wavelength.unit}")
        
        # Convert to microns for consistency
        result['wavelength'] = wavelength.to(u.micron)
    
    # Frequency validation
    if frequency is not None:
        if not hasattr(frequency, 'unit'):
            raise UnitError("Frequency must have units")
        
        if not frequency.unit.is_equivalent(u.Hz):
            raise UnitError(f"Frequency units must be frequency units. "
                          f"Got: {frequency.unit}")
        
        result['frequency'] = frequency.to(u.Hz)
    
    # Flux validation
    if flux is not None:
        if not hasattr(flux, 'unit'):
            raise UnitError("Flux must have units")
        
        # Check if flux units are valid spectral units
        valid_flux_units = [
            u.erg / u.s / u.cm**2 / u.Hz,  # Specific flux density (nu)
            u.erg / u.s / u.cm**2 / u.AA,  # Specific flux density (lambda)
            u.Jy,  # Jansky
            u.W / u.m**2 / u.Hz,  # SI flux density
        ]
        
        is_valid = any(flux.unit.is_equivalent(valid_unit) 
                      for valid_unit in valid_flux_units)
        
        if not is_valid:
            raise UnitError(f"Flux units not recognized as spectral flux density. "
                          f"Got: {flux.unit}")
        
        if flux_units is not None:
            try:
                flux.to(flux_units)
            except u.UnitConversionError:
                raise UnitError(f"Flux units incompatible with required {flux_units}")
        
        result['flux'] = flux
    
    return result


# ============================================================================
# Spectral Unit Conversions
# ============================================================================

class SpectralUnits:
    """
    Comprehensive spectral unit conversion utilities.
    
    Handles conversions between different representations of spectral
    quantities (wavelength/frequency, per-Hz/per-wavelength, etc.)
    """
    
    # Standard wavelength units used in astronomy
    WAVELENGTH_UNITS = {
        'angstrom': u.AA,
        'nanometer': u.nm,
        'micron': u.micron,
        'millimeter': u.mm,
        'centimeter': u.cm,
        'meter': u.m
    }
    
    # Standard frequency units
    FREQUENCY_UNITS = {
        'hz': u.Hz,
        'khz': u.kHz,
        'mhz': u.MHz,
        'ghz': u.GHz,
        'thz': u.THz
    }
    
    # Standard flux density units
    FLUX_UNITS = {
        'cgs_nu': u.erg / u.s / u.cm**2 / u.Hz,
        'cgs_lambda': u.erg / u.s / u.cm**2 / u.AA,
        'si_nu': u.W / u.m**2 / u.Hz,
        'si_lambda': u.W / u.m**2 / u.m,
        'jansky': u.Jy,
        'mjy': u.MJy,
        'ujy': u.microJy
    }
    
    @staticmethod
    def wavelength_to_frequency(wavelength):
        """Convert wavelength to frequency using c = λν."""
        return wavelength.to(u.Hz, equivalencies=spectral())
    
    @staticmethod
    def frequency_to_wavelength(frequency):
        """Convert frequency to wavelength using c = λν."""
        return frequency.to(u.micron, equivalencies=spectral())
    
    @staticmethod
    def flux_lambda_to_flux_nu(flux_lambda, wavelength):
        """
        Convert flux per unit wavelength to flux per unit frequency.
        
        F_ν = F_λ × λ² / c
        """
        if not hasattr(flux_lambda, 'unit'):
            raise UnitError("flux_lambda must have units")
        if not hasattr(wavelength, 'unit'):
            raise UnitError("wavelength must have units")
        
        # Use astropy equivalencies for proper conversion
        return flux_lambda.to(
            u.erg / u.s / u.cm**2 / u.Hz,
            equivalencies=spectral_density(wavelength)
        )
    
    @staticmethod
    def flux_nu_to_flux_lambda(flux_nu, frequency):
        """
        Convert flux per unit frequency to flux per unit wavelength.
        
        F_λ = F_ν × c / λ²
        """
        if not hasattr(flux_nu, 'unit'):
            raise UnitError("flux_nu must have units")
        if not hasattr(frequency, 'unit'):
            raise UnitError("frequency must have units")
        
        # Convert frequency to wavelength first
        wavelength = SpectralUnits.frequency_to_wavelength(frequency)
        
        return flux_nu.to(
            u.erg / u.s / u.cm**2 / u.AA,
            equivalencies=spectral_density(wavelength)
        )
    
    @staticmethod
    def surface_brightness_to_flux_density(surface_brightness, solid_angle):
        """
        Convert surface brightness to flux density.
        
        F = B × Ω
        """
        return surface_brightness * solid_angle
    
    @staticmethod
    def flux_density_to_surface_brightness(flux_density, solid_angle):
        """
        Convert flux density to surface brightness.
        
        B = F / Ω
        """
        return flux_density / solid_angle
    
    @staticmethod
    def luminosity_to_flux(luminosity, distance):
        """
        Convert luminosity to flux using inverse square law.
        
        F = L / (4π d²)
        """
        return luminosity / (4 * np.pi * distance**2)
    
    @staticmethod
    def flux_to_luminosity(flux, distance):
        """
        Convert flux to luminosity using inverse square law.
        
        L = F × 4π d²
        """
        return flux * 4 * np.pi * distance**2
    
    @staticmethod
    def magnitude_to_flux(magnitude, zeropoint_flux):
        """
        Convert magnitude to flux.
        
        m = -2.5 log₁₀(F/F₀)
        F = F₀ × 10^(-0.4 × m)
        """
        return zeropoint_flux * 10**(-0.4 * magnitude)
    
    @staticmethod
    def flux_to_magnitude(flux, zeropoint_flux):
        """
        Convert flux to magnitude.
        
        m = -2.5 log₁₀(F/F₀)
        """
        return -2.5 * np.log10(flux / zeropoint_flux)


# ============================================================================
# ISRF-Specific Unit Conversions
# ============================================================================

class ISRFUnits:
    """
    Specialized unit conversions for ISRF calculations.
    
    Handles radiation field strength, dust heating rates,
    and other quantities specific to ISM modeling.
    """
    
    @staticmethod
    def flux_to_habing_units(flux, wavelength_range=(6*u.eV, 13.6*u.eV)):
        """
        Convert flux to Habing units (G₀).
        
        Integrates flux over FUV range (6-13.6 eV) and normalizes
        to the Habing unit definition.
        
        Parameters
        ----------
        flux : Quantity
            Spectral flux density
        wavelength_range : tuple
            Energy range for integration (default: FUV range)
        """
        # Convert energy range to wavelength
        lambda_min = (const.h * const.c / wavelength_range[1]).to(u.AA)
        lambda_max = (const.h * const.c / wavelength_range[0]).to(u.AA)
        
        # This is a simplified version - full implementation would
        # require integration over the specified wavelength range
        warnings.warn("Simplified conversion to Habing units. "
                     "Full integration not implemented.", AstropyWarning)
        
        # Placeholder conversion
        return flux.to(habing_unit, equivalencies=u.dimensionless_unscaled())
    
    @staticmethod
    def heating_rate_units():
        """Return standard units for dust heating rates."""
        return u.erg / u.s / u.g  # erg/s/g of dust
    
    @staticmethod
    def cooling_rate_units():
        """Return standard units for gas cooling rates."""
        return u.erg / u.s / u.cm**3  # erg/s/cm³
    
    @staticmethod
    def optical_depth_units():
        """Return dimensionless units for optical depth."""
        return u.dimensionless_unscaled
    
    @staticmethod
    def extinction_coefficient_units():
        """Return standard units for extinction coefficients."""
        return u.cm**2 / u.g  # cm²/g
    
    @staticmethod
    def uv_flux_to_g0(uv_flux, wavelength):
        """
        Convert UV flux to G₀ units more accurately.
        
        Parameters
        ----------
        uv_flux : Quantity
            UV flux density
        wavelength : Quantity
            Wavelength corresponding to flux
        """
        # Convert wavelength to energy
        energy = (const.h * const.c / wavelength).to(u.eV)
        
        # Check if in FUV range (6-13.6 eV)
        fuv_mask = (energy >= 6*u.eV) & (energy <= 13.6*u.eV)
        
        if not np.any(fuv_mask):
            warnings.warn("No flux in FUV range (6-13.6 eV)", AstropyWarning)
            return 0 * habing_unit
        
        # Simple conversion (more sophisticated integration needed for real use)
        fuv_flux = np.mean(uv_flux[fuv_mask]) if np.any(fuv_mask) else 0*uv_flux.unit
        
        # Convert to Habing units
        conversion_factor = 1.6e-3 * u.erg / u.cm**2 / u.s
        g0_value = (fuv_flux / conversion_factor).decompose()
        
        return g0_value * habing_unit


# ============================================================================
# Temperature and Energy Conversions
# ============================================================================

class TemperatureUnits:
    """Temperature and thermal energy conversions."""
    
    @staticmethod
    def temperature_to_wavelength_peak(temperature):
        """
        Wien's displacement law: λ_max = b / T
        where b = 2.898 × 10^(-3) m⋅K
        """
        wien_constant = 2.898e-3 * u.m * u.K
        return (wien_constant / temperature).to(u.micron)
    
    @staticmethod
    def wavelength_peak_to_temperature(wavelength_peak):
        """Inverse Wien's displacement law."""
        wien_constant = 2.898e-3 * u.m * u.K
        return (wien_constant / wavelength_peak).to(u.K)
    
    @staticmethod
    def energy_to_temperature(energy):
        """Convert photon energy to equivalent temperature: E = k_B T"""
        return (energy / const.k_B).to(u.K)
    
    @staticmethod
    def temperature_to_energy(temperature):
        """Convert temperature to thermal energy: E = k_B T"""
        return (const.k_B * temperature).to(u.eV)
    
    @staticmethod
    def brightness_temperature(flux, frequency, solid_angle):
        """
        Calculate brightness temperature from flux and frequency.
        
        Uses Rayleigh-Jeans approximation for radio/mm wavelengths.
        """
        return flux.to(
            u.K,
            equivalencies=brightness_temperature(frequency, solid_angle)
        )


# ============================================================================
# Coordinate System Units
# ============================================================================

class CoordinateUnits:
    """Coordinate system and angular unit conversions."""
    
    @staticmethod
    def degrees_to_radians(degrees):
        """Convert degrees to radians."""
        return (degrees * u.deg).to(u.rad)
    
    @staticmethod
    def radians_to_degrees(radians):
        """Convert radians to degrees."""
        return (radians * u.rad).to(u.deg)
    
    @staticmethod
    def arcsec_to_radians(arcsec):
        """Convert arcseconds to radians."""
        return (arcsec * u.arcsec).to(u.rad)
    
    @staticmethod
    def healpix_pixel_area(nside):
        """
        Calculate pixel area for HEALPix map.
        
        Area = 4π / (12 * Nside²) steradians
        """
        npix = 12 * nside**2
        total_area = 4 * np.pi * u.sr
        return total_area / npix
    
    @staticmethod
    def solid_angle_to_area(solid_angle, distance):
        """Convert solid angle to physical area at given distance."""
        return solid_angle * distance**2
    
    @staticmethod
    def area_to_solid_angle(area, distance):
        """Convert physical area to solid angle at given distance."""
        return area / distance**2


# ============================================================================
# Unit System Standards
# ============================================================================

class StandardUnits:
    """
    Standard unit systems for different calculation contexts.
    
    Provides consistent unit choices for different parts of the
    ISRF calculation pipeline.
    """
    
    # CGS units (traditional astrophysics)
    CGS = {
        'length': u.cm,
        'mass': u.g,
        'time': u.s,
        'temperature': u.K,
        'energy': u.erg,
        'flux_nu': u.erg / u.s / u.cm**2 / u.Hz,
        'flux_lambda': u.erg / u.s / u.cm**2 / u.AA,
        'luminosity': u.erg / u.s,
        'surface_brightness': u.erg / u.s / u.cm**2 / u.Hz / u.sr
    }
    
    # SI units
    SI = {
        'length': u.m,
        'mass': u.kg,
        'time': u.s,
        'temperature': u.K,
        'energy': u.J,
        'flux_nu': u.W / u.m**2 / u.Hz,
        'flux_lambda': u.W / u.m**2 / u.m,
        'luminosity': u.W,
        'surface_brightness': u.W / u.m**2 / u.Hz / u.sr
    }
    
    # Astronomical units (mixed system)
    ASTRONOMICAL = {
        'length_small': u.AU,
        'length_large': u.pc,
        'mass_stellar': u.M_sun,
        'time': u.yr,
        'temperature': u.K,
        'energy': u.eV,
        'wavelength': u.micron,
        'frequency': u.GHz,
        'flux': u.Jy,
        'luminosity': u.L_sun
    }
    
    @classmethod
    def get_unit_system(cls, system='cgs'):
        """Get standard unit system."""
        systems = {
            'cgs': cls.CGS,
            'si': cls.SI,
            'astronomical': cls.ASTRONOMICAL
        }
        
        if system.lower() not in systems:
            raise ValueError(f"Unknown unit system: {system}")
        
        return systems[system.lower()]


# ============================================================================
# Utility Functions
# ============================================================================

def ensure_quantity(value, default_unit):
    """
    Ensure a value is a Quantity with appropriate units.
    
    Parameters
    ----------
    value : float, array, or Quantity
        Input value
    default_unit : astropy.units.Unit
        Unit to apply if value has no units
        
    Returns
    -------
    quantity : Quantity
        Value with units
    """
    if hasattr(value, 'unit'):
        return value
    else:
        return value * default_unit


def strip_units(quantity):
    """
    Strip units from a Quantity, returning just the numerical value.
    
    Parameters
    ----------
    quantity : Quantity
        Input quantity with units
        
    Returns
    -------
    value : array or float
        Numerical value without units
    """
    if hasattr(quantity, 'value'):
        return quantity.value
    else:
        return quantity


def convert_to_standard(quantity, quantity_type='flux'):
    """
    Convert quantity to standard units for given type.
    
    Parameters
    ----------
    quantity : Quantity
        Input quantity
    quantity_type : str
        Type of quantity ('flux', 'wavelength', 'temperature', etc.)
        
    Returns
    -------
    converted : Quantity
        Quantity in standard units
    """
    standard_units = {
        'flux': u.erg / u.s / u.cm**2 / u.Hz,
        'wavelength': u.micron,
        'frequency': u.Hz,
        'temperature': u.K,
        'luminosity': u.erg / u.s,
        'distance': u.pc,
        'mass': u.M_sun,
        'time': u.yr
    }
    
    if quantity_type not in standard_units:
        warnings.warn(f"Unknown quantity type: {quantity_type}")
        return quantity
    
    try:
        return quantity.to(standard_units[quantity_type])
    except u.UnitConversionError:
        warnings.warn(f"Cannot convert {quantity.unit} to standard "
                     f"{quantity_type} units ({standard_units[quantity_type]})")
        return quantity


def check_unit_consistency(*quantities):
    """
    Check that multiple quantities have consistent units.
    
    Parameters
    ----------
    *quantities : Quantity
        Quantities to check for unit consistency
        
    Returns
    -------
    consistent : bool
        True if all quantities have equivalent units
    """
    if len(quantities) < 2:
        return True
    
    reference_unit = quantities[0].unit
    
    for qty in quantities[1:]:
        if not qty.unit.is_equivalent(reference_unit):
            return False
    
    return True


# ============================================================================
# Export commonly used units and functions
# ============================================================================

# Make commonly used units easily accessible
__all__ = [
    # Custom units
    'habing_unit', 'draine_unit', 'mjy_per_sr', 'jy_per_arcsec2',
    
    # Main classes
    'SpectralUnits', 'ISRFUnits', 'TemperatureUnits', 'CoordinateUnits',
    'StandardUnits',
    
    # Utility functions
    'validate_spectral_units', 'ensure_quantity', 'strip_units',
    'convert_to_standard', 'check_unit_consistency',
    
    # Decorators
    'require_units',
    
    # Exceptions
    'UnitError'
]