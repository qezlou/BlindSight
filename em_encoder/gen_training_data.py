"""
A module to add an emission line on topf the empty fiber spectra in HETDEX.
"""

import numpy as np
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18 as cosmo



class EmissionLine:
    """
    Class to handle the generation of emission lines for mock spectra.

    """
    
    def __init__(self, wave= None, R=800, delta_v=300, size=10_000_000):
        """
        Initialize and get the emission line profile and continuum.
        Parameters:
        ------------
        - wave (array): Wavelength array in Angstroms. If None, a default range
        from 3550 to 5500 Angstroms is used.
        - R (int): Spectral resolution of the instrument.
        - delta_v (int): Velocity dispersion in km/s, should come from theory.
        - size (int): Number of sources to draw.
        ------------
        Computed attributes:
        - log_profile (array): Logarithm of the emission line profile.
        - log_cont (array): Logarithm of the continuum.

        # TODO: 
        1. Replace the constant equivalent width with a distribution from Karl's work.
        2. Tweak the luminosity function to match the observed data. Maybe iteratively as 
        Karl is doing.
        """

        if wave is None:
            # Default wavelength range in Angstroms
            self.wave = np.arange(3550, 5500, 2)
        # Spectral resolution of the instrument
        self.R = R 
         # Angstroms, rest-frame equivalent width of the emission line
         # FIX: To be replaced with Karl's distribution 
        self.CONSTANT_EW = 4
        self.size = size  # Number of sources to draw
        self.lya_rest = 1215.67  # Angstroms, rest wavelength of Lyman-alpha
        self.z_min = self.wave[0] / self.lya_rest - 1  # Minimum redshift based on the wavelength range
        self.z_max = self.wave[-1] / self.lya_rest - 1
        self.z = self.draw_z_distribution()
        
        # Velocity dispersion in km/s, should come from theory
        self.delta_v = delta_v
        self.log_profile, self.log_cont = self.draw_line_profile()

    def draw_line_profile(self):
        """
        Generate a Gaussian line profile.

        Parameters:
        wave (array): Wavelength array.
        line_center (float): Center of the emission line.
        line_width (float): Width of the emission line.
        amplitude (float): Amplitude of the emission line.

        Returns:
        array: Gaussian profile of the emission line.
        """

        line_center = self.lya_rest * (1 + self.z)  # Convert to observed wavelength
        # Get the FWHM of the line
        fwhm = self.draw_fwhm(line_center)
        sigma = fwhm / 2.355

        # Draw line fluxes
        log_flux = self.draw_line_flux()
        log_amplitude = log_flux - np.log10( sigma * np.sqrt(2 * np.pi))
        log_profile = np.zeros((self.size, self.wave.size))
        for i in range(self.size):
            log_profile[i,:] = log_amplitude[i] - 0.5 * ( (self.wave-line_center[i])/sigma[i])**2 / np.log10(np.exp(1))

        # Continuum based on the EW distribution
        ew_rest = self.draw_equivalent_width()
        ew_obs = ew_rest * (1 + self.z)
        log_cont = log_flux - np.log10(ew_obs)

        return log_profile, log_cont

    def draw_fwhm(self, line_center):
        """
        Draw the full width at half maximum (FWHM) of the emission line.
        Combines instrumental and intrinsic FWHM based on the line center.
        # NOTE: We have assumed a constant intrinsic FWHM based on the velocity dispersion.
        Parameters:
        ------------
        - line_center (float): Center of the emission line in Angstroms.
        Returns:
        ------------
        - fwhm_total (float): Total FWHM of the emission line in Angstroms.
        """

        # Instrumental and intrinsic in Angstroms
        fwhm_instr = line_center / self.R
        fwhm_intr = line_center * (self.delta_v / 3e5)
        # Combined FWHM
        fwhm_total = np.sqrt(fwhm_instr**2 + fwhm_intr**2)
        return fwhm_total

    def draw_equivalent_width(self):
        """
        Draw a list of rest-frame equivalent widths from a distribution.
        NOTE: For now we assume a constant number for EW. 4 Angstroms.
        """
        # For now, we assume a uniform distribution over EW
        return np.full(self.size, self.CONSTANT_EW)

    def draw_line_flux(self):
        """
        Draw a list of line fluxes from a theoretical distribution.
        Returns:
        ------------
        - log_flux (array): Logarithm of the fluxes in erg/s/cm^
        """
        # Draw a luminosity from the luminosity function
        logL = self.draw_luminosity_func(closest_z=2.2)
        # Luminosity distance in cm
        dL = cosmo.luminosity_distance(self.z).to('cm').value
        # Convert luminosity to flux
        log_flux = logL - np.log10(4 * np.pi * dL**2)  

        return log_flux

    def draw_z_distribution(self):
        """
        Draw a redshift distribution for the sources.
        NOTE: For, now we assume a uniform distribution over z.
        """
        # For now, we assume a uniform distribution over z
        # In the future, we can use a more realistic distribution
        return np.random.uniform(self.z_min, self.z_max, size=self.size)

    def draw_luminosity_func(self, logL=None, closest_z=2.2):
        """
        parametrized schechter fit: Ouchine+20 review arxiv:2012.07960
        Draw a list of luminosities from the luminosity function.
        Parameters:
        ------------
        - logL (array): Logarithm of the luminosity. If None, a
        default range is used.
        - closest_z (float): The redshift for which the luminosity function is defined.
        Returns:
        ------------
        - sampled_logL (array): Sampled luminosities from the luminosity function.
        """
        if logL is None:
            logL = np.linspace(42, 44, 10_000_000)
        
        if closest_z == 2.2:
            self.lum_model = 'Konno+16'
            # Konno+16 parameters for the luminosity function
            alpha = -1.8 # It was fixed during fitting
            logL_star = 42.0+np.log10(4.87)
            logPhi_star = -4.0+np.log10(3.37)
        elif closest_z == 3.1:
            self.lum_model = 'Ouchi+08 + Konno+16'
            # Ouchi+04 parameters for the luminosity function
            alpha = -1.8 # It was fixed during fitting
            logL_star = 42.0+np.log10(8.49)
            logPhi_star = -4.0+np.log10(3.90)
        else:
            raise ValueError("Unsupported redshift for luminosity function: {}".format(closest_z))

        # Call the Schechter function to calculate the probability distribution
        # We assume uniform distribution over z here
        prob_logL = self.schechter_prob(logL, alpha, 10**logPhi_star, 10**logL_star)
        
        # Using inverse transform sampling to draw luminosities from the distribution
        cdf = np.cumsum(prob_logL)
        cdf /= cdf[-1]  # Normalize the CDF
        sampler = interp1d(cdf, logL, bounds_error=False, fill_value=(logL[0], logL[-1]))
        sampled_logL = sampler(np.random.uniform(0, 1, self.size))

        return sampled_logL

        
    def schechter_prob(self, logL, alpha, phi_star, L_star):
        """
        Schechter function for the luminosity function. 
        Eq 6 from Ouchine+20 review arxiv:2012.07960

        Returns:
        array: Schechter function values for the log luminosity:
        phi_logL dlogL = phi_star * 10**( (alpha + 1) * (logL - logL_star) ) * np.exp(-10**(logL - logL_star)) dlogL
        Parameters:
        logL (array): Logarithm of the luminosity.
        alpha (float): The faint-end slope of the luminosity function.
        phi_star (float): The normalization factor of the luminosity function.
        L_star (float): The characteristic luminosity of the luminosity function.
        """
        
        logL_star = np.log10(L_star)

        #phi_logL = np.log(10) * phi_star * 10**( (alpha + 1) * (logL - logL_star) ) * np.exp(-10**(logL - logL_star))
        # Modeling phi_logL up to a normalization constant, as we only want the probability distribution
        prob_logL =  10**( (alpha + 1) * (logL - logL_star) ) * np.exp(-10**(logL - logL_star))

        # Normalize the probability distribution
        prob_logL /= np.trapz(prob_logL, logL)

        return prob_logL

class FiberSpectra:
    """
    Class to handle the get observed spectra from the HETDEX database.
    """
    def __init__(self):
        
        self.wave, self.spectra = self.get_empty_fiber_spectra()
        pass
    
    def get_empty_fiber_spectra(self):
        """
        Get the empty fiber spectra from the HETDEX database.

        Returns:
        wave (array): Wavelength array of the empty fiber spectra.
        spectra (array): Empty fiber spectra.
        """

        # This method should connect to the HETDEX database and retrieve the empty fiber spectra.
        pass

class MockSpectra:
    """
    Build Mock spectra: 
        1. Draw an emission line from a theoretical distribution
        2. Add the lines to random "Empty Fiber" spectra directly from the observations
    """
    def __init__(self, size):
        """
        Initialize the MockSpectra class with a redshift value.
        """
        #self.fiber_spectra = FiberSpectra()
        self.emission_line = EmissionLine(size=100)  # Initialize with appropriate values

        #self.wave, self.spectra = self.get_mock_spectra()

    def get_mock_spectra(self):
        """
        Generate mock spectra by adding emission lines to empty fiber spectra.

        Returns:
        array: Mock spectra with emission lines added.
        
        Returns:
        array: Mock spectra with emission lines added.
        """
        empty_spectra = self.fiber_spectra.get_empty_fiber_spectra()
        # Logic to add emission lines to the empty spectra
        pass

    