# XSpect
Python3 code to measure equivalent widths from high-resolution spectra. Output file can be used with pymoogi or MOOG.

## Install
pip install XSpect-EW

## XSpect - EW (Equivalent Widths)
 - Loads spectral data from Keck HIRES fits files
 - Includes options for loading data from arrays
 - Normalizes reduced data
 - Applies waveshift correction (comparison spectrum required)
 - Measures equivalent widths of input line list (line list required, sample provided)
 - Outputs pymoogi readable file used to derive and determine stellar parameters and elemental abundances

### Python Required Packages
 - Numpy
 - Matplotlib
 - Scipy
 - Astropy 
 - george - Gaussian Process Regression
