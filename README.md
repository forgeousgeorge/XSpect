# XSpect
Python3 code to measure equivalent widths from high-resolution spectra. Can be used in combination with pymoogi or MOOG

## XSpect - EW (Equivalent Widths)
 - Opens Keck HIRES .fits files
 - Normalizes reduced data
 - Applies Waveshift correction (comparison spectrum required)
 - Measures Equivalent Widths of input line list (line list required, sample provided)
 - Outputs pymoogi readable file used to derive and determine stellar parameters and elemental abundances

### Python Required Packages
 - Numpy
 - Matplotlib
 - Scipy
 - Astropy 
 - George - Gaussian Process Regression
