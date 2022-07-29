"""Python3 - User guided equivalent wideth measurement and analysis tool for stellar absorption spectra """

__version__ = "dev"

import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import fmin
from scipy.stats import chisquare
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from numpy.random import multivariate_normal
from astropy.io import fits
from scipy.optimize import curve_fit
import glob
import george
from george import kernels
from scipy.optimize import minimize
import pickle
import copy
from scipy.spatial.distance import cdist
from numpy.linalg import inv
from numpy.linalg import slogdet
import subprocess

#----------------------------------------------------------Functions-------------------------------------------------------#
def plot_line_info(star, name, filt = None):
    fig = plt.figure(figsize = (10,5))
    info_plot = fig.add_subplot(111)
    if filt == None:
        info_plot.errorbar(star.lines, star.lines_ew, yerr = star.lines_ew_err, fmt='.', zorder = 2, ecolor='k', c = 'k')
        info_plot.scatter(star.lines, star.lines_ew, c= star.lines_gauss_Xsquare, cmap = plt.cm.Reds, edgecolors='k', zorder = 3, s = 30)
        info_plot.scatter(star.lines[star.lines_check_flag], star.lines_ew[star.lines_check_flag], c = 'w', edgecolors= 'r',  zorder = 0, s = 100)
    else:
        info_plot.errorbar(star.lines[filt], star.lines_ew[filt], yerr = star.lines_ew_err[filt], fmt='.', zorder = 2, ecolor='k', c = 'k')
        info_plot.scatter(star.lines[filt], star.lines_ew[filt], c= star.lines_gauss_Xsquare[filt], cmap = plt.cm.Reds, edgecolors='k', zorder = 3, s = 30)
        info_plot.scatter(star.lines[filt][star.lines_check_flag[filt]], star.lines_ew[filt][star.lines_check_flag[filt]], c = 'w', edgecolors= 'r',  zorder = 0, s = 100)
        
    info_plot.grid()
    info_plot.set_title(name, size = 20)
    info_plot.set_xlabel(r'$\rm Wavelength\ (nm)$', size = 15)
    info_plot.set_ylabel(r'$\rm Equivalent\ Width\ (mA)$', size = 15)
    #plt.savefig(name+'_line_ew_info.pdf')
    plt.show()
    
def plot_comparison_res(star,hand_measured, name,xy = [0,100], filt = None):
    fig = plt.figure(figsize = (10,5))
    #top plot
    info_plot = fig.add_subplot(211)
    if filt == None:
        info_plot.errorbar(hand_measured, star.lines_ew, yerr = star.lines_ew_err, fmt='.', zorder = 2,ecolor='k', c = 'k')
        info_plot.scatter(hand_measured, star.lines_ew, c= star.lines_gauss_Xsquare, cmap = plt.cm.Reds, edgecolors='k', zorder = 3, s = 30)
        info_plot.scatter(hand_measured[star.lines_check_flag], star.lines_ew[star.lines_check_flag], c = 'w', edgecolors= 'r',  zorder = 0, s = 100)
    else:
        info_plot.errorbar(hand_measured[filt], star.lines_ew[filt], yerr = star.lines_ew_err[filt], fmt='.', zorder = 2,ecolor='k', c = 'k')
        info_plot.scatter(hand_measured[filt], star.lines_ew[filt], c= star.lines_gauss_Xsquare[filt], cmap = plt.cm.Reds, edgecolors='k', zorder = 3, s = 30)
        info_plot.scatter(hand_measured[filt][star.lines_check_flag[filt]], star.lines_ew[filt][star.lines_check_flag[filt]], c = 'w', edgecolors= 'r',  zorder = 0, s = 100)
        
    info_plot.plot([xy[0],xy[1]],[xy[0],xy[1]], 'k--')
    info_plot.set_title(name, size = 20)
    info_plot.grid()
    info_plot.set_ylabel(r'$\rm Auto\ Measured\ (mA)$', size = 15)
    
    #residuals plot
    res_plot = fig.add_subplot(212, sharex=info_plot)
    if filt == None:
        star_res_values = star.lines_ew - hand_measured
        res_plot.errorbar(hand_measured, star_res_values, yerr = star.lines_ew_err, fmt='.', zorder = 2,ecolor='k', c = 'k')
        res_plot.scatter(hand_measured, star_res_values, c= star.lines_gauss_Xsquare, cmap = plt.cm.Reds, edgecolors='k', zorder = 3, s = 30)
        res_plot.scatter(hand_measured[star.lines_check_flag], star_res_values[star.lines_check_flag], c = 'w', edgecolors= 'r',  zorder = 0, s = 100)
    else:
        star_res_values = star.lines_ew - hand_measured
        res_plot.errorbar(hand_measured[filt], star_res_values[filt], yerr = star.lines_ew_err[filt], fmt='.', zorder = 2,ecolor='k', c = 'k')
        res_plot.scatter(hand_measured[filt], star_res_values[filt], c= star.lines_gauss_Xsquare[filt], cmap = plt.cm.Reds, edgecolors='k', zorder = 3, s = 30)
        res_plot.scatter(hand_measured[filt][star.lines_check_flag[filt]], star_res_values[star.lines_check_flag], c = 'w', edgecolors= 'r',  zorder = 0, s = 100)
    #plt.savefig(name+'_ew_comparison.pdf')
    res_plot.plot([xy[0],xy[1]],[0,0],'k--')
    res_plot.set_xlabel(r'$\rm Hand\ Measured\ (mA)$', size = 15)
    res_plot.grid()
    plt.tight_layout()
    plt.show()

def get_line_window(line, wave, flux, left_bound, right_bound, 
                    line_input, window_size = 1.5):
    boundaries = [0,0]
    #if no line is specified, auto fine best line guess
    if line_input == 0.0:
        #find line tip
        left_look = np.where((wave <= line)&(wave >= line - 0.1))
        right_look = np.where((wave >= line)&(wave <= line + 0.1))
        #find min
        mins = [flux[left_look].min(),flux[right_look].min()]
        best_line_guess = wave[np.where(flux == np.min(mins))][0]
    else:
        best_line_guess = line_input
        
    #get_window around line
    window = np.where((wave >= best_line_guess-window_size/2.0)&(wave <= best_line_guess+window_size/2.0))

    #calc derivative
    dy = np.gradient(flux[window])
    dy_std = dy.std()

    #if no left or right bound given set using std
    auto_bound_l = False
    auto_bound_r = False
    if left_bound == 0:
        dy_l = dy_std/2.0
        auto_bound_l = True
    if right_bound == 0:
        dy_r = dy_std/2.0
        auto_bound_r = True
    
    #if no line boundaries specified auto find boundaries
    if auto_bound_l:
        left_look = np.where(wave[window] <= best_line_guess - 0.05)
        dy1_left = np.where((dy[left_look] < dy_l)&(dy[left_look] > (-1)*dy_l))
        if len(wave[window][left_look][dy1_left]) ==0:
            print('line ',line,' very close to edge or dy selection value too small')
            print('will attempt to remeasure, if not possible, add line to exclude lines list in .measure_all_ew() function')
            plt.clf()
            plt.plot(wave[window],flux[window])
            plt.plot([line,line],[0.95,1.0], 'k')
            plt.annotate(str(line), xy=[line,1.01])
            plt.plot([best_line_guess,best_line_guess],[0.95,1.0], 'k--')
            plt.annotate(str(best_line_guess), xy=[best_line_guess,1.01])
            plt.show()
            return 1,1,0,0

        else:
            boundaries[0] = wave[window][left_look][dy1_left][-1]
    else:
        boundaries[0] = left_bound
    if auto_bound_r:
        right_look = np.where(wave[window] >= best_line_guess + 0.05)
        dy1_right = np.where((dy[right_look] < dy_r)&(dy[right_look] > (-1)*dy_r))
        if len(wave[window][right_look][dy1_right]) ==0:
            print('line ',line,' very close to edge or dy selection value too small')
            print('will attempt to remeasure, if not possible, add line to exclude lines list in .measure_all_ew() function')
            plt.clf()
            plt.plot(wave[window],flux[window])
            plt.plot([line,line],[0.95,1.0], 'k')
            plt.annotate(str(line), xy=[line,1.01])
            plt.plot([best_line_guess,best_line_guess],[0.95,1.0], 'k--')
            plt.annotate(str(best_line_guess), xy=[best_line_guess,1.01])
            plt.show()
            return 0,0,1,1

        else:
            boundaries[1] = wave[window][right_look][dy1_right][0]
    else:
        boundaries[1] = right_bound

    return window,best_line_guess, boundaries,dy

def gauss_model(x,A,mu,sigma, baseline): 
    return A*np.exp(-(x-mu)**2/2/sigma**2) + baseline

def gfit(wav,flux,wav_cen, fwhm):
        sigma = fwhm/2.355
        #limit window of search center +- 2*fwhm to exclude other emission lines
        gwave = np.where((wav >= wav_cen-30)&(wav <= wav_cen+30))

        #find better center to account for small doppler shift within same window of search
        bet_cen = wav[np.where(flux == flux[gwave].max())[0][0]]

        #Initial value for guass max value guess from max of curve
        guess = flux[np.where(flux == flux[gwave].max())[0][0]]

        #Set parameters for gauss curve fit
        p0 = [guess,bet_cen,sigma, 0.]
        bf,cov = curve_fit(gauss_model,wav[gwave],flux[gwave],p0)

        #plt.plot(wav[gwave], flux[gwave], 'r')
        return bf, np.sqrt(np.diag(cov)), p0
    
def gfit_simple(x_array, y_array, mu, sigma, baseline):
    A = y_array.max()
    p0 = [A, mu, sigma, baseline]
    try:
        bf, cov = curve_fit(gauss_model, x_array, y_array, p0)
        return bf, np.sqrt(np.diag(cov)), p0
    except:
        bf, cov = [0,0,0,0],None
        return bf, cov, p0

def gauss_ew(a, fwhm):
    if a == 0 or fwhm == 0:
        return 0
    else:
        return 500.*a*np.sqrt(np.pi/np.log(2))*fwhm #From Adamow pyMOOG ew measure

#Gaussian Process code and kernals taken from LSSTC DSFP notebook
def SEKernel(par, x1, x2):
    A, Gamma = par
    D2 = cdist(x1.reshape(len(x1),1), x2.reshape(len(x2),1), metric = 'sqeuclidean')
    return A*np.exp(-Gamma*D2)

def Pred_GP(CovFunc, CovPar, xobs, yobs, eobs, xtest):
    # evaluate the covariance matrix for pairs of observed inputs
    K = CovFunc(CovPar, xobs, xobs) 
    # add white noise
    K += np.identity(xobs.shape[0]) * eobs**2
    # evaluate the covariance matrix for pairs of test inputs
    Kss = CovFunc(CovPar, xtest, xtest)
    # evaluate the cross-term
    Ks = CovFunc(CovPar, xtest, xobs)
    # invert K
    Ki = inv(K)
    # evaluate the predictive mean
    m = np.dot(Ks, np.dot(Ki, yobs))
    # evaluate the covariance
    cov = Kss - np.dot(Ks, np.dot(Ki, Ks.T))
    return m, cov

def NLL_GP(p,CovFunc,x,y,e):
    # Evaluate the covariance matrix
    K = CovFunc(p,x,x)
    # Add the white noise term
    K += np.identity(x.shape[0]) * e**2
    # invert it
    Ki = inv(K)
    # evaluate each of the three terms in the NLL
    term1 = 0.5 * np.dot(y,np.dot(Ki,y))
    term2 = 0.5 * slogdet(K)[1]
    term3 = 0.5 * len(y) * np.log(2*np.pi)
    # return the total
    return term1 + term2 + term3

def make_line(x,m,b):
    return m*x+b

def combine_files(empty_obj,objects = []):
    final_wavelength = []
    final_flux = []
    final_norm_flux = []
    final_shifted_wavelength = []
    final_estimated_shift = []
    final_continuum = []
    final_obs_err = []
    final_pred_all = []
    final_pred_var_all = []
    final_gain = []
    
    for j in objects:

        for i in range(len(j.flux)):
            final_norm_flux.append(j.normalized_flux[i])
            final_shifted_wavelength.append(j.shifted_wavelength[i])
            final_wavelength.append(j.wavelength[i])
            final_flux.append(j.flux[i])
            final_estimated_shift.append(j.estimated_shift[i])
            final_continuum.append(j.continuum[i])
            final_obs_err.append(j.obs_err[i])
            final_pred_all.append(j.pred_all[i])
            final_pred_var_all.append(j.pred_var_all[i])
            final_gain.append(j.gain[i])
               
    empty_obj.wavelength = np.array(final_wavelength)
    empty_obj.flux = np.array(final_flux)
    empty_obj.shifted_wavelength = np.array(final_shifted_wavelength)
    empty_obj.normalized_flux = np.array(final_norm_flux)
    empty_obj.estimated_shift = np.array(final_estimated_shift)
    empty_obj.continuum = np.array(final_continuum)
    empty_obj.obs_err = np.array(final_obs_err)
    empty_obj.pred_all = np.array(final_pred_all)
    empty_obj.pred_var_all = np.array(final_pred_var_all)
    empty_obj.gain = np.array(final_gain)
    del final_wavelength
    del final_flux
    del final_norm_flux
    del final_shifted_wavelength
    del final_estimated_shift
    del final_continuum
    del final_obs_err
    del final_pred_all
    del final_pred_var_all
    del final_gain
    
    return empty_obj

def reduce_cc(x,y,lines,lines_removed,limit=0.12):
    #check correlation before going further
    cc = np.corrcoef(x,y)
    print('starting cc', cc[0,1])
    
    if abs(cc[0,1]) < limit:
        print('cc good enough')
        return lines,x,y,lines_removed
    
    check_ccs = np.zeros(len(x))
    
    #remove largest cc difference
    for i in range(len(x)):
        new_x = np.delete(x,i)
        new_y = np.delete(y,i)
        new_cc = np.corrcoef(new_x,new_y)
        check_ccs[i] = new_cc[0,1]
    
    #Calculate differences
    diffs = [abs(cc[0,1])- abs(j) for j in check_ccs]
    #which gives largest difference?
    biggest_diff = np.where(diffs == max(diffs))[0][0]
    #remove that one line
    lines_removed.append([lines[biggest_diff],x[biggest_diff],y[biggest_diff]])
    x = np.delete(x,biggest_diff)
    y = np.delete(y,biggest_diff)
    lines = np.delete(lines,biggest_diff)
    print('line removed: ', lines_removed)
    
    #recalculate cc
    cc = np.corrcoef(x,y)
    print('ending cc', cc[0,1])
    
    #Call function again to remove lines until 0.12 is passed
    lines,x,y,lines_removed = reduce_cc(x,y,lines,lines_removed)
    
    return lines,x,y,lines_removed

def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def make_plots_folder():
    folder_name = 'line_plots'
    #check if folder exists
    filenames = glob.glob('*')
    if folder_name in filenames:
        pass
    else:
        #make folder if not
        cmd = 'mkdir '+folder_name
        subprocess.call(cmd, shell=True)
    

def replace_w(array, replace_value, replace_with = 'med' ):
    """
        Use to replace individual values with either median or average of the order
        or with a specific value
        array format: [[order1],[order2],...] orderN = [x1,x2,x3,...]
        replace_value - value to be replaced
        repace_with - 'med', 'avg', or value to replace with
    """

    for i in range(len(array)):
        gd_0 = np.where(array[i] == replace_value)
        gd = np.where(array[i] != replace_value)

        for j in gd_0:
            if replace_with == 'med':
                array[i][j] = np.median(array[i][gd])
            elif replace_with == 'avg':
                array[i][j] = np.average(array[i][gd])
            else:
                array[i][j] = replace_with
            
    return None


#--------------------------------------------------Classes--------------------------------------------------#
class Spectrum_Data():
    def __init__(self, filename, KECK_file = True, spectx=False, specty=False, order_split = (False,3500)):
        """
            filenmae - used to name plots and files
            KECK_file - True if loading KECK fits file
            spectx, specty - input spectrum if loading from arrays 
                if not using order_split:
                    input format: [[order1],[order2],...] orderN = [x1,x2,x3,...]
                if using order_split:
                    input format: [x1,x2,x3,...]
        """
        self.filename = filename
        self.gain = None
        if KECK_file:
            self.wavelength, self.flux = self.read_spec()
        else:
            #split input array into orders 
            if order_split[0]:
                #Try targetting about 3500 points per order, too many points per order will slow code down
                order_count = int(len(spectx)/order_split[1])
                order_split_wave = np.array_split(spectx, order_count)
                order_split_flux = np.array_split(specty, order_count)
                self.wavelength = order_split_wave
                self.flux = order_split_flux
                del order_split_wave
                del order_split_flux
                print('If values need to be replaced, use replace_w() function')
            else:
                self.wavelength = spectx
                self.flux = specty
        self.normalized_flux = copy.deepcopy(self.flux) #deepcopy creates new object
        self.combined_flux = None
        self.shifted_wavelength = copy.deepcopy(self.wavelength)
        self.estimated_shift = np.zeros(len(self.wavelength))
        self.rv = None #km/s
        
        #continuum information
        self.continuum = np.full(len(self.wavelength), None)
        #print('cont array empty', self.continuum)
        self.pred_all = np.full(len(self.wavelength), None)
        self.pred_var_all = np.full(len(self.wavelength), None)
        self.obs_err = np.full(len(self.wavelength), None)
        for i in range(len(self.wavelength)):
            false_array = np.full(len(self.wavelength[i]), False)
            #print('false array created', false_array)
            self.continuum[i] = false_array
            self.pred_all[i] = np.full(len(self.wavelength[i]), 0)
            self.pred_var_all[i] = np.full(len(self.wavelength[i]), 0)
            self.obs_err[i] = np.sqrt(self.flux[i])
        #print('empty continuum arrays created', self.continuum)
        #Old way of setting these variables (change back if above code causes problems)
        # self.continuum = np.full((len(self.wavelength),len(self.wavelength[0])), False)
        # self.pred_all = np.zeros((len(self.wavelength),len(self.wavelength[0])))
        # self.pred_var_all = np.zeros((len(self.wavelength),len(self.wavelength[0])))
        # self.obs_err = np.zeros((len(self.wavelength),len(self.wavelength[0])))
        
        #line information
        self.lines = None
        #line - extra parameters [0] - shift continuum, [1] - left boundary in Angstroms
        #[2] - right boundary in Angstroms, [3] - line center in Angstroms
        self.lines_exp = None
        #line - extra data [0] - element, [1] - excitation potential
        #[2] - gf, [3] - rad
        self.lines_exd = None
        #line - equivalent width
        self.lines_ew = None
        #line - equivalent width error
        self.lines_ew_err = None
        #line - equivalent width calculated by simpon's rule integration
        self.lines_ew_simp = None
        #line - equivalent width error from integraion
        self.lines_ew_simp_err = None
        #line - best fit parameters for gaussian fit
        self.lines_bf_params = None
        #line - X squared value for gaussian and data
        self.lines_gauss_Xsquare = None
        #line - X squared threshold value
        self.X_thresh = 0.003
        #line - X squared value above threshold or EW = 0
        self.lines_check_flag = None
        #used to switch between Adamow ew calculation and simpson's rule integration
        self.temp_line_ew = None
        self.temp_line_ew_err = None
        
    def normalize_all(self, window_width = 1.5, continuum_depth = 90):
        #loop through orders        
        for i in range(len(self.flux)):

            #use Gaussian Process to fit continuum
            self.normalize(i, window_width, continuum_depth)

            #Replace un-normalized points with value before it
            #This should only be replacing the last point in the 
            #spectrum that is always missed by normalize
            # err_est = self.obs_err[i]/self.pred_all[i]
            # non_norm_points = np.where(self.normalized_flux[i] > np.average(self.normalized_flux[i][self.continuum[i]]+err_est[self.continuum[i]]*100))
            # #replace non norm points
            # self.normalized_flux[i][non_norm_points] = self.normalized_flux[i][non_norm_points[0]-1]

        return None

    def normalize(self, order, window_width = 1.5, continuum_depth = 90, clip = [-999,-999]):
        if clip[0] != -999 and clip[1] != -999:
            #clipped = True
            clipl = np.where(self.wavelength[order] <= clip[0])[0][-1]
            clipr = np.where(self.wavelength[order] >= clip[1])[0][0]
        else:
            #clipped = False
            clipl = 0
            clipr = len(self.flux[order])
 
        err = np.sqrt(self.flux[order][clipl:clipr])
        continuum_scan_obj = Continuum_scan(window_width, continuum_depth)
        continuum_scan_obj.load_data(self.wavelength[order][clipl:clipr],self.flux[order][clipl:clipr])
        continuum_scan_obj.scan()
        cont = continuum_scan_obj.get_selected()
        del continuum_scan_obj
        
        #Gaussian Process to fit continuum
        kernel = np.var(self.flux[order][clipl:clipr][cont]) * kernels.Matern32Kernel(10)
        #print("cont", len(self.flux[order][cont]))
        #kernel = np.var(self.flux[order][cont]) * kernels.ExpSquaredKernel(10)
        gp = george.GP(kernel,mean=self.flux[order][clipl:clipr][cont].mean())
        gp.compute(self.wavelength[order][clipl:clipr][cont], err[cont])
        x_pred = self.wavelength[order][clipl:clipr].copy()
        pred, pred_var = gp.predict(self.flux[order][clipl:clipr][cont], x_pred, return_var=True)
        #print("ln-likelihood: {0:.2f}".format(gp1.log_likelihood(self.flux[order][cont])))
        params = [gp,self.flux[order][clipl:clipr][cont]]
        result = minimize(self.neg_ln_like, gp.get_parameter_vector(), args = params, jac=self.grad_neg_ln_like)
        #print(result)
        gp.set_parameter_vector(result.x)
        pred, pred_var = gp.predict(self.flux[order][clipl:clipr][cont], x_pred, return_var=True)
        #print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(self.flux[order][cont])))

        self.continuum[order][clipl:clipr] = cont
        self.pred_all[order][clipl:clipr] = pred
        self.pred_var_all[order][clipl:clipr] = pred_var
        self.obs_err[order][clipl:clipr] = err
        self.normalized_flux[order][clipl:clipr] = self.flux[order][clipl:clipr]/pred
        return None

    def grad_neg_ln_like(self,p, params):
        params[0].set_parameter_vector(p)
        neg_ln = (-1)*params[0].grad_log_likelihood(params[1])
        return neg_ln

    def neg_ln_like(self,p, params):
        params[0].set_parameter_vector(p)
        neg_ln = (-1)*params[0].log_likelihood(params[1])
        return neg_ln

    def S_N(self, rows = 5, cols = 4, save_plot = False):
        #Siganl to noise is overestimated compared to MAKEE output
        plt.clf()
        f = plt.figure(figsize=(20,15))
        plt.suptitle(str(self.filename) + ' S/N', size = 20)
        for i in range(len(self.wavelength)):
            order = i
            i += 1
            ax = f.add_subplot(rows, cols,i)
            ax.plot(self.wavelength[order],np.sqrt(self.flux[order]*self.gain[order]))
            ax.set_title(str(i))
            ax.grid()
        plt.tight_layout()
        if save_plot:
            plt.savefig(str(self.filename)+'_SNR.pdf')
        plt.show()

    def load_normalized(self, name):
        pathnames = glob.glob(name+'*'+'.npy')
        for i in range(len(pathnames)):
            if '_flux' in pathnames[i]:
                self.normalized_flux = np.load(pathnames[i])
                print('flux loaded')
            elif '_wavelength' in pathnames[i]:
                self.wavelength = np.load(pathnames[i])
                print('wavelength loaded')
            elif '_cont' in pathnames[i]:
                self.continuum = np.load(pathnames[i])
                print('continuum loaded')
            elif '_obs_err' in pathnames[i]:
                self.obs_err = np.load(pathnames[i])
                print('errors loaded')
            elif '_pred' in pathnames[i]:
                self.pred_all = np.load(pathnames[i])
                print('all preds loaded')
            elif '_pred_var' in pathnames[i]:
                self.pred_var_all = np.load(pathnames[i])
                print('all pred vars loaded')

    def save_normalized(self, name):
        #arrays are separated by order
        np.save(name+'_flux',self.normalized_flux)
        np.save(name+'_wavelength',self.shifted_wavelength)
        np.save(name+'_cont',self.continuum)
        np.save(name+'_obs_err',self.obs_err)
        np.save(name+'_pred',self.pred_all)
        np.save(name+'_pred_var',self.pred_var_all)
        return None
            
    def wave_shift(self, order, shift):
        self.shifted_wavelength[order] = self.wavelength[order] + shift
        self.estimated_shift[order] = shift

    def combine_spectra(self, spectB, resolution = 1000, shift=True):
        print('Use self.update_combined() when you are happy with the combined flux to override self.flux')
        #find corresponding orders that match self in spectB
        med_A = [np.median(self.shifted_wavelength[i]) for i in range(len(self.shifted_wavelength))]
        med_B = [np.median(spectB.shifted_wavelength[i]) for i in range(len(spectB.shifted_wavelength))]
        b_order = []
        #find corresponding B order
        for k in range(len(med_A)):
            diff = abs(med_A[k] - med_B)
            loc = np.where(diff == diff.min())
            b_order.append(loc)
            
        combined_flux_orders = np.zeros_like(self.flux)
        if shift:
            #first shift A to match B with higher accuracy (higher resolution)
            #may have to include a try statement for errors
            self.estimate_shift([spectB], shift_spacing=resolution)
            self.clean_shift()
        
        #loop through orders
        for i in range(len(self.shifted_wavelength)):
            
            #combining flux values for each wavelength value
            combined_flux = np.zeros(len(self.shifted_wavelength[i]))
            
            print('A order', i, 'B order', b_order[i][0][0])
            
            #loop through each shifted wavelength value
            for j in range(len(self.shifted_wavelength[i])):
                #difference between one shifted wavelength value and all B wavelength values
                #element closest to zero is location of closest wavelength values
                ed = 5
                if j < ed:
                    le = 0
                    re = j + ed
                elif j > len(self.shifted_wavelength[i])-ed:
                    le = j - ed
                    re = len(self.shifted_wavelength[i]) + 1
                else:
                    le = j - ed
                    re = j + ed
                near_point = spectB.wavelength[b_order[i][0][0]][le:re]
                diff_array = abs(self.shifted_wavelength[i][j] - near_point)
                loc = np.where(diff_array == diff_array.min())
                #add A flux with B flux at location where diff = 0
                combined_flux[j] = self.flux[i][j] + spectB.flux[b_order[i][0][0]][le:re][loc]
            #print(combined_flux)
            #collect flux values for each order
            combined_flux_orders[i] = combined_flux
            self.obs_err[i] = np.sqrt(combined_flux)
            
            plt.plot(self.shifted_wavelength[i], self.flux[i], label = 'A')
            plt.plot(spectB.wavelength[b_order[i][0][0]], spectB.flux[b_order[i][0][0]], label = 'B')
            plt.plot(self.shifted_wavelength[i], combined_flux, label = 'A+B')
            plt.xlim([np.median(self.shifted_wavelength[i]-(self.shifted_wavelength[i].max() - self.shifted_wavelength[i].min())/10),
                np.median(self.shifted_wavelength[i]+(self.shifted_wavelength[i].max() - self.shifted_wavelength[i].min())/10)])
            plt.grid()
            plt.legend()
            plt.show()
            print('#-----------------------#')
        #replace original flux for A with combined flux
        self.combined_flux = combined_flux_orders

    def update_combined(self):
        self.flux = self.combined_flux
        
    def estimate_shift(self, sun_spectra, shift_max = 5, shift_min = -5, shift_spacing = 100):
        #setup num orders, place holder for chi min, shifts array
        orders = len(self.wavelength)
        chi = np.zeros(shift_spacing)
        shifts = np.linspace(shift_min, shift_max, shift_spacing)

        #begin looping through each order in star spectrum
        for q in range(orders):
            order = q
            order_found = False
            order_mean = self.shifted_wavelength[order].mean()
                
            #loop through each included solar spectrum to find a matching order
            for j in range(len(sun_spectra)):
                sun = sun_spectra[j]

                #Loop through sun orders to find similar order
                for i in range(len(sun.wavelength)):
                    #use average wavelength value in an order to look for 
                    #matching solar order
                    if order_mean < sun.wavelength[i].max() and order_mean > sun.wavelength[i].min():
                        order_found = True
                        sun_range = sun.wavelength[i].max() - sun.wavelength[i].min()
                        sun_window = [sun.wavelength[i][0], sun.wavelength[i][-1]]
                        
                        #loop through specified shift values to find best match
                        for k in range(len(shifts)):
                            shift = shifts[k]
                            self.wave_shift(order,shift)
                            order_range = self.shifted_wavelength[order].max() - self.shifted_wavelength[order].min()
                            order_window = [self.shifted_wavelength[order][0], self.shifted_wavelength[order][-1]]

                            #    |------------|     -->    |------------|
                            #|------------|         --> |xx|----------|
                            if order_window[0] > sun_window[0]:
                                sun_window[0] = sun.wavelength[i][np.where(sun.wavelength[i] > order_window[0])][0]

                            #|------------|         --> |------------|
                            #    |------------|     -->     |-------|xxxxx| 
                            if order_window[1] < sun_window[1]:
                                sun_window[1] = sun.wavelength[i][np.where(sun.wavelength[i] < order_window[1])][-1]

                            compare_window_star = np.where((self.shifted_wavelength[order] >= order_window[0])&(self.shifted_wavelength[order] <= order_window[1]))
                            compare_window_sun = np.where((sun.wavelength[i] >= sun_window[0])&(sun.wavelength[i] <= sun_window[1]))
                            spect_interp = interp1d(self.shifted_wavelength[order][compare_window_star],self.normalized_flux[order][compare_window_star], kind = 'linear')
                            result_y = spect_interp(sun.wavelength[i][compare_window_sun])

                            diff = (result_y - sun.normalized_flux[i][compare_window_sun])**2
                            chi[k] = np.sum(diff)

                            #chi[k] = chisquare(result_y, constraint_value)[0]
                            #chi[k] = chisquare(result_y, sun.normalized_flux[i][compare_window_sun])[0]             
                        break
                if order_found:
                    break
            if not order_found:
                #if order is not found in solar spectrum, value is set to -999
                #to be cleaned later
                print('order '+str(order)+' in star not found in solar specturm')
                print('missing values can be interpolated/extrapolated using clean_shift() method')
                self.estimated_shift[order] = -999.0
            else:
                min_shift = shifts[np.where(chi == chi.min())][0]
                self.estimated_shift[order] = min_shift
                self.wave_shift(order,min_shift)
                
    def clean_shift(self):
        #remove orders not found
        gd = np.where(self.estimated_shift != -999)
        bad_points = np.where(self.estimated_shift == -999)[0]

        #get mean wavelength values
        means = np.zeros(len(self.shifted_wavelength))
        for i in range(len(self.estimated_shift)):
            means[i] = self.shifted_wavelength[i].mean()

        #get standard deviation of good points
        stds = self.estimated_shift[gd].std()

        #fit to good points
        best_fit = np.polyfit(means[gd], self.estimated_shift[gd], 1)
        x_range = np.linspace(means[gd][0],means[gd][-1],len(self.shifted_wavelength))
        line = make_line(x_range, best_fit[0], best_fit[1])

        #residuals of good points
        residuals = self.estimated_shift - line

        #remove points > std/2
        good_points = np.where((residuals < stds/2)&(residuals > -10))
        bad_points = (np.append(bad_points, np.where(residuals > stds/2)[0]),)

        #fit again with best points
        best_fit = np.polyfit(means[good_points], residuals[good_points], 1)
        x_range = np.linspace(means[good_points][0],means[good_points][-1],len(residuals))

        #line to be used to interpolate and extrapolate missing or bad points
        line2 = make_line(x_range, best_fit[0], best_fit[1])

        #interpolate or extrapolate bad points using line 2 and replace values
        for i in range(len(self.estimated_shift[bad_points])):
            current_index = bad_points[0][i]
            wave = means[current_index]
            self.estimated_shift[current_index] = make_line(wave, best_fit[0], best_fit[1]) + line[current_index]
            self.wave_shift(current_index, self.estimated_shift[current_index])
        
        #get radial velocity from slope of shifts
        best_fit, C = np.polyfit(means, self.estimated_shift*(-1), 1, cov=True)
        self.rv = (np.round(best_fit[0]*3e5,3), np.round(np.sqrt(np.diag(C))[1], 3))
        
    def load_lines(self, filename):
        self.lines = np.genfromtxt(filename, skip_header = 1, usecols = 0)
        elmnt = np.genfromtxt(filename, skip_header = 1, usecols = 1)
        ep = np.genfromtxt(filename, skip_header = 1, usecols = 2)
        gf = np.genfromtxt(filename, skip_header = 1, usecols = 3)
        rad = np.genfromtxt(filename, skip_header = 1, usecols = 4)
        self.lines_exd = np.zeros((len(self.lines),4))
        self.lines_exp = np.zeros((len(self.lines),4))
        self.lines_ew = np.zeros(len(self.lines))
        self.lines_ew_err = np.zeros(len(self.lines))
        self.lines_ew_simp = np.zeros(len(self.lines))
        self.lines_ew_simp_err = np.zeros(len(self.lines))
        self.lines_bf_params = np.array([None]*len(self.lines))
        self.lines_gauss_Xsquare = np.array([np.nan]*len(self.lines))
        self.lines_check_flag = np.array([False]*len(self.lines))
        for i in range(len(self.lines)):
            self.lines_exd[i] = np.array([elmnt[i],ep[i],gf[i],rad[i]])

    # def switch_ew_values(self):
    #     if self.temp_line_ew == None:
    #         print("switching from Adamow calculation for EW to Simpson's rule integration")
    #         self.temp_line_ew = self.lines_ew
    #         self.temp_line_ew_err = self.lines_ew_err

    #         self.lines_ew = self.lines_ew_simp
    #         self.lines_ew_err = self.lines_ew_simp_err
    #     else:
    #         print("switching from Simpson's rule integration for EW to Adamow calculation for EW")
    #         self.lines_ew = self.temp_line_ew
    #         self.lines_ew_err = self.temp_line_ew_err

    #         self.temp_line_ew = self.lines_ew_simp
    #         self.temp_line_ew_err = self.lines_ew_simp_err
        
    def make_ew_doc(self, name,doc_title='STARNAME, PROJECT, YEAR; '):
        doc = open(name, 'w')
        doc.write(doc_title+'Extended Fe Linelist based on the SWP (2010) paper plus additions from Ivan\n')
        removed_lines = []
        for i in range(len(self.lines)):
            if self.lines_ew[i] != 0.0: 
                wave = "  "+str(self.lines[i])
                elmnt = str(self.lines_exd[i][0])
                if self.lines_exd[i][0] < 10:
                    elmnt = '0'+elmnt
                while len(elmnt) < 6:
                    elmnt = elmnt+'0'
                ep = str(self.lines_exd[i][1])
                gf = str(self.lines_exd[i][2])
                rad = str(self.lines_exd[i][3])
                ew = str(np.round(self.lines_ew[i],3))
                err = str(np.round(self.lines_ew_err[i],3))
                current_line = "{0:14s}{1:11s}{2:8s}{3:15s}{4:17s}{5:10s}{6:5s}\n".format(wave,elmnt,ep,gf,rad,ew,err)
                doc.write(current_line)
            else:
                removed_lines.append(self.lines[i])
        doc.close()
        return np.array(removed_lines)
        
    def measure_ew(self, i, order, plot = False, ex_params = [0,0,0,0], save_plot = False, window_size = 1.5):
        #extra parameters [0] - shift continuum
        #                 [1] - left boundary in Angstroms
        #                 [2] - right boundary in Angstroms
        #                 [3] - line center in Angstroms
        norm = 1.0
        wind, found_line, line_bound,dy = get_line_window(self.lines[i],self.shifted_wavelength[order],self.normalized_flux[order],ex_params[1],ex_params[2],ex_params[3], window_size)

        if [wind,found_line,line_bound,dy] == [1,1,0,0]:
            pass
            if order != 0:
                concat_wave = np.concatenate((self.shifted_wavelength[order-1],self.shifted_wavelength[order]), axis=None)
                concat_flux = np.concatenate((self.normalized_flux[order-1],self.normalized_flux[order]), axis=None)
                concat_err = np.concatenate((self.obs_err[order-1],self.obs_err[order]), axis=None)
                concat_pred = np.concatenate((self.pred_all[order-1],self.pred_all[order]), axis=None)
                wind, found_line, line_bound, dy = get_line_window(self.lines[i], concat_wave, concat_flux,ex_params[1],ex_params[2],ex_params[3], window_size)
                measure_x_array = concat_wave[wind]
                measure_y_array = concat_flux[wind]
                temp_err_array = concat_err[wind]
                temp_pred_array = concat_pred[wind]
        elif [wind,found_line,line_bound,dy] == [0,0,1,1]:
            pass
            if self.shifted_wavelength[order] != self.shifted_wavelength[-1]:
                concat_wave = np.concatenate((self.shifted_wavelength[order],self.shifted_wavelength[order+1]), axis=None)
                concat_flux = np.concatenate((self.normalized_flux[order],self.normalized_flux[order+1]), axis=None)
                concat_err = np.concatenate((self.obs_err[order],self.obs_err[order+1]), axis=None)
                concat_pred = np.concatenate((self.pred_all[order],self.pred_all[order+1]), axis=None)
                wind, found_line, line_bound, dy = get_line_window(self.lines[i], concat_wave, concat_flux,ex_params[1],ex_params[2],ex_params[3], window_size)
                measure_x_array = concat_wave[wind]
                measure_y_array = concat_flux[wind]
                temp_err_array = concat_err[wind]
                temp_pred_array = concat_pred[wind]
        else:
            measure_x_array = self.shifted_wavelength[order][wind]
            measure_y_array = self.normalized_flux[order][wind]
            temp_err_array = self.obs_err[order][wind]
            temp_pred_array = self.pred_all[order][wind]

        other_than_line = np.where((measure_x_array <= line_bound[0])|(measure_x_array >= line_bound[1]))
        only_line = np.where((measure_x_array >= line_bound[0])|(measure_x_array <= line_bound[1]))
        flat_wing = measure_y_array.copy() + ex_params[0]
        flat_wing[other_than_line] = norm 
        #highlight points within errors of continuum (or 1.0)
        upper_cont_bounds = measure_y_array+ ex_params[0] + 2*temp_err_array/temp_pred_array
        lower_cont_bounds = measure_y_array+ ex_params[0] - 2*temp_err_array/temp_pred_array
        points_within_norm = np.where((norm > lower_cont_bounds)&(norm < upper_cont_bounds))
        #GP fit
        xtest = np.linspace(measure_x_array[0], measure_x_array[-1], len(measure_x_array))
        m,C=Pred_GP(SEKernel,[1,100],measure_x_array,flat_wing,2*temp_err_array/temp_pred_array, xtest)
        try:
            samples = multivariate_normal(m,C,500)
        except:
            print('SVD did not converge, setting samples to 0')
            print('If line is close to an edge, try remeasuring line with a smaller window size')
            samples = np.array([0]*500)

        m_plot=m.copy()
        m = (-1)*(m-1)
        samp_ew = np.zeros(len(samples))
        a_values = np.zeros(len(samples))
        mu_values = np.zeros(len(samples))
        sig_values = np.zeros(len(samples))
        base_values = np.zeros(len(samples))
        simp_values = np.zeros(len(samples))
        plot_gaussian = False
        for j in range(len(samples)):
            #plt.plot(xtest,(-1)*(samples[j]-1), 'b--')
            bf, err, p0 = gfit_simple(xtest, (-1)*(samples[j]-1), found_line, 0.5,0)
            #print('best fit:', bf, err, p0)
            #if bf[0] > 0.0:
            if abs(gauss_ew(bf[0], bf[2]*2.355)) > 2 and abs(gauss_ew(bf[0], bf[2]*2.355)) < 200:
                samp_ew[j] = abs(gauss_ew(bf[0], bf[2]*2.355))
                a_values[j] = bf[0]
                mu_values[j] = bf[1]
                sig_values[j] = abs(bf[2])
                base_values[j] = bf[3]

                #simpson's rule integration
                # line_inpterp = interp1d(measure_x_array, flat_wing, kind='linear', bounds_error = False)
                # x = np.linspace(flat_wing[0], flat_wing[-1],100) 
                # result_y = line_inpterp(x)
                
                #y =  gauss_model(x,bf[0],bf[1],bf[2],abs(bf[3]))-abs(bf[3])
                simp_values[j] = simpson((-1)*(samples[j]-1), xtest)*1000 #integrates each sample data
            else:
                samp_ew[j] = 0
                a_values[j] = 0
                mu_values[j] = 0
                sig_values[j] = 0
                base_values[j] = 0
                simp_values[j] = 0

        best_bf = np.array([a_values[np.where(a_values!=0)].mean(),mu_values[np.where(mu_values!=0)].mean(),sig_values[np.where(sig_values!=0)].mean(),base_values[np.where(base_values!=0)].mean()])
        fit_gauss = gauss_model(xtest,best_bf[0],best_bf[1],best_bf[2],best_bf[3])*(-1)+1
        #set values for line

        diff = (fit_gauss[only_line] - flat_wing[only_line])**2
        self.lines_gauss_Xsquare[i] = np.sum(diff)


        #self.lines_gauss_Xsquare[i] = chisquare(fit_gauss[only_line], flat_wing[only_line])[0]
        self.lines_bf_params[i] = best_bf
        if len(samp_ew[np.where(samp_ew==0)]) == len(samples):
            self.lines_ew[i] = 0
            self.lines_ew_err[i] = 0 
            self.lines_ew_simp[i] = 0
            self.lines_ew_simp_err[i] = 0
        else:
            self.lines_ew[i] = samp_ew[np.where(samp_ew!=0)].mean()
            self.lines_ew_err[i] = samp_ew[np.where(samp_ew!=0)].std()
            self.lines_ew_simp[i] = simp_values[np.where(simp_values!=0)].mean()
            self.lines_ew_simp_err[i] = simp_values[np.where(simp_values!=0)].std()
        print('line to measure:', ELEMENTS[self.lines_exd[i][0]],self.lines[i], '- Line found:', found_line)
        print('EW:',np.round(self.lines_ew[i],2),u"\u00B1",np.round(self.lines_ew_err[i],2), 'simps-int:', np.round(self.lines_ew_simp[i],2),u"\u00B1", np.round(self.lines_ew_simp_err[i],2))

        #Plotting stuff
        if plot:
            fig = plt.figure(figsize=(12,5))
            fig.suptitle("Order: " + str(order) + " " + "(" + str(np.round(self.shifted_wavelength[order].min(),3)) + "-" + str(np.round(self.shifted_wavelength[order].max(),3)) + ")")
            fit_view = fig.add_subplot(121)
            fit_view.grid()
            fit_view.set_xlabel(r'$\rm Wavelength~(\AA)$', size = 14)
            fit_view.set_ylabel('Normalized Flux', size = 14)
            fit_view.errorbar(measure_x_array,measure_y_array + ex_params[0],
                 yerr=2*temp_err_array/temp_pred_array,capsize=0,fmt='.', color = 'k', label = 'cont', zorder = 2)
            fit_view.scatter(measure_x_array[points_within_norm],measure_y_array[points_within_norm] + ex_params[0], s = 10, c='#4daf4a', zorder = 3, alpha = 0.8)
            fit_view.fill_between(xtest,m_plot+2*np.sqrt(np.diag(C)),
                     m_plot-2*np.sqrt(np.diag(C)),color='#999999',alpha=0.5)
            fit_view.plot([self.lines[i],self.lines[i]],[norm,norm*0.95], '--', color = 'k', alpha = 0.75)
            fit_view.plot([found_line,found_line],[norm,norm*0.95], '-', color='k')
            fit_view.plot([line_bound[0],line_bound[0]],[norm*1.025,norm*0.95], '--', color = '#e41a1c', alpha = 0.5)
            fit_view.plot([line_bound[1],line_bound[1]],[norm*1.025,norm*0.95], '--', color = '#e41a1c', alpha = 0.5)
            fit_view.annotate(str(self.lines[i]), xy = [self.lines[i], norm*1.025])
            fit_view.plot(xtest, fit_gauss, '--', color = '#377eb8', lw= 2)
            fit_view.plot([xtest[0],xtest[-1]],[norm,norm], '--', color = '#4daf4a')
            
            data_view = fig.add_subplot(122)
            data_view.grid()
            data_view.set_xlabel(r'$\rm Wavelength~(\AA)$', size = 14)
            data_view.scatter(measure_x_array,measure_y_array+ ex_params[0], s = 5, c = 'k', zorder = 2)
            data_view.errorbar(measure_x_array,measure_y_array + ex_params[0],
                 yerr=2*temp_err_array/temp_pred_array,capsize=0,fmt='.', color = 'k', zorder = 3, alpha = 0.5)
            plt.tight_layout()


            # fig1, coarse_view = plt.subplots()
            # coarse_view.set_title("Order: " + str(order) + " " + "(" + str(np.round(self.shifted_wavelength[order].min(),3)) + "-" + str(np.round(self.shifted_wavelength[order].max(),3)) + ")")
            # coarse_view.grid()
            # coarse_view.set_xlabel(r'$\rm Wavelength~(\AA)$', size = 14)
            # coarse_view.set_ylabel('Normalized Flux', size = 14)
            #coarse_view.plot(xtest,m_plot, 'k--', alpha = 0.75)
            # coarse_view.errorbar(self.shifted_wavelength[order][wind],self.normalized_flux[order][wind] + ex_params[0],
            #      yerr=2*self.obs_err[order][wind]/self.pred_all[order][wind],capsize=0,fmt='.', color = 'k', label = 'cont', zorder = 2)
            # coarse_view.scatter(self.shifted_wavelength[order][wind][points_within_norm],self.normalized_flux[order][wind][points_within_norm] + ex_params[0], s = 10, c='#4daf4a', zorder = 3, alpha = 0.8)
            # coarse_view.fill_between(xtest,m_plot+2*np.sqrt(np.diag(C)),
            #          m_plot-2*np.sqrt(np.diag(C)),color='#999999',alpha=0.5)
            #coarse_view.plot(xtest,samples.T,alpha=0.1, color='#cccccc')
            # coarse_view.plot([self.lines[i],self.lines[i]],[norm,norm*0.95], '--', color = 'k', alpha = 0.75)
            # coarse_view.plot([found_line,found_line],[norm,norm*0.95], '-', color='k')
            # coarse_view.plot([line_bound[0],line_bound[0]],[norm*1.025,norm*0.95], '--', color = '#e41a1c', alpha = 0.5)
            # coarse_view.plot([line_bound[1],line_bound[1]],[norm*1.025,norm*0.95], '--', color = '#e41a1c', alpha = 0.5)
            # coarse_view.annotate(str(self.lines[i]), xy = [self.lines[i], norm*1.025])
            #coarse_view.plot(xtest,dy+norm, '--', color = '#e41a1c', lw = 2) #view gradient
            # if plot_gaussian:
            #     coarse_view.plot(xtest, fit_gauss, '--', color = '#377eb8', lw= 2)
            # coarse_view.plot([xtest[0],xtest[-1]],[norm,norm], '--', color = '#4daf4a')
            if save_plot:
                fig_title = ELEMENTS[self.lines_exd[i][0]] + '_' + str(self.lines[i]) + '_' + str(order) + '.pdf'
                plt.savefig('line_plots/'+fig_title)
            plt.show()

            print('#-----------------------#')
        
        #print extra parameter stuff
        if ex_params == [0,0,0,0]:
            pass
        else:
            self.lines_exp[i] = np.array(ex_params)
            print('extra params:',ex_params)

#------------------------------------------------------------------------------------#        
    def measure_all_ew(self, exclude_lines= [], plot_lines=[], ex_params = {}, window_size = 1.5, save_all = False):
        if save_all:
            make_plots_folder()

        for order in range(len(self.wavelength)):
            for i in range(len(self.lines)):
                if self.lines[i] in exclude_lines:
                    self.lines_ew[i] = 0.0
                    self.lines_ew_simp[i] = 0.0
                    self.lines_gauss_Xsquare[i] = np.nan
                    self.lines_bf_params[i] = None
                    self.lines_ew_err[i] = np.nan
                    self.lines_ew_simp_err[i] = np.nan
                    self.lines_exp[i] = [0,0,0,0]
                    #self.lines_check_flag[i] = False
                elif self.lines[i] >= self.shifted_wavelength[order][0] and self.lines[i] <= self.shifted_wavelength[order][-1]:
                    plot = False
                    exp = [0,0,0,0]
                    if self.lines[i] in plot_lines:
                        plot = True
                        if self.lines[i] in ex_params.keys():
                            exp = ex_params[self.lines[i]]   
                    if save_all:
                        self.measure_ew(i,order, plot, exp, True, window_size)
                    else:
                        self.measure_ew(i,order, plot, exp, False, window_size)
        #self.lines_bf_params = np.array(self.lines_bf_params)
        
    def measure_line_ew(self,line,ex_params=[0,0,0,0], save_line = False, save_plot = False, window_size = 1.5):
        if save_plot:
            make_plots_folder()
        i = np.where(self.lines == line)[0][0]
        found = False
        for order in range(len(self.wavelength)):
            if self.lines[i] >= self.shifted_wavelength[order][0] and self.lines[i] <= self.shifted_wavelength[order][-1]:
                if not found:
                    self.lines_ew[i] = 0.0
                    self.lines_ew_simp[i] = 0.0
                    self.lines_gauss_Xsquare[i] = np.nan
                    self.lines_bf_params[i] = None
                    self.lines_ew_err[i] = np.nan
                    self.lines_ew_simp_err[i] = np.nan
                    #self.lines_check_flag[i] = False
                    self.measure_ew(i,order, True, ex_params, save_plot, window_size)
                    found = True
                    if save_line:
                        with open('line_'+str(line)+'.txt','w') as f:
                            for k in range(len(self.shifted_wavelength[order])):
                                f.write("{0:10f}\t{1:10f}\n".format(self.shifted_wavelength[order][k],self.normalized_flux[order][k]))
                            print('order', order, 'saved!')

                    
    def check_for_flags(self):
        for i in range(len(self.lines)):
            self.lines_check_flag[i] = False
            #error measure check - above 10% is a problem
            if self.lines_ew_err[i]/self.lines_ew[i] >= .1:
                self.lines_check_flag[i] = True
                print(self.lines[i], 'has more than a 10% error', np.round(self.lines_ew_err[i],2))
            #shallow line check - below 2 mA is a problem
            if self.lines_ew[i] < 2.0:
                self.lines_check_flag[i] = True
                print(self.lines[i], 'might be too shallow', np.round(self.lines_ew[i],2))
            if self.lines_gauss_Xsquare[i] == np.nan:
                self.lines_check_flag[i] = True
                print(self.lines[i], 'no fit, line may not be found in spectrum')
            #X square check - if fit above threshold (problem)
            elif self.lines_gauss_Xsquare[i] > self.X_thresh:
                self.lines_check_flag[i] = True
                print(self.lines[i], 'might have a bad fit', self.lines_gauss_Xsquare[i])
                
                
    
#------------------------------------------------------------------------------------#    
    
    
    def wat_info(self, hdul):
        '''Gather starting wavelengths and spacing for each order

        Parameters
        ----------
        hdul : fits file as hdul object
        open with astropy.io, fits.open()

        Returns
        -------
        starting_wavs : numpy array of starting wavelengths for each order
        wavelength (in Ang)
        wave_spacing : numpy array of wavelength spacing for each order
        '''
        spectrum_information = ''
        #loop through header keys
        for key in hdul[0].header.keys():
            #find 'WAT' key which holds wavelength and order information
            if 'WAT' in key:
                #collect spectrum information
                spectrum_information = spectrum_information + hdul[0].header[key]

        #split up spectrum information to grab starting wavelength and spacing for each order
        min_wave = 3000
        starting_wavs = np.zeros(hdul[0].header['NAXIS2'])
        wave_spacing = np.zeros(hdul[0].header['NAXIS2'])
        count = 0
        for stuff in spectrum_information.split('spec'):
            #print(stuff)
            if ' = ' in stuff:
                for i,item in enumerate(stuff.split(' ')):
                    #print(i,len(item))
                    try:
                        floats = float(item)
                        spacing = 0.0
                        starting_wave = 0.0
                        if floats > min_wave:
                            starting_wave = floats
                            spacing = float(stuff.split(' ')[i+1])
                            break

                    except:
                        if len(item) >= 30:
                            print()
                        non_floats = item
                        pass
                starting_wavs[count] = starting_wave
                #print(count, starting_wave)
                wave_spacing[count] = spacing
                #print(spacing)
                count += 1
        return starting_wavs, wave_spacing

    def read_spec(self):
        '''Read a KECK HIRES spectrum

        Parameters
        ----------
        filename : string
        name of the fits file with the data

        Returns
        -------
        wavelength : np.ndarray (orders,points)
        wavelength (in Ang)
        flux : np.ndarray (orders,points)
        flux (in erg/s/cm**2)
        '''
        with fits.open(self.filename) as hdul:
            header = hdul[0].header
            num_orders = header['NAXIS2']
            num_points = header['NAXIS1']

            #make index array
            wavelength = np.zeros((num_orders, num_points))

            try:#get ccd info and gain
                self.gain = np.ones(len(wavelength))*header['CCDGN01']
            except:
                self.keck_chip_gains(header, len(wavelength))

            try:
                #get wavelength information for each order
                for i in range(num_orders):
                    #get key for starting wavelength and spacing for each order
                    if i+1 < 10:
                        start_wave_key = 'CRVL1_'+'0'+str(i+1)
                        spacing_wave_key = 'CDLT1_'+'0'+str(i+1)
                    else:
                        start_wave_key = 'CRVL1_'+str(i+1)
                        spacing_wave_key = 'CDLT1_'+str(i+1)

                    #fill wavelength array            
                    wavelength[i][0] = header[start_wave_key] 
                    for j in range(num_points-1):
                        j += 1
                        wavelength[i][j] = wavelength[i][j-1] + header[spacing_wave_key]
                print('CRVL stuff found')

            except: #faster and possibly more common
                ##get wavelength information for each order
                starting_waves, wave_spacing = self.wat_info(hdul)

                #fill wavelength array
                for i in range(num_orders):
                    wavelength[i][0] = starting_waves[i]
                    for j in range(num_points-1):
                        j += 1
                        wavelength[i][j] = wavelength[i][j-1] + wave_spacing[i]
                print('CRVL stuff not found')

            #get flux
            flux = hdul[0].data

        return wavelength, flux
    
    def check_spectra(self, norm=True, lines=False):
        orders = len(self.wavelength)
        
        if norm:
            for i in range(orders):
                plt.plot(self.shifted_wavelength[i],self.normalized_flux[i], linewidth = 0.5)
            if lines:
                for j in range(len(self.lines)):
                    plt.plot([self.lines[j],self.lines[j]],[0.95,1.0], 'k')
                    plt.annotate(str(self.lines[j]), xy=[self.lines[j],1.01])
            #plt.ylim([0.4,1.1])
            plt.show()
        else:
            for i in range(orders):
                fig = plt.figure(figsize=(8,4))
                ax = fig.add_subplot(111)
                ax.set_title('Order: '+str(i))
                ax.plot(self.wavelength[i],self.flux[i])
                #ax.set_xticks([])
                #ax.set_yticks([])
                #ax.set_ylabel(str(i+1))
                plt.tight_layout()
                plt.show()

    def save_object(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def keck_chip_gains(self, header, num_orders):
        #Determine chip color based on starting wavelength of chip
        #Determine gain based on chip color info from https://www2.keck.hawaii.edu/inst/hires/ccdgain.html
        spec_info = header['WAT2_001'].split()
        for i in range(len(spec_info)):
            try:
                num = float(spec_info[i])
                if num/100 > 1.0:
                    break
            except ValueError:
                pass

        starting_wavelength = num
        print("starting wavelength:", starting_wavelength)
        low_high = header['CCDGAIN']
        print("Chip Gain:", low_high)
        if starting_wavelength < 5000.0:
            if low_high == 'low':
                self.gain = np.ones(num_orders)*1.95
            if low_high == 'high':
                self.gain = np.ones(num_orders)*0.78
        elif starting_wavelength < 6500.0:
            if low_high == 'low':
                self.gain = np.ones(num_orders)*2.09
            if low_high == 'high':
                self.gain = np.ones(num_orders)*0.84
        else:
            if low_high == 'low':
                self.gain = np.ones(num_orders)*2.09
            if low_high == 'high':
                self.gain = np.ones(num_orders)*0.89
                
class Continuum_scan():
    '''Selects points at the continuum
    '''
    def __init__(self, distx, depth):
        #values currently viewed for selection
        self.select_window = None
        #standard deviation of selected window
        #self.current_sig = None
        #Size of selection box in x axis
        self.distx = distx
        self.points_in_window = None
        #Input spectra
        self.data = None
        #Points selected as part of the continuum
        self.select_points = None
        #Relates to how deeply to move selection box into data
        self.depth = depth
        return None
        
    def load_data(self,x,y):
        self.data = np.array([x,y])
        self.select_points = np.zeros(len(x))
        self.points_in_window = len(self.data[0][np.where(self.data[0] <= self.data[0][0]+self.distx)])
        return None
    
    def scan(self):
        split_order_into = int(np.ceil(len(self.data[0])/self.points_in_window))
        split_order_x = np.array_split(self.data[0], split_order_into)
        split_order_y = np.array_split(self.data[1], split_order_into)
        for i in range(len(split_order_y)):
            dex = np.where((self.data[0] >= split_order_x[i][0])&(self.data[0] <= split_order_x[i][-1]))
            percent = np.percentile(split_order_y[i], self.depth)
            self.select_points[dex] = (split_order_y[i] >= percent)
        return None



        # left_lim = self.data[0][0]
        # while left_lim < self.data[0][-1]:
        #     right_lim = left_lim + self.distx
        #     self.select_window = np.where((self.data[0]>=left_lim)&(self.data[0]<=right_lim))
        #     #self.current_sig = np.sqrt(self.data[1][self.select_window]).mean()
        #     #self.current_sig = self.data[1][self.select_window].std()
        #     self.above_sigma()
        #     left_lim = right_lim
        #     #self.view_window()
        # return None
            
#     def view_window(self):
#         percent = np.percentile(self.data[1][self.select_window], self.depth)
#         fig = plt.figure()
#         ax = fig.add_subplot(121)
#         ax.scatter(self.data[0][self.select_window], self.data[1][self.select_window], c = '#cccccc', alpha = 0.75)
#         bool_points = (self.select_points == 1)
#         ax.scatter(self.data[0][bool_points],self.data[1][bool_points], c = 'g')
#         ax.plot([self.data[0][self.select_window][0], self.data[0][self.select_window][-1]],
#                 [self.data[1][self.select_window].max(),self.data[1][self.select_window].max()], 'k--')
# #         ax.plot([self.data[0][self.select_window][0], self.data[0][self.select_window][-1]],
# #                 [self.data[1][self.select_window].max() - self.depth*self.current_sig,
# #                  self.data[1][self.select_window].max() - self.depth*self.current_sig],'g--')
#         ax.set_xlim([self.data[0][self.select_window][0], self.data[0][self.select_window][-1]])
#         plt.axhline(percent, color='k', linestyle='dashed', linewidth=1)
#         hist = fig.add_subplot(122)
#         hist.hist(self.data[1][self.select_window])
#         plt.axvline(percent, color='k', linestyle='dashed', linewidth=1)
#         plt.show()
#         return None
        
    def view_selected(self):
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(111)
        ax.scatter(self.data[0], self.data[1], c = '#cccccc', alpha = 0.75, s = 5)
        bool_points = (self.select_points == 1)
        ax.scatter(self.data[0][bool_points],self.data[1][bool_points], c = 'g', s = 5)
        plt.show()
        return None
        
    def get_selected(self):
        bool_points = (self.select_points == 1)
        return bool_points
#-------------------------------------------------------------DICTIONARY--------------------------------------------------------------#
ELEMENTS = {1:'H I',2:'He I',3:'Li I',4:'Be I',5:'B I',6:'C I',7:'N I',8:'O I',9:'F I',10:'Ne I',11:'Na I',12:'Mg I',13:'Al I',
14:'Si I',15:'P I',16:'S I',17:'Cl I',18:'Ar I',19:'K I',20:'Ca I',21:'Sc I',21.1:'Sc II',22:'Ti I',22.1:'Ti II',23:'V I',24:'Cr I',
25:'Mn I',26:'Fe I',26.1:'Fe II',27:'Co I',28:'Ni I',29:'Cu I',30:'Zn I', 56.1:'Ba II', 39.1:'Y II'}

print('Working')
