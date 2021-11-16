#Python 3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from scipy.integrate import simps
from numpy.linalg import inv
from numpy.linalg import slogdet
import pickle
import glob
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
            plt.plot(wave[window],flux[window])
            plt.plot([line,line],[0.95,1.0], 'k')
            plt.annotate(str(line), xy=[line,1.01])
            plt.plot([best_line_guess,best_line_guess],[0.95,1.0], 'k--')
            plt.annotate(str(best_line_guess), xy=[best_line_guess,1.01])
            plt.show()
        else:
            boundaries[0] = wave[window][left_look][dy1_left][-1]
    else:
        boundaries[0] = left_bound
    if auto_bound_r:
        right_look = np.where(wave[window] >= best_line_guess + 0.05)
        dy1_right = np.where((dy[right_look] < dy_r)&(dy[right_look] > (-1)*dy_r))
        if len(wave[window][right_look][dy1_right]) ==0:
            print('line ',line,' very close to edge or dy selection value too small')
            plt.plot(wave[window],flux[window])
            plt.plot([line,line],[0.95,1.0], 'k')
            plt.annotate(str(line), xy=[line,1.01])
            plt.plot([best_line_guess,best_line_guess],[0.95,1.0], 'k--')
            plt.annotate(str(best_line_guess), xy=[best_line_guess,1.01])
            plt.show()
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
    



#-------------------------------------------------------------DICTIONARY--------------------------------------------------------------#
ELEMENTS = {1:'H I',2:'He I',3:'Li I',4:'Be I',5:'B I',6:'C I',7:'N I',8:'O I',9:'F I',10:'Ne I',11:'Na I',12:'Mg I',13:'Al I',
14:'Si I',15:'P I',16:'S I',17:'Cl I',18:'Ar I',19:'K I',20:'Ca I',21:'Sc I',21.1:'Sc II',22:'Ti I',22.1:'Ti II',23:'V I',24:'Cr I',
25:'Mn I',26:'Fe I',26.1:'Fe II',27:'Co I',28:'Ni I',29:'Cu I',30:'Zn I'}
