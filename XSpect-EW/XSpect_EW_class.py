#Python 3
import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import fmin
from scipy.stats import chisquare
from scipy.interpolate import interp1d
from numpy.random import multivariate_normal
from astropy.io import fits
from scipy.optimize import curve_fit
import glob
#from scipy.stats import pearsonr as ptest
from XSpect_EW_func import *
import george
from george import kernels
from scipy.optimize import minimize

#--------------------------------------------------Classes--------------------------------------------------#
class Spectrum_Data():
    def __init__(self, filename):
        self.filename = filename
        self.wavelength, self.flux = self.read_spec()
        self.normalized_flux = self.flux.copy()
        self.shifted_wavelength = self.wavelength.copy()
        self.estimated_shift = np.zeros(len(self.wavelength))
        self.rv = None #km/s
        
        #continuum information
        self.continuum = np.full((len(self.wavelength),len(self.wavelength[0])), False)
        self.pred_all = np.zeros((len(self.wavelength),len(self.wavelength[0])))
        self.pred_var_all = np.zeros((len(self.wavelength),len(self.wavelength[0])))
        self.obs_err = np.zeros((len(self.wavelength),len(self.wavelength[0])))
        
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
        #line - best fit parameters for gaussian fit
        self.lines_bf_params = None
        #line - X squared value for gaussian and data
        self.lines_gauss_Xsquare = None
        #line - X squared threshold value
        self.X_thresh = 0.001
        #line - X squared value above threshold or EW = 0
        self.lines_check_flag = None
        
    def normalize_all(self, window_width = 1.5, continuum_depth = 90):
        #loop through orders        
        for i in range(len(self.flux)):

            #use Gaussian Process to fit continuum
            self.normalize(i, window_width, continuum_depth)
        return None

    def normalize(self, order, window_width = 1.5, continuum_depth = 90, clip = [-999,-999]):
        if clip[0] != -999 and clip[1] != -999:
            #clipped = True
            clipl = np.where(self.wavelength[order] <= clip[0])[0][-1]
            clipr = np.where(self.wavelength[order] >= clip[1])[0][0]
        else:
            #clipped = False
            clipl = 0
            clipr = -1
 
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

    def S_N(self, order):
        return self.flux[order]/self.obs_err[order]

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
                            chi[k] = chisquare(result_y, sun.normalized_flux[i][compare_window_sun])[0]             
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
        self.lines_bf_params = np.array([None]*len(self.lines))
        self.lines_gauss_Xsquare = np.array([np.nan]*len(self.lines))
        self.lines_check_flag = np.array([False]*len(self.lines))
        for i in range(len(self.lines)):
            self.lines_exd[i] = np.array([elmnt[i],ep[i],gf[i],rad[i]])
        
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
        
    def measure_ew(self, i, order, plot = False, ex_params = [0,0,0,0], save_plot = False):
        #extra parameters [0] - shift continuum
        #                 [1] - left boundary in Angstroms
        #                 [2] - right boundary in Angstroms
        #                 [3] - line center in Angstroms
        norm = 1.0
        wind, found_line, line_bound,dy = get_line_window(self.lines[i],self.shifted_wavelength[order],self.normalized_flux[order],ex_params[1],ex_params[2],ex_params[3])
        other_than_line = np.where((self.shifted_wavelength[order][wind] <= line_bound[0])|(self.shifted_wavelength[order][wind] >= line_bound[1]))
        only_line = np.where((self.shifted_wavelength[order][wind] >= line_bound[0])|(self.shifted_wavelength[order][wind] <= line_bound[1]))
        flat_wing = self.normalized_flux[order][wind].copy() + ex_params[0]
        flat_wing[other_than_line] = norm 
        xtest = np.linspace(self.shifted_wavelength[order][wind][0], self.shifted_wavelength[order][wind][-1], len(self.wavelength[order][wind]))
        m,C=Pred_GP(SEKernel,[1,100],self.shifted_wavelength[order][wind],flat_wing,2*self.obs_err[order][wind]/self.pred_all[order][wind], xtest)
        samples = multivariate_normal(m,C,500)
        m_plot=m.copy()
        m = (-1)*(m-1)
        samp_ew = np.zeros(len(samples))
        a_values = np.zeros(len(samples))
        mu_values = np.zeros(len(samples))
        sig_values = np.zeros(len(samples))
        base_values = np.zeros(len(samples))
        plot_gaussian = False
        for j in range(len(samples)):
            #plt.plot(xtest,(-1)*(samples[j]-1), 'b--')
            bf, err, p0 = gfit_simple(xtest, (-1)*(samples[j]-1), found_line, 0.5,0)
            #print('best fit:', bf, err, p0)
            if bf[0] > 0.0:
                if abs(gauss_ew(bf[0], bf[2]*2.355)) > 2 and abs(gauss_ew(bf[0], bf[2]*2.355)) < 175:
                    samp_ew[j] = abs(gauss_ew(bf[0], bf[2]*2.355))
                    a_values[j] = bf[0]
                    mu_values[j] = bf[1]
                    sig_values[j] = abs(bf[2])
                    base_values[j] = bf[3]
        if plot:
            plot_gaussian = True
        best_bf = np.array([a_values[np.where(a_values!=0)].mean(),mu_values[np.where(mu_values!=0)].mean(),sig_values[np.where(sig_values!=0)].mean(),base_values[np.where(base_values!=0)].mean()])
        fit_gauss = gauss_model(xtest,best_bf[0],best_bf[1],best_bf[2],best_bf[3])*(-1)+1
        #set values for line
        self.lines_gauss_Xsquare[i] = chisquare(fit_gauss[only_line], flat_wing[only_line])[0]
        self.lines_bf_params[i] = best_bf
        self.lines_ew[i] = samp_ew[np.where(samp_ew!=0)].mean()
        self.lines_ew_err[i] = samp_ew[np.where(samp_ew!=0)].std()
        #Plotting stuff
        if plot:
            fig1, coarse_view = plt.subplots()
            coarse_view.set_title(str(order))
            coarse_view.grid()
            coarse_view.set_xlabel(r'$\rm Wavelength~(\AA)$', size = 14)
            coarse_view.set_ylabel('Normalized Flux', size = 14)
            coarse_view.plot(xtest,m_plot, 'k--', alpha = 0.75)
            coarse_view.errorbar(self.shifted_wavelength[order][wind],self.normalized_flux[order][wind] + ex_params[0],
                 yerr=2*self.obs_err[order][wind]/self.pred_all[order][wind],capsize=0,fmt='k.', label = 'cont')
            coarse_view.fill_between(xtest,m_plot+2*np.sqrt(np.diag(C)),
                     m_plot-2*np.sqrt(np.diag(C)),color='k',alpha=0.2)
            coarse_view.plot(xtest,samples.T,alpha=0.1, color='#cccccc')
            coarse_view.plot([self.lines[i],self.lines[i]],[norm,norm*0.95], 'k--', alpha = 0.75)
            coarse_view.plot([found_line,found_line],[norm,norm*0.95], 'k-')
            coarse_view.plot([line_bound[0],line_bound[0]],[norm*1.025,norm*0.95], 'r--', alpha = 0.5)
            coarse_view.plot([line_bound[1],line_bound[1]],[norm*1.025,norm*0.95], 'r--', alpha = 0.5)
            coarse_view.annotate(str(self.lines[i]), xy = [self.lines[i], norm*1.025])
            coarse_view.plot(xtest,dy+norm, 'r--', lw = 2)
            if plot_gaussian:
                coarse_view.plot(xtest, fit_gauss, 'b--')
            coarse_view.plot([xtest[0],xtest[-1]],[norm,norm], 'g--')
            if save_plot:
                fig_title = str(order) + '_' + str(self.lines[i]) + '.pdf'
                plt.savefig(fig_title)
            plt.show()
            print(np.round(self.lines_ew[i],2),np.round(self.lines_ew_err[i],2))
        
        #print extra parameter stuff
        if ex_params == [0,0,0,0]:
            pass
        else:
            self.lines_exp[i] = np.array(ex_params)
            print('extra params:',ex_params)

#------------------------------------------------------------------------------------#        
    def measure_all_ew(self, exclude_lines= [], plot_lines=[], ex_params = {}):
        for order in range(len(self.wavelength)):
            for i in range(len(self.lines)):
                if self.lines[i] in exclude_lines:
                    self.lines_ew[i] = 0.0
                    self.lines_gauss_Xsquare[i] = np.nan
                    self.lines_bf_params[i] = None
                    self.lines_ew_err[i] = np.nan
                    self.lines_exp[i] = [0,0,0,0]
                    #self.lines_check_flag[i] = False
                elif self.lines[i] >= self.shifted_wavelength[order][0] and self.lines[i] <= self.shifted_wavelength[order][-1]:
                    plot = False
                    exp = [0,0,0,0]
                    if self.lines[i] in plot_lines:
                        plot = True
                        if self.lines[i] in ex_params.keys():
                            exp = ex_params[self.lines[i]]    
                    self.measure_ew(i,order, plot, exp)
        #self.lines_bf_params = np.array(self.lines_bf_params)
        
    def measure_line_ew(self,line,ex_params=[0,0,0,0], save_line = False, save_plot = False):
        i = np.where(self.lines == line)[0][0]
        found = False
        for order in range(len(self.wavelength)):
            if self.lines[i] >= self.shifted_wavelength[order][0] and self.lines[i] <= self.shifted_wavelength[order][-1]:
                if not found:
                    self.lines_ew[i] = 0.0
                    self.lines_gauss_Xsquare[i] = np.nan
                    self.lines_bf_params[i] = None
                    self.lines_ew_err[i] = np.nan
                    #self.lines_check_flag[i] = False
                    self.measure_ew(i,order, True, ex_params, save_plot)
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
                        non_floats = item
                        pass
                starting_wavs[count] = starting_wave
                wave_spacing[count] = spacing
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
            plt.ylim([0.4,1.1])
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
        #Input spectra
        self.data = None
        #Points selected as part of the continuum
        self.select_points = None
        #Relates to how deeply to move selection box into data
        self.depth = depth
        return None
    
    def above_sigma(self): #
        #within window, select points above (max value - depth*sigma)
        current_data_x = self.data[0][self.select_window]
        current_data_y = self.data[1][self.select_window]
        #top_value = current_data_y.max()
        #y_lim = top_value - self.depth*self.current_sig
        percent = np.percentile(current_data_y, self.depth)
        #self.select_points[self.select_window] = (current_data_y >=  y_lim)
        self.select_points[self.select_window] = (current_data_y >= percent)
        return None
        
    def load_data(self,x,y):
        self.data = np.array([x,y])
        self.select_points = np.zeros(len(x))
        return None
    
    def scan(self):
        left_lim = self.data[0][0]
        while left_lim < self.data[0][-1]:
            right_lim = left_lim + self.distx
            self.select_window = np.where((self.data[0]>=left_lim)&(self.data[0]<=right_lim))
            #self.current_sig = np.sqrt(self.data[1][self.select_window]).mean()
            #self.current_sig = self.data[1][self.select_window].std()
            self.above_sigma()
            left_lim = right_lim
            #self.view_window()
        return None
            
    def view_window(self):
        percent = np.percentile(self.data[1][self.select_window], self.depth)
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.scatter(self.data[0][self.select_window], self.data[1][self.select_window], c = '#cccccc', alpha = 0.75)
        bool_points = (self.select_points == 1)
        ax.scatter(self.data[0][bool_points],self.data[1][bool_points], c = 'g')
        ax.plot([self.data[0][self.select_window][0], self.data[0][self.select_window][-1]],
                [self.data[1][self.select_window].max(),self.data[1][self.select_window].max()], 'k--')
#         ax.plot([self.data[0][self.select_window][0], self.data[0][self.select_window][-1]],
#                 [self.data[1][self.select_window].max() - self.depth*self.current_sig,
#                  self.data[1][self.select_window].max() - self.depth*self.current_sig],'g--')
        ax.set_xlim([self.data[0][self.select_window][0], self.data[0][self.select_window][-1]])
        plt.axhline(percent, color='k', linestyle='dashed', linewidth=1)
        hist = fig.add_subplot(122)
        hist.hist(self.data[1][self.select_window])
        plt.axvline(percent, color='k', linestyle='dashed', linewidth=1)
        plt.show()
        return None
        
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

print('Working')