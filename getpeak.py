# getpeak.py
###########################################
#   Y.S.Murakami, 2019
#   sterling.astro@berkeley.edu
#
# - Designed for gamma ray spectroscopy.
#
# - for given x and y data (sorted in x), this 
#   pipeline can find the peak in y value and
#   returns the gaussian fitting parameters.
#
# - sampling parameter is the range of index
#   that are sampled for smoothing.
#   use smaller sample number for narrower peak
#   and larger sample number for wider peak.
#
# - note that N data points at the edge of the data
#   are completely flattened for sampling=N
###########################################
# example usage:
#
#  fname = 'data/Co_5min_coarse=32_fine=2.5_g0=3.5_550V.dat'
#  do_analysis(fname,sampling=30,N_peak_search=4)
#
# See example.py for more detailed example.
###########################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import cm


def get_data(fname):
    time = np.loadtxt(fname,skiprows=1,max_rows=2,dtype='str')[1]
    xdata,ydata,_,_ = np.loadtxt(fname,skiprows=3,unpack=True)
    return time,xdata,ydata


def get_smooth(ydata,sampling=3):
    y_smooth=np.linspace(0,1,len(ydata)) # dummy
    for i in range(len(ydata)):
        if i <= int(sampling/2):
            y_smooth[i] = np.average(ydata[0:sampling])
        elif i >= len(ydata)-int(sampling/2):
            y_smooth[i] = np.average(ydata[-1*sampling:-1])
        else:
            y_smooth[i] = np.average(ydata[i-int(sampling/2):i+int(sampling/2)])
    return y_smooth


def get_peak(xdata,ydata,sampling=30,N_peak_search=2,max_search_width=100):
    y_smooth = get_smooth(ydata,sampling)
    y_dot_smooth = get_smooth(np.gradient(y_smooth,xdata),sampling)
    y_dot2_smooth = get_smooth(np.gradient(y_dot_smooth,xdata),sampling)

    dot2_tmp = np.array(y_dot2_smooth)
    boundaries = []
    for n in range(N_peak_search):
        nth_peak_idx = np.argmin(dot2_tmp)
        L_bound_idx = 0
        R_bound_idx = 0
        L_min = abs(dot2_tmp[nth_peak_idx])+1
        R_min = abs(dot2_tmp[nth_peak_idx])+1
        for w in range(max_search_width):
            if L_min > abs(dot2_tmp[nth_peak_idx - w]):
                L_min = abs(dot2_tmp[nth_peak_idx - w])
                #print(L_min)
                dot2_tmp[nth_peak_idx - w] = 0 if not w==0 else dot2_tmp[nth_peak_idx - w]
            else:
                L_bound_idx = nth_peak_idx - w
                break
        for w in range(max_search_width):
            if R_min > abs(dot2_tmp[nth_peak_idx + w]):
                R_min = abs(dot2_tmp[nth_peak_idx + w])
                dot2_tmp[nth_peak_idx + w] = 0
            else:
                R_bound_idx = nth_peak_idx + w
                break
        boundaries.append([L_bound_idx,nth_peak_idx,R_bound_idx])
    return boundaries # index


def plot_peak_analysis(xdata,ydata,sampling=30,N_peak_search=2,result_only=False):
    boundaries = get_peak(xdata,ydata,sampling,N_peak_search)

    if result_only:
        plt.figure(figsize=(12,5))
        plt.plot(xdata,ydata)
        for boundary in boundaries:
            L_bound_idx,nth_peak_idx,R_bound_idx = boundary
            plt.axvspan(xdata[L_bound_idx],xdata[R_bound_idx],alpha=0.5,color='orange')
        plt.grid()
        
    else:
        y_smooth = get_smooth(ydata,sampling=sampling)
        y_dot_smooth = get_smooth(np.gradient(y_smooth,xdata),sampling)
        y_dot2_smooth = get_smooth(np.gradient(y_dot_smooth,xdata),sampling)

        fig = plt.figure(figsize=(12,15))
        ax1 = plt.subplot(3,1,1)
        ax1.plot(xdata,ydata,label='raw data')
        ax1.plot(xdata,y_smooth,label='smoothed data')
        ax1.legend()
        ax1.grid()
        ax1.set_title('Raw + Smoothed Data')

        ax2 = plt.subplot(3,1,2)

        for boundary in boundaries:
            L_bound_idx,nth_peak_idx,R_bound_idx = boundary
            ax2.axvline(xdata[nth_peak_idx])
            ax2.axvspan(xdata[L_bound_idx],xdata[R_bound_idx],alpha=0.3)
        ax2.plot(xdata,y_dot2_smooth)
        ax2.grid()
        ax2.set_title('second derivative')

        ax3 = plt.subplot(3,1,3)
        for boundary in boundaries:
            L_bound_idx,nth_peak_idx,R_bound_idx = boundary
            ax3.axvspan(xdata[L_bound_idx],xdata[R_bound_idx],alpha=0.5,color='orange')
        ax3.grid()
        ax3.plot(xdata,ydata,label='Raw Data')
        ax3.set_title('Raw Data + Detected Peak')


def gaussian(x,sigma,mu,N):
    return (N/(np.sqrt(2*np.pi)*sigma)) * np.exp((-1/2)*((x-mu)/sigma)**2)


def fitting(xdata,ydata,boundaries):
    fitted_curves = []
    fitted_params = []
    for boundary in boundaries:
        L_idx, peak_idx, R_idx = boundary
        x_peak = xdata[L_idx:R_idx]
        y_peak = ydata[L_idx:R_idx]
        p0 = [(xdata[R_idx]-xdata[peak_idx]),xdata[peak_idx],ydata[peak_idx]]
        popt,pcov = curve_fit(gaussian,x_peak,y_peak,p0=p0)
        x_th = np.linspace(np.min(x_peak),np.max(x_peak),100)
        y_th = gaussian(x_th,*popt)
        fitted_curves.append([x_th,y_th])
        fitted_params.append(popt)
    return fitted_curves, fitted_params


def do_analysis(fname,sampling=10,N_peak_search=2,show_plot=True):
    time,xdata,ydata = get_data(fname) # modify get_data for differently formatted data
    xdata      = np.array(xdata)
    ydata      = np.array(ydata) / float(time) # total photon count -> flux
    boundaries = get_peak(xdata,ydata,sampling,N_peak_search)
    fitted_curves, fitted_params = fitting(xdata,ydata,boundaries)
    
    if show_plot:
        plt.figure(figsize=(12,3))
        plt.plot(xdata,ydata,alpha=0.5)
        colors = cm.get_cmap('winter')(np.linspace(0,1,N_peak_search))
        for i in range(N_peak_search):
            L_idx, nth_peak_idx, R_idx = boundaries[i]
            x_fit, y_fit               = fitted_curves[i]
            eqn = "σ={:.2f}, μ={:.2f}, N={:.2f}".format(*fitted_params[i])
            plt.axvspan(xdata[L_idx],xdata[R_idx],alpha=0.1,color=colors[i])
            plt.plot(x_fit,y_fit,label=eqn,color=colors[i],linewidth=2)
        plt.grid()
        plt.legend()
        plt.title(fname)
    return fitted_params
