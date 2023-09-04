#!/home/fsarandrea/anaconda3/envs/igwn-py39/bin/python
"""
This script generates a webpage to display the results of a BayesWave run.

This script was originally put together by Jonah Kanner and Francecso Pannarale.
Major and minor modifications have been made by Meg Millhouse, Sudarshan Ghonge, Sophie Hourihane, and probably others

This code is tested with python2.7 and python 3.9
"""

######################################################################################################################
#
# Import modules
#
######################################################################################################################
import argparse
import glob
import math
import matplotlib
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as scs
import numpy as np
import os
import pwd
import subprocess
import sys
import getopt
import re
import traceback
from bayeswave_pipe import bayeswave_pipe_utils as pipe_utils
import lalsimulation as lalsim
from bayeswave_plot import BW_Flags



print('Path to megaplot: ')
print(sys.argv[0])
print('\n')


# Allow navigation into specified working directory

argParser = argparse.ArgumentParser()
argParser.add_argument("-o", "--outputDir", help="output directory")
argParser.add_argument("-s", "--start_time", help="start time of data")
argParser.add_argument("-t", "--event_time", help="time coordinate of glitch")
args = argParser.parse_args()

workdir=args.outputDir
start_time=args.start_time
event_time=args.event_time

os.chdir(workdir)
RUN_FLAGS = BW_Flags.Flags(os.getcwd())
BW_Flags.readbwb(RUN_FLAGS)

print('Workdir: ')
print(workdir)
print('\n')

import pkg_resources
postprocesspath = pkg_resources.resource_filename('bayeswave_plot',
    '../bayeswave_plot_data/')

from scipy import integrate

######################################################################################################################
#
# Dictionaries and other global variables
#
######################################################################################################################
# Item labels must coincide with moment names given in BayesWavePost
# 1st entry = x-axis label
# 2nd entry = file name tag
moments_dict = {}
moments_dict['dur_rec']               = ['Duration', 'duration']
moments_dict['t_energy_rec']          = ['log10(Signal Energy)', 'energy']
moments_dict['f0_rec']                = ['Central Freq (Hz)', 'freq']
moments_dict['band_rec']              = ['Bandwidth (Hz)', 'band']
moments_dict['t0_rec']                = ['Central Time (s)', 'tzero']
moments_dict['overlap']               = ['Overlap - recovered signal and injection', 'overlap']
moments_dict['network_overlap']       = ['Network overlap - recovered signal and injection', 'network_overlap']
moments_dict['snr']                   = ['Recovered SNR','snr']

# 1st entry = html file name
# 2nd entry = subpage header
# 3rd entry = plot file name tag
html_dict = {}
html_dict['dur_rec']           = ['duration', 'Histogram of duration', moments_dict['dur_rec'][1]]
html_dict['t_energy_rec']      = ['energy', 'Histogram of signal energy, (aka ||<i>h(t)</i>||<sup>2</sup>)', moments_dict['t_energy_rec'][1]]
html_dict['f0_rec']            = ['f0', 'Histogram of central frequency', moments_dict['f0_rec'][1]]
html_dict['band_rec']          = ['band', 'Histogram of bandwidth', moments_dict['band_rec'][1]]
html_dict['t0_rec']            = ['t0', 'Histogram of central time', moments_dict['t0_rec'][1]]
html_dict['overlap']           = ['overlap', 'Overlap Histogram between recovered signal and injection', moments_dict['overlap'][1]]
html_dict['signal_waveform']   = ['', 'Median Signal Model Waveforms and 1-sigma Uncertainties', 'signal_waveform']
html_dict['glitch_waveform']   = ['', 'Median Glitch Model Waveforms and 1-sigma Uncertainties', 'glitch_waveform']
html_dict['spec']              = ['spec', 'Spectrogram of median reconstructed model waveform', 'self_spec']
html_dict['diagnostics']       = ['diagnostics', 'Diagnostic plots']
html_dict['snr']               = ['snr','SNR','snr']
html_dict['distance']               = ['distance','Distance to cbc [MPC]','distance']
html_dict['Q']               = ['Q','Mass Ratio','Q']
html_dict['Mc']               = ['Mc','Chirp Mass','Mc']
html_dict['mass1']               = ['mass1','Larger binary component mass','mass1']
html_dict['mass2']               = ['mass2','Smaller binary component mass','mass2']
html_dict['chi_eff']               = ['chi_eff','Effective spin parameter','chi_eff']
html_dict['Z']               = ['Z','Redshift','Z']


# 1st entry = index of parameter
# 2nd entry html file name
# 3rd entry plot formmated 
# 4th entry =  subpage header
cbc_dict = {}

### Source frame 
cbc_dict['mass1_s'] = [0, 'mass1_s', r'$M_{1 \, \mathrm{source}}$',  'Mass1 source frame']
cbc_dict['mass2_s'] = [1, 'mass2_s', r'$M_{2 \, \mathrm{source}}$' , 'Mass 2 source frame']
cbc_dict['mchirp_s'] = [2, 'mchirp_s', r'$M_{C \, \mathrm{source}}$', 'Chirp mass source frame']
cbc_dict['mtot_s'] = [3, 'mtot_s', r'$M_{\mathrm{tot source}}$', 'Total mass source frame']

# spin parameters 
cbc_dict['spin1'] = [4, 'spin1', r'$s_1$', 'Spin 1']
cbc_dict['spin2'] = [5, 'spin2', r'$s_2$', 'Spin 2']
cbc_dict['chi_eff'] = [6, 'chi_eff', '$\chi_{\mathrm{eff}}$', 'Effective spin parameter']

cbc_dict['coa_phase'] = [7, 'coa_phase', r'$\phi$', 'Phase at coalescence']
# coalesence time measured from trigger time 
cbc_dict['t_segment'] = [8, 'coa_t', 't [s]', 'Time since trigger'] 
# coalesence time measured in gps time
cbc_dict['coa_t'] = [8, 'coa_t', 't [s]', 'GPS time of event'] 

cbc_dict['distance'] = [9, 'distance', 'D [mpc]', 'Distance'] 
cbc_dict['Z'] = [10, 'Z', 'Z', 'Redshift']

cbc_dict['ra'] = [11, 'ra', 'RA', 'Right Ascension']  # right ascension (longitude)
cbc_dict['sin_dec'] = [12, 'sin_dec', r'$\mathrm{sin}(\mathrm{dec})$', 'Sin Declination'] # sin declenation (sin latitude)
cbc_dict['psi']      = [13, 'psi', r'$\Psi$', 'Polarization'] # polaization of signal
cbc_dict['cos_iota'] = [14, 'cos_iota', r'$\mathrm{cos}(\iota)$', 'Cos angle of inclination'] # angle between obs and orbital plane of cbc 

### Detector frame 
cbc_dict['mass1'] = [15, 'mass1_s', r'$M_{1}$',  'Mass1 Detector frame']
cbc_dict['mass2'] = [16, 'mass2_s', r'$M_{2}$' , 'Mass 2 Detector frame']
cbc_dict['mchirp'] = [17, 'mchirp_s', r'$M_{C}$', 'Chirp mass Detector frame']
cbc_dict['mtot'] = [18, 'mtot_s', r'$M_{\mathrm{Tot}}$', 'Total mass Detector frame']

### Tidal parameters 
cbc_dict['lambda1'] = [19, 'lambda1', r'$\lambda_1$', 'Lambda 1']
cbc_dict['lambda2'] = [20, 'lambda2', r'$\lambda_2$', 'Lambda 2']
cbc_dict['lambda3'] = [21, 'lambda3', r'$\lambda_3$', 'Lambda 3']
cbc_dict['lambda4'] = [22, 'lambda4', r'$\lambda_4$', 'Lambda 4']

modelList = ('signal', 'glitch', 'cbc')

postDir   = 'post/'
plotsDir  = 'plots/'
htmlDir   = 'html/'

CMAP_NAME = 'viridis'


# I think RUN_FLAGS gets set twice, I dont think this codeblock is necesssary
if os.path.exists(postDir+'full'):
    RUN_FLAGS.fullOnly_flag = 1
    RUN_FLAGS.multi_type = True
    print("Found --fullOnly flag\n")
if os.path.exists(postDir+'cbcglitch'):
    RUN_FLAGS.GlitchCBC_flag = 1
    print("Found --GlitchCBC flag\n")
if os.path.exists(postDir+'cbcsignal'):
    RUN_FLAGS.SignalCBC_flag = 1
    print("Found --SignalCBC flag\n")

#Adopt common color scheme for different models
ncolor = 'darkgrey'
gcolor = 'darkgoldenrod'
scolor = 'darkorchid'
cbccolor = '#d81159' # magenta #'#45d9a5' # teal  #'#ffa756' # pale orange 
glitchcbccolor = '#ad3400' # fox brown 
signalcbccolor = '#dfa9fc' # lilac
injcolor = 'teal'

def set_color(model):
    if model == 'glitch':
        return(gcolor)
    elif model == 'signal':
        return(scolor)
    elif model == 'cbc':
        return(cbccolor)
    elif model == 'full':
        return(scolor)
    elif model == 'cbcglitch':
        return(gcolor) #glitchcbccolor)
    elif model == 'cbcsignal':
        return(signalcbccolor)
    elif model == 'clean':
        return('mediumseagreen')
    elif model == 'injection' or model == 'injected':
        return(injcolor)
    return(ncolor)


ifoColors = ['darkgoldenrod','darkkhaki','darkseagreen','olive','cadetblue','green','slategray','darkcyan']
signal_ifoColorList = ['darkorchid','fuchsia','indigo','orchid','slateblue','mediumvioletred','palevioletred']


######################################################################################################################
#
# Read in run data and info
#
######################################################################################################################
# ---------------------------------------------
# Get Median waveform with CIs
# ---------------------------------------------
def get_waveform(filename):
    names = ['samples','median_waveform','50low','50high','90low','90high']
    data = np.recfromtxt(filename,names=names)
    return (data['samples'],data['median_waveform'],data['50low'],data['50high'],data['90low'],data['90high'])

def get_wavelet_params(filename, model, chirpflag=False, O1version=False, **keyword_parameters):
    """
    Read in chain file and get all wavelet params

    arguments
    ---------
    filename (str): the chain file

    model (str): signal or glitch

    optional, chirpflag: True if using chirplets
    optional, O1version: True if using O1 era chains
    optional, restrict (int): line number if you only want one draw from the chain

    outputs
    -------
    dictionary of the wavelet params
    """
    NW = 5 # number of intrinsic parameters (changes for chirplets)
    NE = 6 # number of extrinsic parameters
    start = 1

    labels = ['t','f','Q','logA','phi_int'] # parameters of individual wavelets
    extlabels = ['alpha','sindelta','psi','elip', 'phi_ext','scale'] # Common extrinsic parameters
    if chirpflag:
        NW = 6
        labels.append('beta')

    data = {}
    for l in labels:
        data[l] = []

    if model == 'signal': # get extrinsic parameters
        for l in extlabels:
            data[l] = []

    data['D'] = []

    infile = open(filename)
    lines = infile.readlines()

    if ('restrict' in keyword_parameters):
        restrict = int(keyword_parameters['restrict'])
        rn = [restrict]
    else:
        rn = np.arange(0,len(lines))


    for j in rn:
        line = lines[j]
        spl = line.split()
        waveletnumber = int(spl[0]) # how many wavelets
        data['D'].append(waveletnumber)
        if model == 'signal':
            start = NE+1 # extra parameters
            if O1version:
                start += 1
            for l in range(0,NE):
                data[extlabels[l]].append(float(spl[l+1]))
        for i in range(0,waveletnumber):
            for l in range(0,NW):
                if labels[l] == 'logA':
                    data[labels[l]].append(np.log10(float(spl[start+i*NW+l])))
                else:
                    data[labels[l]].append(float(spl[start+i*NW+l]))

    return data

#Plotting functions

######################################################################################################################
#....................................................
# New function to get SNR^2(t) for x-axis
#....................................................
def snrfunctime(median_waveform):
    powerList = []
    wave_integral = median_waveform
    time_integrate = time
    #integral at each
    dt = 0.0009765
    h_t_2 = []
    for line, line_1, time_i in zip(wave_integral, wave_integral[1:], time_integrate):
        t = time_i
        w0 = line
        h_t_2.append(line**2)
        w1  = line_1
        snr_t = (((w0**2)+(w1**2))/2)*dt
        powerList.append([snr_t,t])

    return(powerList)

# -------------------------------------------------
# Read in data and median waveform to get plot axis
# Now using SNR^2 for time axis determination
# -------------------------------------------------
def get_axes(jobName, postDir, ifoList, model, time, runFlags = RUN_FLAGS):
    axisList = []
    for ifo in ifoList:
        # -- Read Signal model
        local_runType, local_model = runType_and_model(model, runFlags)
        filename = str(jobName)+postDir+'{runType}/{model}_median_time_domain_waveform_{ifo}.dat'.format(ifo = ifoNames[int(ifo)], 
                model = local_model, runType = local_runType)

        try:
            timesamp, median_waveform, dummy1, dummy2, down_vec, up_vec = get_waveform(filename)
        except:
            print("get_axes: Couldn't find waveform file %s" % filename)


        # -- Get axis info
        snr_t = snrfunctime(median_waveform)
        #y
        wave = up_vec
        wave_max = wave.max()
        #x
        wave = median_waveform
        power, and_time = zip(*snr_t)
        power = list(power)

        #Fixed ~90$ rep. from 5% to 95% (10000x faster than loop)
        snr_t_arr = np.array(snr_t)
        sig_times = snr_t_arr[(snr_t_arr[:, 0] >= 0.0001*np.max(power)) & (snr_t_arr[:, 0]  <= 0.9999*np.max(power))][:, 1]

        #sig_times = []
        #Fixed ~90$ rep. from 5% to 95%
        #for row in snr_t:
        #    if (row[0] >= 0.0001*np.max(power) and row[0] <= 0.9999*np.max(power)):
        #        sig_times.append(row[1])
        try:
            axisList.append([np.min(sig_times), np.max(sig_times)+0.1*(np.max(sig_times)-np.min(sig_times)), -wave_max*1.1, wave_max*1.1])
        except:
            axisList.append([time[0],time[-1],-4.,4.])
    
    
    # -- Select Axis with biggest y-value
    ymax = 0
    axwinner = axisList[0]
    for i, axcand in enumerate(axisList):
        if axcand[3] > ymax:
            ymax = axcand[3]
            axwinner = axcand
    return(axwinner)


def get_axes_fdom(jobName, postDir, ifoList, model, time):
    
    ymin = np.zeros(len(ifoList))
    ymax = np.zeros(len(ifoList))
    
    for ifo in ifoList:
        # -- Read Signal model
    
        names = ['f','dpower','rpower','psd']
    
        filename = str(jobName)+postDir+'gaussian_noise_model_{0}.dat'.format(ifoNames[int(ifo)])
        
        data = np.recfromtxt(filename,names=names)
        
        # figure out what fmin is-- psd padded with 1's until fmin
        imin = 0
        while data['psd'][imin] == 1:
            imin += 1
        
        ymin[int(ifo)] = min(data['rpower'][imin:])
        ymax[int(ifo)] = max(data['rpower'][imin:])

        xmin = data['f'][imin]
        xmax = data['f'][-1]
    

    axwinner = [xmin, xmax, min(ymin), max(ymax)]
    return(axwinner)


# --------------
# Plot Evidence
# --------------
def plot_evidence(jobName, plotsDir, runFlags):

    sig_noise=0
    sig_gl=0
    sig_si=0
    err_sig_noise=0
    err_sig_gl=0
    err_sig_si=0

    # -- Read evidence data
    try:
        infile = open(str(jobName)+'evidence.dat', 'r')
        maxodds = 20
        for line in infile:
            spl = line.split()
            if spl[0] == 'noise':
                sig_noise = float(spl[1])
                err_sig_noise = float(spl[2])
                #evidence.dat files have alread store variance (not std dev)
                #err_sig_noise *= err_sig_noise
            if spl[0] == 'glitch':
                sig_gl = float(spl[1])
                err_sig_gl = float(spl[2])
                #err_sig_gl *= err_sig_gl
            if spl[0] == 'signal':
                sig_si = float(spl[1])
                err_sig_si = float(spl[2])
                #err_sig_si *= err_sig_si
        infile.close()
    except:
        sig_noise=0
        sig_gl=0
        sig_si=0
        err_sig_noise=0
        err_sig_gl=0
        err_sig_si=0
        

    sig_noise = sig_si - sig_noise
    sig_gl    = sig_si - sig_gl

    err_sig_noise += err_sig_si
    err_sig_noise = math.sqrt(err_sig_noise)
    err_sig_gl += err_sig_si
    err_sig_gl = math.sqrt(err_sig_gl)
    # -- Report to the user 
    print('   log( E_signal / E_noise ) = %f'%(sig_noise))
    print('   log( E_signal / E_glitch ) = %f'%(sig_gl))
    
    # -- Account for case where noise model is skipped
    if runFlags.noNoiseFlag:
        sig_noise = 0.0
        err_sig_noise = 1.0
        print('   log( E_signal / E_noise ) = N/A (no noise model)')
    else:
        # -- Report to the user
        print('   log( E_signal / E_noise ) = {0}'.format(sig_noise))
         
         
    if runFlags.fullOnly_flag:
        print('   log( E_signal / E_glitch ) = N/A (--fullOnly mode)')
    elif runFlags.modelList == ['signal']:
        sig_gl = 0.0
        err_sig_gl = 1.0
        print('   log( E_signal / E_glitch ) = N/A (no glitch model)')
    elif runFlags.modelList == ['glitch']:
        sig_gl = 0.0
        err_sig_gl = 1.0
        print('   log( E_signal / E_glitch ) = N/A (no signal model)')
    else:
        print('   log( E_signal / E_glitch ) = {0}'.format(sig_gl))
    # -- Plot the data point 
    plt.figure()
    plt.errorbar(sig_gl, sig_noise, 2*err_sig_gl, 2*err_sig_noise, color='black')
    # -- Store maxima and establish axes 
    maxodds = 1.1*np.array( [np.abs(sig_gl)+2*err_sig_gl, np.abs(sig_noise)+2*err_sig_noise, 20] ).max()
    xmaxodds = 1.1*np.maximum(np.abs(sig_gl)+2*err_sig_gl, 20)
    ymaxodds = 1.1*np.maximum(np.abs(sig_noise)+2*err_sig_noise, 20)
    plt.axis([-xmaxodds, xmaxodds, -ymaxodds, ymaxodds])
    # -- Color in the plot 
    plt.fill_between([0,maxodds], [0, 0], [maxodds, maxodds], facecolor=scolor, interpolate=True, alpha=0.3)
    plt.fill_between([-maxodds,0,maxodds], [-maxodds,0,0], [-maxodds, -maxodds, -maxodds], facecolor=ncolor, interpolate=True, alpha=0.3)
    plt.fill_between([-maxodds,0], [-maxodds, 0], [maxodds, maxodds], facecolor=gcolor, interpolate=True, alpha=0.3)
    plt.grid()
    # -- Labels on the plot
    plt.text(0.9*xmaxodds, 0.9*ymaxodds, 'Signal', horizontalalignment='right', verticalalignment='top')
    plt.text(-0.9*xmaxodds, 0.9*ymaxodds, 'Glitch', horizontalalignment='left', verticalalignment='top')
    plt.text(0.9*xmaxodds, -0.9*ymaxodds, 'Noise', horizontalalignment='right', verticalalignment='bottom')
    # -- Final touches 
    plt.xlabel(r'LN($B_{signal/glitch}$)')
    plt.ylabel(r'LN($B_{signal/noise}$)')
    plt.savefig(plotsDir+'odds.png')
    plt.close()
    
    
    # -- Return values to be used later
    return(sig_gl, sig_noise, err_sig_gl, err_sig_noise)

# -----------------------------------------------
# Plot the median waveform and injected waveform
# -----------------------------------------------
# If you are trying to plot 2 waveforms (like for full, cbcGlitch)
#   - Fill the second_XXX waveforms in with the data, and set model = runFlags.run_dirname()
def plot_waveform(jobName, postDir, ifo, plotsDir, worc, mdc, model, axwinner, time, low_50, high_50, low_90, high_90, median_waveform, second_low_50=[], second_high_50=[], second_low_90=[], second_high_90=[], second_median_waveform=[], runFlags = RUN_FLAGS):

    plt.figure()

    fig, ax = plt.subplots()

    # This gets the *data*
    try:
        filename = str(jobName)+postDir+'{1}_data_{0}.dat'.format(ifoNames[int(ifo)], worc)
        ifo_data = np.genfromtxt(filename)
        ax.plot(time, ifo_data, color = '0.75', linewidth=2, alpha=0.25)
    except:
        print("I couldn't find the file {0}".format(filename))

    #print('DEBUGGING')
    #print(filename)
    #print(time)
    # boolean if we are plotting multiple waveforms on one plot
    multi_waveforms = False
    # plot multiple waveforms 
    if runFlags.multi_type:
        #print('DEBUGGING 1')
        if model == runFlags.run_dirname():
            multi_waveforms = True
            colour = set_color(runFlags.modelList[0])
            second_color = set_color(runFlags.modelList[1])

    colour = set_color(model)
    print("COLOUR")
    print(colour)
    ax.fill_between(time, low_50, high_50, facecolor=colour, edgecolor=colour, alpha=0.5)
    ax.fill_between(time, low_90, high_90, facecolor=colour, edgecolor=colour, alpha=0.3)

    # full model double plotting
    if multi_waveforms:
      #print('DEBUGGING 2')
      plt.fill_between(time, second_low_50, second_high_50, facecolor=second_color, edgecolor=second_color, alpha=0.5)
      plt.fill_between(time, second_low_90, second_high_90, facecolor=second_color, edgecolor=second_color, alpha=0.3)

    if mdc:
        #print('DEBUGGING 3')
        # -- Read in the injected waveform
        try:
            filename = str(jobName)+postDir+'injected_{1}_waveform_{0}.dat'.format(ifoNames[int(ifo)], worc)
            inj_median_waveform = np.genfromtxt(filename)
        except:
            filename = str(jobName)+postDir+'injected_waveform_{0}.dat'.format(ifoNames[int(ifo)])
            inj_median_waveform = np.genfromtxt(filename)
        inj_time = time
        plt.plot(inj_time, inj_median_waveform, injcolor, linewidth=1)

    plt.plot(time, median_waveform, color=colour, linewidth=1, alpha=1)
    # plot second waveform 
    if multi_waveforms:
        plt.plot(time, second_median_waveform, color=second_color, linewidth=1, alpha=1)

    #axisList.append([sig_times.min(), sig_times.max(), -wave_max*1.1, wave_max*1.1])
    plt.xlabel('Time (s)')
    plt.ylabel('Whitened strain'.format(model,colour))
    plt.title('Reconstructed {0} model in {1}'.format(model,ifoNames[int(ifo)]))

    # -- Save the full versions of the plot
    if worc == 'whitened':
        plt.savefig(plotsDir+'{1}_waveform_{0}_full.png'.format(ifoNames[int(ifo)], model))
        plt.savefig('/home/fsarandrea/git/BayesWave/testdir_17/'+'{1}_waveform_{0}_full.png'.format(ifoNames[int(ifo)], model)); plt.close()
    else:
        #plt.savefig(plotsDir+'c{1}_waveform_{0}_full.png'.format(ifoNames[int(ifo)], model)) 
        plt.savefig('/home/fsarandrea/git/BayesWave/testdir_17/'+'c{1}_waveform_{0}_full.png'.format(ifoNames[int(ifo)], model)); plt.close()
    plt.axis(axwinner)
    # -- Save the plot
    if worc == 'whitened':
        plt.savefig(plotsDir+'{1}_waveform_{0}.png'.format(ifoNames[int(ifo)], model))
    else:
        plt.savefig(plotsDir+'c{1}_waveform_{0}.png'.format(ifoNames[int(ifo)], model))

    plt.close()




def plot_power_spectrum(jobName, postDir, ifo, plotsDir, worc, mdc, model, axwinner, freq, low_50, high_50, low_90, high_90, median_waveform,type):
    plt.figure()
    
    # To do: data and injected waveforms if possible
    colour = set_color(model)

    if mdc:
        filename = st(jobName)+postDir+'injected_whitened_spectrum_{0}.dat'.format(ifoNames[int(ifo)])

        injected_spectrum = np.genfromtxt(filename)

        plt.semilogy(injected_spectrum[:,0],injected_spectrum[:,1],color = '0.75', linewidth=2, alpha=0.25)



    plt.fill_between(freq, low_50, high_50, facecolor=colour, edgecolor=colour, alpha=0.5)
    plt.fill_between(freq, low_90, high_90, facecolor=colour, edgecolor=colour, alpha=0.3)
    
    if mdc:
        # -- Read in the injected waveform
        try:
            filename = str(jobName)+postDir+'injected_{1}_waveform_{0}.dat'.format(ifoNames[int(ifo)], worc)
            inj_median_waveform = np.genfromtxt(filename)
        except:
            filename = str(jobName)+postDir+'injected_waveform_{0}.dat'.format(ifoNames[int(ifo)])
            inj_median_waveform = np.genfromtxt(filename)
        inj_time = time
        plt.plot(inj_time, inj_median_waveform, injcolor, linewidth=1)

    if type == 'psd':
        filename = str(jobName)+postDir+'gaussian_noise_model_{0}.dat'.format(ifoNames[int(ifo)])
        data = np.recfromtxt(filename, names = ['f', 'dpower', 'rpower', 'Sn'])
        plt.semilogy(data['f'],data['rpower'], ncolor, alpha=0.6)
        plt.ylim(min(data['rpower']),max(data['rpower']))


    plt.semilogy(freq, median_waveform, color=colour, linewidth=1, alpha=1)
    
    #axisList.append([sig_times.min(), sig_times.max(), -wave_max*1.1, wave_max*1.1])
    plt.xlabel('Frequency (Hz)')
    if type == 'psd':
        plt.ylabel('PSD and data (grey)'.format(model))
    if type == 'powerspec':
         plt.ylabel('Whitened power spectrum of median waveform'.format(model))
    plt.title('{0}'.format(ifoNames[int(ifo)]))



    # -- Save the full versions of the plot
    if worc == 'whitened':
        plt.savefig(plotsDir+'{1}_{2}_{0}_full.png'.format(ifoNames[int(ifo)], model, type))
    else:
        plt.savefig(plotsDir+'c{1}_{2}_{0}_full.png'.format(ifoNames[int(ifo)], model, type))


    plt.xlim(32,512)
    
#plt.axis(axwinner)
    # -- Save the plot
    if worc == 'whitened':
        plt.savefig(plotsDir+'{1}_{2}_{0}.png'.format(ifoNames[int(ifo)], model, type))
    else:
        plt.savefig(plotsDir+'c{1}_{2}_{0}.png'.format(ifoNames[int(ifo)], model, type))
    
    plt.close()

def plot_full_spectro(jobName, postDir, ifo, plotsDir, worc, mdc, models, axwinner, psd_info, powerspec_infos, runFlags = RUN_FLAGS):
    
    plt.figure()

    if mdc:
        filename = str(jobName)+postDir+'injected_whitened_spectrum_{0}.dat'.format(ifoNames[int(ifo)])

        injected_spectrum = np.genfromtxt(filename)

        plt.semilogy(injected_spectrum[:,0],injected_spectrum[:,1],injcolor, linewidth=1)


    names = ['f','dpower','rpower','psd']
    filename = str(jobName)+postDir+'gaussian_noise_model_{0}.dat'.format(ifoNames[int(ifo)])

    data = np.recfromtxt(filename,names=names)
    # plot data
    plt.semilogx(data['f'],data['rpower'],color=ncolor,alpha=0.5)

    # plot psd
    plt.fill_between(psd_info[0],psd_info[4],psd_info[5],color='grey',alpha=0.8)
    plt.semilogy(psd_info[0],psd_info[1],color='k',ls='-')

    # plot powerspec of models
    for i, model in enumerate(models):
        if model == 'clean':
            continue
        colour = set_color(model)
        powerspec_info = powerspec_infos[i]
        plt.fill_between(powerspec_info[0],powerspec_info[2],powerspec_info[3],color=colour,alpha=0.5)
        plt.fill_between(powerspec_info[0],powerspec_info[4],powerspec_info[5],color=colour,alpha=0.3)
        plt.plot(powerspec_info[0],powerspec_info[1],color=colour)

    plt.xscale('log')
    try:
        plt.yscale('log',nonpositive='clip')
    except ValueError:
        print("Oops! You're using older version of matplotlib, thats ok")
        plt.yscale('log',nonposy='clip')


    plt.ylim(axwinner[2],axwinner[3])
    plt.xlim(axwinner[0],axwinner[1])

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    if len(models) == 1:
        plt.title("Power Spectra for {0} model in {1}".format(models[0],ifoNames[int(ifo)]))
        plt.savefig(plotsDir+'{0}_frequence_domain_{1}.png'.format(model,ifoNames[int(ifo)]))
    else:
        plt.title("Power Spectra for {0} model in {1}".format(runFlags.run_dirname(),ifoNames[int(ifo)]))
        plt.savefig(plotsDir+'{0}_frequence_domain_{1}.png'.format(runFlags.run_dirname(),ifoNames[int(ifo)]))


    plt.close()



def plot_tf_tracks(jobName, postDir, ifo, plotsDir, worc, mdc, model, axwinner, f_axwinner,time, low_50, high_50, low_90, high_90, median_waveform):
    plt.figure()
    
    # To do: data and injected waveforms if possible
    colour = set_color(model)
    
    #t = time - 2.0
    t = time;

    # -- Kludgy fix to keep megaplot from crashing when data is bad
    if np.amax(high_50) > 1e5: # Something is wrong
      high_50 = np.zeros(high_50.shape)

    if np.amax(high_90) > 1e5:
      high_90 = np.zeros(high_90.shape)


    plt.fill_between(t, low_50, high_50, facecolor=colour, edgecolor=colour, alpha=0.5)
    plt.fill_between(t, low_90, high_90, facecolor=colour, edgecolor=colour, alpha=0.3)

    if mdc:
        # -- Read in the injected waveform
        try:
            filename = str(jobName)+postDir+'injected_tf_{0}.dat'.format(ifoNames[int(ifo)], worc)
            inj_median_waveform = np.genfromtxt(filename)
        except:
            filename = str(jobName)+postDir+'injected_tf_{0}.dat'.format(ifoNames[int(ifo)])
            inj_median_waveform = np.genfromtxt(filename)
        inj_time = time
        plt.plot(inj_time, inj_median_waveform, injcolor, linewidth=1)
    
    
    plt.plot(t, median_waveform, color=colour, linewidth=1, alpha=1)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Time vs. Frequency for {0} model in {1}'.format(model,ifoNames[int(ifo)]))
    
    
    
    # -- Save the full versions of the plot
    plt.savefig(plotsDir+'{1}_tf_{0}_full.png'.format(ifoNames[int(ifo)], model))


    plt.xlim(axwinner[0],axwinner[1])
    plt.ylim(f_axwinner[0],f_axwinner[1])
    
    #plt.axis(axwinner)
    # -- Save the plot
    plt.savefig(plotsDir+'{1}_tf_{0}.png'.format(ifoNames[int(ifo)], model))
    
    plt.close()


# --------------------------------------
# Plot Q scans of waveforms
# --------------------------------------
def Q_scan(subDir,model, Q, t, f, ifo, axwinner, f_axwinner, climwinner=[1,1], runFlags = RUN_FLAGS, cmap_name = 'viridis'):
    if (model == 'data'):
        filename = postDir+'{model}/{model}_spectrogram_{Q}_{ifo}.dat'.format(model = model, Q = Q, ifo = ifoNames[int(ifo)])
    elif (model == 'injected'):
        filename = postDir+'{model}_spectrogram_{Q}_{ifo}.dat'.format(model = model, Q = Q, ifo = ifoNames[int(ifo)])
    elif model == 'clean':
        filename = postDir+'{runType}/{model}_spectrogram_{Q}_{ifo}.dat'.format(runType = 'clean', model = 'glitch', Q = Q, ifo = ifoNames[int(ifo)])
    elif runFlags.multi_type:
        filename = postDir+'{runType}/{model}_spectrogram_{Q}_{ifo}.dat'.format(runType = runFlags.run_dirname(), model = local_model, Q = Q, ifo = ifoNames[int(ifo)])
    else:
        filename = postDir+'{runType}/{model}_spectrogram_{Q}_{ifo}.dat'.format(runType = subDir, model = model, Q = Q, ifo = ifoNames[int(ifo)])
    
    data = np.genfromtxt(filename)

    fig, ax = plt.subplots()
    try:
        qplot = ax.imshow(data,aspect='auto',origin='lower',extent=[t[0],t[-1],f[0],f[-1]], cmap = cmap_name) 
    except:
        print("Oops! you're probably using python 2, {cmap} doesn't exist yet!".format(cmap = cmap_name))
        qplot = ax.imshow(data,aspect='auto',origin='lower',extent=[t[0],t[-1],f[0],f[-1]], cmap='OrRd')


    # Set the clim of the cmap based on which was 'best' as calculated from the data
    if 'residual' in model:
        qplot.set_clim(climwinner)
    else:
        qplot.set_clim(np.min(data),np.max(data))

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Spectrogram of median {0} waveform in {1}, Q={2}'.format(model.replace('_','\_'),ifoNames[int(ifo)], Q))

    ax.set_xlim(axwinner[0],axwinner[1])
    ax.set_ylim(f_axwinner[0],f_axwinner[1])
    try:
        ax.set_yscale('log', base = 2) 
    except ValueError:
        #print("Oops! You're using older version of matplotlib, but thats ok")
        ax.set_yscale('log', basey = 2) 

    #sets yscales to numbers instead of powers 
    for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_formatter(ScalarFormatter())
    # make a colorbar 
    plt.colorbar(qplot)
    if subDir == 'clean' and model == 'glitch_residual':
        plt.savefig(plotsDir+'{model}_spectrogram_Q{Q}_{ifo}.png'.format(model = 'clean_residual', Q = Q, ifo = ifoNames[int(ifo)]))
    else:
        plt.savefig(plotsDir+'{model}_spectrogram_Q{Q}_{ifo}.png'.format(model = model, Q = Q, ifo = ifoNames[int(ifo)]))

    plt.close()

    return [np.min(data),np.max(data)]

# 2d Scatterplot colored by the chain temperature 
def scatter_cbc_chains(param1, param2, chainmin = 0, chainmax = RUN_FLAGS.Nchain, runFlags = RUN_FLAGS, plotsDir = 'plots/', cmap_name = 'viridis'):
    fig, ax = plt.subplots(figsize = (6,6))
    cmap = mpl.cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=0, vmax=runFlags.Nchain)
    # Use the nice latex for the x and y axis 
    ax.set_xlabel(cbc_dict[param1][2])
    ax.set_ylabel(cbc_dict[param2][2])
    index1 = cbc_dict[param1][0] 
    index2 = cbc_dict[param2][0] 
    ax.set_title("Chains {0} through {1}".format(chainmin, chainmax-1))
    # TODO: skip burnin
    for chain_index in np.arange(chainmin, chainmax):
        param_chains = np.genfromtxt('./chains/cbc_params.dat.%i'%chain_index)

        ax.plot(param_chains[:, index1], param_chains[:, index2], ',', 
                     color = cmap(norm(chain_index)), 
                     zorder = 19 - chain_index,  alpha = 0.05) #alpha = 1/(chain_index + 1))

    plt.savefig(plotsDir + '{0}_{1}_2dscatter.png'.format(param1, param2))

def scatter_chains_wavelet(param1, param2, ifo, model, chainmin = 0, chainmax = RUN_FLAGS.Nchain, runFlags = RUN_FLAGS, plotsDir = 'plots/', cmap_name = 'viridis'):
    # Make sure model = cbc for a CBCType run
    fig, ax = plt.subplots(figsize = (6,6))
    cmap = mpl.cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=0, vmax=runFlags.Nchain)
    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_title("Chains {0} through {1}".format(chainmin, chainmax-1))
    for chain_index in np.arange(chainmin, chainmax):
        if runFlags.CBCType:
            # TODO make this work for signalCBC run
            # I don't know what the filename would be for a signalCBC run. 
            filename = 'chains/cbc_params_{ifo}.dat.{chain_index}'.format(ifo = ifo, chain_index = chain_index)
        elif model == 'signal':
            filename = 'chains/{model}_params_h0.dat.{2}'.format(model = model, chain_index = chain_index)
        else:
            filename = 'chains/{model}_params_{ifo}.dat.{chain_index}'.format(model = model, ifo = ifo, chain_index = chain_index)
        param_chains = get_wavelet_params(filename, model, chirpflag = runFlags.chirplets)

        ax.plot(param_chains[param1], param_chains[param2], ',', 
                     color = cmap(norm(chain_index)), 
                        zorder = 19 - chain_index,  alpha = 0.05) #alpha = 1/(chain_index + 1))
        
    plt.savefig(plotsDir + '{model}{ifo}_{param1}_{param2}_2dscatter_chains_{chainmin}_{chainmax}.png'.format(param1 = param1, param2 = param2, 
                                                                                        model = model, chainmin = chainmin, 
                                                                                        chainmax = chainmax, ifo = ifo))
    plt.close()

            



# -------
# Corner plot for CBC Parameters 
# -------
def make_2d_hist_cbc(ax, param_names, Params, cbc_dict, mdc, runFlags = RUN_FLAGS):
    NUM_LEVELS = 10
    index1, index2 = cbc_dict[param_names[0]][0], cbc_dict[param_names[1]][0]
    
    Yd=Params[:,index2]
    Xd=Params[:,index1]
    xmin=Xd.min()
    xmax=Xd.max()
    ymin=Yd.min()
    ymax=Yd.max()
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([Xd, Yd])
    kernel = scs.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    ax.contour(X, Y, Z, NUM_LEVELS, cmap=plt.cm.Blues)
    
    #ax.plot(2,3,label=r'CBC+glitch+noise')
    if mdc:
        xinj = runFlags.get_xml_parameter(param_names[0])
        yinj = runFlags.get_xml_parameter(param_names[1])
        ax.plot(xinj, yinj,'x',color='k',markersize=10)
    #ax.set_xlim(-1,1)
    #ax.set_ylim(200,1500)
    
    ax.grid(True, alpha=0.5)
    #ax.set_xlabel(cbc_dict[param_names[0]][2],fontsize=23)
    #ax.set_ylabel(cbc_dict[param_names[1]][2],fontsize=23)
    #plt.legend(frameon=True,fancybox=True,framealpha=0.5,loc="lower left",fontsize=10)

def make_2d_hist(ax, Xd, Yd, N = 10):
    #ax.scatter(Xd, Yd, color = 'blue', s = 2, alpha = 0.05)
    #ax.hexbin(Xd, Yd, gridsize = 100, bins = 'log')
    xmin=Xd.min()
    xmax=Xd.max()
    ymin=Yd.min()
    ymax=Yd.max()
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([Xd, Yd])
    kernel = scs.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    ax.contour(X, Y, Z, N, cmap=plt.cm.Blues)
    ax.grid(True, alpha=0.5)
    return

def corner_plot_hist(ax, param_name, cbc_dict, mdc, runFlags = RUN_FLAGS):
    index = cbc_dict[param_name][0]
    n, bins, patches = ax.hist(Params[:, index], bins = int(np.sqrt(len(Params[:, index]))), color = '#D69E8D')
    if mdc:
        injected = runFlags.get_xml_parameter(param_name)
        ax.vlines([injected], [0], [max(n)], color = injcolor)
    return

# --------------------
#   Makes a corner plot from a dictionary 
# --------------------
def make_corner_plot_dict(Dict, skip = [], plotsDir= 'plots/', name = 'glitch_cornerplot.png', 
        chaincolor = '#736D6B', histcolor = '#D69E8D', names = None, scatter_color = 'blue', title = None):
    key_list = list(Dict.keys())
    keys = []
    for k in key_list:
        if k in skip:
            continue
        else:
            keys.append(k)
    print("making cornerplot with", keys)
    fig, axes = plt.subplots(len(keys), len(keys) + 1, figsize = (10,10))
    fig.suptitle(title)
    for i in range(len(keys)): # over rows 
        for j in range(len(keys) + 1): # over columns 
            # indexes the 
            ax = axes[i, j]

            if j == i:
                ax.hist(Dict[keys[i]], bins = int(np.sqrt(len(Dict[keys[i]]))), color = histcolor) 
            elif j == i + 1:
                # plot 1/10 of the chains 
                ax.plot(np.arange(0, len(Dict[keys[i]]), 10), Dict[keys[i]][::10], ',',  color = chaincolor)
            elif j > i:
                ax.set_visible(False)
                continue
            else:
                try:
                    if len(Dict[keys[j]]) > 5e5:
                        n = int(len(Dict[keys[j]]) / 5e5)
                        print("Dict is long! Sampling only 5e5 of them", keys[j], len(Dict[keys[j]]), len(Dict[keys[j]]) / n)
                        ax.plot(Dict[keys[j]][::n],Dict[keys[i]][::n], ',', 
                                alpha=0.1, color = scatter_color)
                        # makes sure that the kernel estimate isn't too huge!
                        #make_2d_hist(ax, np.array(Dict[keys[j]])[::n], np.array(Dict[keys[i]])[::n])
                    else:
                        ax.plot(Dict[keys[j]],Dict[keys[i]], ',', 
                                alpha=0.1, color = scatter_color)
                        #make_2d_hist(ax, np.array(Dict[keys[j]]), np.array(Dict[keys[i]]))
                except ValueError as e:
                    print("making 2d hist, slice on {key1} and {key2} wasn't 2d, continuing".format(key1 = keys[j], key2 = keys[i]))
                    print(e)
                
            # set labels
            if j == 0:
                if names is None:
                    ax.set_ylabel(keys[i])
                else:
                    ax.set_ylabel(names[i])
            #elif i == 0:
            #    ax.set_ylabel("")
            else:
                ax.set_yticklabels([])
            
            #last row 
            if (i == len(keys) - 1) and j < len(keys):
                if names is None:
                    ax.set_xlabel(keys[j])
                else:
                    ax.set_xlabel(names[j])
            else:
                ax.set_xticklabels([])
    plt.savefig(plotsDir + name)
    plt.close()

    
    
def make_cbc_corner_plot(param_names, Params, cbc_dict, plotsDir, mdc, 
                         chain_index = 0, chain_color = '#736D6B', scatter_color = 'blue', runFlags = RUN_FLAGS):
    print('making corner_plot with cbc parameters')
    fig, axes = plt.subplots(len(param_names), len(param_names) + 1, figsize = (20,20))
    for i in range(len(param_names)): # rows (y axis)
        for j in range(len(param_names) + 1): # columns (x axis)
            # indexes the 
            ax = axes[i, j]

            if j == i:
                # Plot the histogram of this parameter 
                if mdc:
                    inj = runFlags.get_xml_parameter(param_names[i])
                corner_plot_hist(ax, param_names[i], cbc_dict, mdc)
            elif j == i + 1:
                if mdc:
                    # plot what injected value actuall was 
                    inj = runFlags.get_xml_parameter(param_names[i])
                    ax.plot([0,len(Params[:, cbc_dict[param_names[i]][0]])], 
                            [inj, inj],  color = set_color('injection'))
                # Plot the chains for this parameter, but only plotting 1/10th of the total samples
                ax.plot(np.arange(0,len(Params[:, cbc_dict[param_names[i]][0]]), 10), 
                        Params[:, cbc_dict[param_names[i]][0]][::10], ',',  color = chain_color)
                # plot the injected value 

            elif j > i:
                ax.set_visible(False)
                continue
            else:
                # plot the 2d histogram of these parameters
                param_index_x = cbc_dict[param_names[j]][0]
                param_index_y = cbc_dict[param_names[i]][0]
                if len(param_names[i]) > 5e5:
                    n = max([int(len(param_names[i]) / 5e5), 1])
                    ax.plot(Params[:, param_index_x][::n], Params[:, param_index_y][::n], ',', 
                            alpha=0.1, color = scatter_color)
                    #make_2d_hist_cbc(ax, [param_names[j], param_names[i]], Params, cbc_dict, mdc)
                else:
                    ax.plot(Params[:, param_index_x], Params[:, param_index_y], ',',
                            alpha=0.1, color = scatter_color)
                    #make_2d_hist_cbc(ax, [param_names[j], param_names[i]], Params, cbc_dict, mdc)
                if mdc:
                    xinj = runFlags.get_xml_parameter(param_names[j])
                    yinj = runFlags.get_xml_parameter(param_names[i])
                    ax.plot(xinj, yinj,'x',color='k',markersize=10)

            # sets y axis names 
            if j == 0:
                #print(j, cbc_dict[param_names[i]][2])
                ax.set_ylabel(cbc_dict[param_names[i]][2])
            else:
                ax.set_yticklabels([])
                
            if i == len(param_names) - 1 and j != len(param_names):
                #print(j, cbc_dict[param_names[j]][2])
                ax.set_xlabel(cbc_dict[param_names[j]][2])
            else:
                ax.set_xticklabels([])

    fig.suptitle("CBC Cornerplot Chain {0}".format(chain_index))
    plt.savefig(plotsDir + 'cbc_cornerplot_{0}.png'.format(chain_index))
    plt.close()

def make_skymap(RA, sindec, injected = None, plotsDir = "plots/", chain_index = 0, scatter_color = 'blue', cmap_name = 'viridis'):
    index = np.arange(0, len(RA))
    fig, axs = plt.subplots(2,2, figsize = (8,8))
    #cmap = mpl.cm.get_cmap('plasma')
    #norm = mpl.colors.Normalize(vmin=min(indices), vmax=max(indices))

    # Plot the skymap 
    ax = axs[1,0]
    if injected is not None:
        ax.scatter(injected[0], injected[1], color = 'black', zorder = 10)
    ax.hist2d(RA,sindec, bins = int(np.sqrt(len(RA))), cmap = cmap_name)
    ax.set_xlabel('RA')
    ax.set_ylabel('sindec')

    # Scatter the skymap 
    ax = axs[0,1]
    if injected is not None:
        ax.scatter(injected[0], injected[1], color = 'black', zorder = 10)
    ax.plot(RA, sindec, ',', alpha = 0.5, color = scatter_color)
    #ax.set_xlim(0, 2 * np.pi)
    #ax.set_ylim(-1, 1)

    # Plot chains 
    ax = axs[1, 1]
    # Plot chains for sindec 
    if injected is not None:
        ax.plot([0, len(sindec)], [injected[1], injected[1]], color = 'teal')
    ax.plot(index[::10], sindec[::10], ',', color = '#736D6B')

    ax = axs[0, 0]
    if injected is not None:
        ax.plot([0, len(RA)], [injected[0], injected[0]], color = 'teal')
    ax.plot(index[::10], RA[::10], ',', color = '#736D6B')

    plt.savefig(plotsDir + 'cbc_skymap_{0}.png'.format(chain_index))
    plt.close()

def skymap_posterior(ra,sindec,inj_ra=None,inj_sindec=None):
    '''
    Megasky is dead. Long live megasky.
    inputs: ra and sindec chains
    outputs: Plot of 2d histogram of RA and Dec in Mollweide projection
    '''
    print('\nMaking sky map.')
    # Toss out burnin
    ra = ra[int(len(ra)/2):]
    sindec = sindec[int(len(sindec)/2):]

    ra = np.pi-np.array(ra) # map from -pi to pi instead of 0 to 2pi
    dec = np.arcsin(np.array(sindec))
    sky_data = np.array([ra,dec])

    # create bin edges
    bin_number = 150
    xedges = np.linspace(-np.pi, np.pi, bin_number + 1)
    yedges = np.linspace(-np.pi/2., np.pi/2., bin_number + 1)

    hist, xedges, yedges = np.histogram2d(
        *sky_data, bins=[xedges, yedges], density=True
    )

    cmap = plt.cm.OrRd

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')

    ax.pcolor( 
        xedges[:-1], yedges[:-1], hist.T,  # transpose from (row, column) to (x, y)
        cmap=cmap, shading='auto'
    )

    if inj_ra and inj_sindec:
        inj_ra = np.pi-inj_ra
        inj_dec = np.arcsin(inj_sindec)

        ax.scatter(inj_ra,inj_dec,label='injected sky location')
        ax.legend(loc='best')


    # Set axes ticks to match what was the standard with previous sky maps
    ax.set_yticks([-1,-.5 ,0,.5,1])
    ax.tick_params(labelright=True)
    ax.set_xticks([-2.25,-1.5,-0.75,0,0.75,1.5,2.25])
    xlab = ['21h','18h','15h','12h','9h','6h','3h']
    ax.set_xticklabels(xlab)
    ax.grid(which='major',alpha=0.3)

    plt.savefig(plotsDir + 'skymap.png')

    print('Done making skymap.')


def make_histogram(moments, moment, model, ifo, plotsDir):
    colour = set_color(model)
    injcolour = injcolor

    # -- Start figure
    plt.figure()

    histmoments = moments[moment]

    if moment == 't0_rec': 
        rn,rbins,patches = plt.hist(histmoments - 2, bins=50, label='Recovered', alpha=0.5, linewidth=0, color=colour)
    elif 'overlap' in moment: # For overlap want to make sure we get the right axes range
        rbins = np.arange(-1.02,1.02,0.01)
        rn,rbins,patches = plt.hist(histmoments, rbins, alpha=0.5, linewidth=0, color=colour)
        plt.xlim(-1.1, 1.1)
    else:
        rn,rbins,patches = plt.hist(histmoments, bins=50, label='Recovered', alpha=0.5, linewidth=0, color=colour)
    if 'overlap' not in moment:
        if moment == 't0_rec':
            inmode = np.median(injmoments[moment] - 2) # Correcting for the 2s added by BW to the data time window
        else:
            inmode = np.median(injmoments[moment])

        if mdc:
            plt.axvline(x=inmode, label='Injected', color=injcolour)
            plt.legend()

    # -- Set title by IFO, or overlap ()
    if 'network_overlap' in moment:
        plt.title('Network Overlap')
    else:
        plt.title('{0}'.format(ifoNames[int(ifo)]))

    # -- Plot settings
    plt.ylim(1, 2*rn.max())
    plt.xlabel(moments_dict[moment][0])
    plt.grid()
        
    if not moment == 'network_overlap':
        plt.savefig(plotsDir+model+'_'+moments_dict[moment][1]+'_{0}.png'.format(ifoNames[int(ifo)]))
    else:
        plt.savefig(plotsDir+model+'_'+moments_dict[moment][1]+'.png')
                    
    plt.close()



# -----------------
# Plot Temp Chains
# -----------------
def plot_likelihood_1(modelList, plotsDir, runFlags = RUN_FLAGS):
    plt.figure()
    plt.title('Likelihood')
    plt.xlim([1e-4, 1])
    minlist = []
    maxlist = []

    # TODO add CBC stuff to this? 
    if runFlags.fullOnly_flag:
        mod = 'full'
        colour = 'seagreen'
        try:
            names = ['temp','likehood','error','acl']
            data = np.recfromtxt(str(jobName)+"{0}_evidence.dat".format(mod), names=names)
        except:
            try:
                names = ['temp','likehood','error']
                data = np.recfromtxt(str(jobName)+"{0}_evidence.dat".format(mod), names=names)
            except:
                pass
        #error = np.zeros(likehood.shape)
        plt.semilogx(data['temp'], data['likehood'], label=mod, linewidth=2, color=colour)
        plt.errorbar(data['temp'], data['likehood'], 2*data['error'], color=colour)
        minlist.append(data['likehood'][np.where(data['temp']>1e-4)[0][-1]])
        maxlist.append(data['likehood'][0])



    else:
        for mod in modelList:
            colour = set_color(mod)
            try:
                names = ['temp','likehood','error','acl']
                data = np.recfromtxt(str(jobName)+"{0}_evidence.dat".format(mod), names=names)
            except:
                try:
                   names = ['temp','likehood','error']
                   data = np.recfromtxt(str(jobName)+"{0}_evidence.dat".format(mod), names=names)
                except:
                    continue
            #error = np.zeros(likehood.shape)
            plt.semilogx(data['temp'], data['likehood'], label=mod, linewidth=2, color=colour)
            plt.errorbar(data['temp'], data['likehood'], 2*data['error'], color=colour)
            minlist.append(data['likehood'][np.where(data['temp']>1e-4)[0][-1]])
            maxlist.append(data['likehood'][0])

        # Now do noise

        if not runFlags.noNoiseFlag:
            mod = "noise"

            colour = ncolor

            try:
                names = ['temp','likehood','error','acl']
                data = np.recfromtxt(str(jobName)+"{0}_evidence.dat".format(mod), names=names)
            except:
                try:
                    names = ['temp','likehood','error']
                    data = np.recfromtxt(str(jobName)+"{0}_evidence.dat".format(mod), names=names)
                except:
                    print('No noise evidence')
            #error = np.zeros(likehood.shape)
            plt.semilogx(data['temp'], data['likehood'], label=mod, linewidth=2, color=colour)
            plt.errorbar(data['temp'], data['likehood'], 2*data['error'], color=colour)
            minlist.append(data['likehood'][np.where(data['temp']>1e-4)[0][-1]])
            maxlist.append(data['likehood'][0])

    plt.ylim(min(minlist)-1000,max(maxlist)+500)
    plt.xlabel( '1/Temp' )
    plt.ylabel('log(L)')
    plt.grid()
    plt.legend(loc=2)
    plt.savefig(plotsDir+'likelihood.png')
    plt.close()

def plot_likelihood_2(modelList, plotsDir, runFlags = RUN_FLAGS):
    plt.figure()
    plt.title('Likelihood')

    if runFlags.fullOnly_flag:
        mod = 'full'
        try:
            names = ['temp','likehood','error','acl']
            data = np.recfromtxt(str(jobName)+"{0}_evidence.dat".format(mod), names=names)
        except:
            try:
                names = ['temp','likehood','error']
                data = np.recfromtxt(str(jobName)+"{0}_evidence.dat".format(mod), names=names)
            except:
                pass
        plt.semilogx(data['temp'], data['likehood']*data['temp'], label=mod, linewidth=2, color=colour)
        plt.errorbar(data['temp'], data['likehood']*data['temp'], 2*data['error'], color=colour)


    else:
        for mod in modelList:
            colour = set_color(mod)
            try:
                names = ['temp','likehood','error','acl']
                data = np.recfromtxt(str(jobName)+"{0}_evidence.dat".format(mod), names=names)
            except:
                try:
                   names = ['temp','likehood','error']
                   data = np.recfromtxt(str(jobName)+"{0}_evidence.dat".format(mod), names=names)
                except:
                    continue
            plt.semilogx(data['temp'], data['likehood']*data['temp'], label=mod, linewidth=2, color=colour)
            plt.errorbar(data['temp'], data['likehood']*data['temp'], 2*data['error'], color=colour)
    plt.grid()
    plt.xlabel('1/Temp')
    plt.ylabel('log(L) X 1/Temp')
    plt.legend(loc=2)
    plt.savefig(plotsDir+'TL.png')
    plt.close()

# ------------------------
# Plot chains of CBC model
# ------------------------

def plot_chains(param_chain, param_name,  plotsDir, runFlags, color = 'mediumseagreen'):
    fig, ax = plt.subplots()
    index = np.arange(0, len(param_chain), 1)
    ax.plot(index, param_chain, ',',  color = color)
    ax.set_xlabel('Chain index')
    ax.set_ylabel(param_name)
    plt.savefig(plotsDir + 'cbc_{0}_chain.png'.format(param_name))
    plt.close()

def plot_hists(param_chain, param_name,  plotsDir, runFlags, color = 'mediumseagreen', injected_value = None):
    fig, ax = plt.subplots()
    N_bins = int(np.sqrt(len(param_chain)))
    n, bins, patches, = ax.hist(param_chain, bins = N_bins, color = color, alpha = 0.5, linewidth = 0, zorder = 0)
    if injected_value is not None:
        ax.vlines(injected_value, 0, max(n), color = injcolor, label = 'injected', zorder = 10)
    ax.set_xlabel(param_name)
    ax.set_ylabel('counts')
    plt.savefig(plotsDir + 'cbc_{0}_hist.png'.format(param_name))
    plt.close()

# -----------------
# Plot Stokes Parameters
# -----------------
def stokes_plots(postDir, plotsDir, flow, srate):
    U_datafile = postDir + "/signal/signal_median_frequency_domain_stokes_U.dat"
    eps_sqr_datafile = postDir + "/signal/signal_median_frequency_domain_stokes_eps_sqr.dat"
    eps1_datafile = postDir + "/signal/signal_median_frequency_domain_stokes_eps1.dat"
    eps2_datafile = postDir + "/signal/signal_median_frequency_domain_stokes_eps2.dat"

    U_data = np.loadtxt(U_datafile)
    eps_sqr_data = np.loadtxt(eps_sqr_datafile)
    eps1_data = np.loadtxt(eps1_datafile)
    eps2_data = np.loadtxt(eps2_datafile)

    U = U_data[:,1]                 # U  -- should be 0 if elliptically polarized
    eps_sqr = eps_sqr_data[:,1]     # (I-Q)/(I+Q) -- should be epsilon^2 if elliptically polarized
    eps1 = eps1_data[:,1]           # V/(I+Q) -- should be epsilon if elliptically polarized
    eps2 = eps2_data[:,1]           # (I-Q)/V -- should be epsilon if elliptically polarized

    U_lower50 = U_data[:,2]
    eps_sqr_lower50 = eps_sqr_data[:,2]
    eps1_lower50 = eps1_data[:,2]
    eps2_lower50 = eps2_data[:,2]

    U_upper50 = U_data[:,3]
    eps_sqr_upper50 = eps_sqr_data[:,3]
    eps1_upper50 = eps1_data[:,3]
    eps2_upper50 = eps2_data[:,3]

    N = U_data.shape[0]
    ff = U_data[:,0]
    plot_range = np.where((flow<ff) & (ff<srate/2))

    injStokes = False
    if os.path.exists(postDir+"/injected_frequency_domain_stokes_U.dat"):
        injStokes = True

    if injStokes:
        print("Plotting injected Stokes parameters as well")
        
        U_inj_datafile = postDir + "/injected_frequency_domain_stokes_U.dat"
        eps_sqr_inj_datafile = postDir + "/injected_frequency_domain_stokes_eps_sqr.dat"
        eps1_inj_datafile = postDir + "/injected_frequency_domain_stokes_eps1.dat"
        eps2_inj_datafile = postDir + "/injected_frequency_domain_stokes_eps2.dat"

        U_inj_data = np.loadtxt(U_inj_datafile)
        eps_sqr_inj_data = np.loadtxt(eps_sqr_inj_datafile)
        eps1_inj_data = np.loadtxt(eps1_inj_datafile)
        eps2_inj_data = np.loadtxt(eps2_inj_datafile)

        U_inj = U_inj_data[:,1]                 # U  -- should be 0 if elliptically polarized
        eps_sqr_inj = eps_sqr_inj_data[:,1]     # (I-Q)/(I+Q) -- should be epsilon^2 if elliptically polarized
        eps1_inj = eps1_inj_data[:,1]           # V/(I+Q) -- should be epsilon if elliptically polarized
        eps2_inj = eps2_inj_data[:,1]           # (I-Q)/V -- should be epsilon if elliptically polarized

    #full axis plots
    fig, axs = plt.subplots(2, 2)
    
    axs[0,0].fill_between(ff[plot_range], U_lower50[plot_range], U_upper50[plot_range], color=scolor, alpha=0.3)
    if injStokes: axs[0,0].plot(ff[plot_range], U_inj[plot_range], color=injcolor, linewidth=0.5)
    axs[0,0].plot(ff[plot_range], U[plot_range], color=scolor, linewidth=0.5)
    axs[0,0].set_xscale('log')
    axs[0,0].set_xlim((flow,srate/2))
    axs[0,0].set_xlabel(r'$f$ [Hz]')
    axs[0,0].set_ylabel(r'$U \ (0)$')

    axs[0,1].fill_between(ff[plot_range], eps_sqr_lower50[plot_range], eps_sqr_upper50[plot_range], color=scolor, alpha=0.3)
    if injStokes: axs[0,1].plot(ff[plot_range], eps_sqr_inj[plot_range], color=injcolor, linewidth=0.5)
    axs[0,1].plot(ff[plot_range], eps_sqr[plot_range], color=scolor, linewidth=0.5)
    axs[0,1].set_xscale('log')
    axs[0,1].set_xlim((flow,srate/2))
    axs[0,1].set_yscale('log')
    axs[0,1].set_xlabel(r'$f$ [Hz]')
    axs[0,1].set_ylabel(r'$(I-Q)/(I+Q) \ (\varepsilon ^2)$')

    axs[1,0].fill_between(ff[plot_range], eps1_lower50[plot_range], eps1_upper50[plot_range], color=scolor, alpha=0.3)
    if injStokes: axs[1,0].plot(ff[plot_range], eps1_inj[plot_range], color=injcolor, linewidth=0.5)
    axs[1,0].plot(ff[plot_range], eps1[plot_range], color=scolor, linewidth=0.5)
    axs[1,0].set_xscale('log')
    axs[1,0].set_xlim((flow,srate/2))
    axs[1,0].set_xlabel(r'$f$ [Hz]')
    axs[1,0].set_ylabel(r'$V/(I+Q) \ (\varepsilon)$')

    axs[1,1].fill_between(ff[plot_range], eps2_lower50[plot_range], eps2_upper50[plot_range], color=scolor, alpha=0.3)
    if injStokes: axs[1,1].plot(ff[plot_range], eps2_inj[plot_range], color=injcolor, linewidth=0.5)
    axs[1,1].plot(ff[plot_range], eps2[plot_range], color=scolor, linewidth=0.5)
    axs[1,1].set_xscale('log')
    axs[1,1].set_xlim((flow,srate/2))
    axs[1,1].set_xlabel(r'$f$ [Hz]')
    axs[1,1].set_ylabel(r'$(I-Q)/V \ (\varepsilon)$')

    fig.tight_layout()
    fig.savefig(plotsDir+'stokes_combinations.png', dpi=300)

    #zoom-in plots
    axs[0,1].set_yscale('linear')
    axs[0,1].set_ylim((0,10.0))

    axs[1,0].set_ylim((-3.0,3.0))

    axs[1,1].set_ylim((-10.0,10.0))

    fig.savefig(plotsDir+'stokes_combinations_zoomin.png', dpi=300)


# -------------------------
# Plot dimensions of model
# -------------------------

def plot_model_dims(modelList, ifoList, ifoNames, plotsDir, runFlags = RUN_FLAGS):
    lineStyles = ['-', '--', ':','-','--',':']
    lineColors = ifoColors
    glitchChains = []
    signalChains = []
    # -- Read in data
    if runFlags.fullOnly_flag==1:
        intChainFile = 'chains/'+str(jobName)+"full_model.dat.0"
        N = len(open(intChainFile).readlines())
        chains = np.loadtxt(intChainFile,skiprows=int(N/2))
        chains = np.transpose(chains)
        glitchChains = chains[3:3+len(ifoList)]
        signalChains = chains[2]
    # means we have a cbc run 
    elif runFlags.CBCType:
        intChainFile = 'chains/'+'cbc_model.dat.0'
        N = len(open(intChainFile).readlines())
        chains = np.loadtxt(intChainFile,skiprows=int(N/2))
        chains = np.transpose(chains)
        glitchChains = chains[3:3+len(ifoList)]
    else:
        for mod in modelList:
            intChainFile = 'chains/'+str(jobName)+"{0}_model.dat.0".format(mod)
            N = len(open(intChainFile).readlines())
            chains = np.loadtxt(intChainFile,skiprows=int(N/2))
            chains = np.transpose(chains)
            if mod == 'glitch':
                glitchChains = chains[3:3+len(ifoList)]
            elif mod == 'signal':
                signalChains = chains[2]

    signalOnly_flag = 0
    glitchOnly_flag = 0
    
    if len(signalChains) == 0:
        signalOnly_flag == 1
    if len(glitchChains) == 0:
        glitchOnly_flag == 1

    # -- Variation of the model dimension over MCMC iterations. 2 subplots for signal and glitch models 
    # if glitch and not signal or signal and not glitch 
    if ('glitch' in modelList) != ('signal' in modelList):
        fig = plt.figure()
        ax1 = plt.subplot(111)
        ax1.set_title('Model Dimension')
        if len(signalChains) > 0:
            ax1.plot(signalChains, linewidth=2, color=scolor, label='Signal')
        if len(glitchChains) > 0:
            for ifo in ifoList:
                ifoint=int(ifo)
                ax1.plot(glitchChains[ifoint], lineStyles[ifoint], color=lineColors[ifoint], linewidth=2, label=ifoNames[ifoint])
            
            # -- Legend placement outside the plot
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width*0.8, box.height])
            ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # -- Make subplots close to each other and hide x ticks for all but bottom plot
        #plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        plt.xlabel( 'MC step/10' )
        ax1.set_ylabel('{0} model'.format(modelList[0]))
        ax1.grid()
        plt.savefig(plotsDir+'model_dimensions.png')
        plt.close()
    elif ('glitch' in modelList) and ('signal' in modelList):
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
        ax1.set_title('Model Dimension')
        if len(signalChains) > 0:
            ax1.plot(signalChains, linewidth=2, color=scolor, label='Signal')
        if len(glitchChains) > 0:
            for ifo in ifoList:
                ifoint=int(ifo)
                ax2.plot(glitchChains[ifoint], lineStyles[ifoint], color=lineColors[ifoint], linewidth=2, label=ifoNames[ifoint])

            # -- Legend placement outside the plot
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width*0.8, box.height])
            box = ax2.get_position()
            ax2.set_position([box.x0, box.y0, box.width*0.8, box.height])
            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # -- Make subplots close to each other and hide x ticks for all but bottom plot
            fig.subplots_adjust(hspace=0.1, right=0.8)
            plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
            plt.xlabel( 'MC step/10' )
            ax1.set_ylabel('Signal Model')
            ax1.grid()
            ax2.set_ylabel('Glitch Model')
            ax2.grid()
            plt.savefig(plotsDir+'model_dimensions.png')
            plt.close()
    
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)

    ax1.set_title('Model Dimension Histogram')
    if len(signalChains) > 0:
        n,bins,patches = ax1.hist(signalChains, bins=np.arange(int(min(signalChains))-.5, int(max(signalChains))+1.5, 1), histtype='bar', color=scolor, log=False)
    if len(glitchChains) > 0:
        data = np.dstack(glitchChains)[0]
        n,bins,patches = ax2.hist(data, bins=np.arange(int(data.min())-.5, int(data.max())+1.5, 1), label=ifoNames, histtype='bar', log=False, color=[lineColors[int(i)] for i in ifoList])
        # -- Legend placement outside the plot
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width*0.8, box.height])
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # -- Make subplots close to each other and hide x ticks for all but bottom plot
        fig.subplots_adjust(hspace=0.1, right=0.8)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        plt.xlabel('Model Dimension')
        ax1.set_ylabel('Signal Model')
        ax1.grid()
        ax2.set_ylabel('Glitch Model')
        ax2.grid()
        plt.savefig(plotsDir+'model_dimensions_histo.png')
    
    elif len(modelList) == 1:
        # -- Histogram of the model dimension. 2 subplots for signal and glitch models
        fig = plt.figure()
        ax1 = plt.subplot(111)
        ax1.set_title('Model Dimension Histogram')
        if len(signalChains) > 0:
            n,bins,patches = ax1.hist(signalChains, bins=np.arange(int(min(signalChains))-.5, int(max(signalChains))+1.5, 1), histtype='bar', alpha = 0.7, edgecolor='k', color=scolor, log=False)
        if len(glitchChains) > 0:
            data = np.dstack(glitchChains)[0]
            #ax2.set_prop_cycle(cycler('color',['darkgoldenrod','darkkhaki','dkarsage']))
            n,bins,patches = ax1.hist(data, bins=np.arange(int(data.min())-.5, int(data.max())+1.5, 1), label=ifoNames, histtype='bar', alpha=0.7, edgecolor='k', log=False, color=[lineColors[int(i)] for i in ifoList])
        # -- Legend placement outside the plot
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # -- Make subplots close to each other and hide x ticks for all but bottom plot
        #plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        plt.xlabel('Model Dimension')
        ax1.set_ylabel('{0} model'.format(modelList[0]))
        ax1.grid()
        plt.savefig(plotsDir+'model_dimensions_histo.png')

    # -- Calculate bayes factors for signal+glitch vs signal only, glitch only
    if runFlags.fullOnly_flag:
        signal_on = signalChains > 0
        glitch_on = np.sum(glitchChains[0:len(ifoList)+1],axis=0)>0
        
        ev_sig = 0
        ev_glitch = 0
        ev_full = 0
        for s, g in zip(signal_on, glitch_on):
            if s and not g:
                ev_sig += 1
            if g and not s:
                ev_glitch += 1
            if s and g:
                ev_full += 1
    else:
        ev_sig = 0
        ev_glitch = 0
        ev_full = 0

    # Actually full vs. signal only
    try:
        sig_noise = float(ev_full)/float(ev_sig)
    except:
        sig_noise = np.inf

    # Actually full vs. glitch only
    try:
        sig_gl = float(ev_full)/float(ev_glitch)
    except:
        sig_gl = np.inf
        
    if runFlags.fullOnly_flag:
        print("\nNumber of iterations in signal only, glitch only, and signal+glitch modes:\nNsig = {0}, Ngl = {1}, Nboth = {2}\n".format(ev_sig,ev_glitch,ev_full))

    return sig_noise, sig_gl

#--------------------
# plot cleaning phase
#--------------------
def plot_cleaning_phase_time_domain(runFlags, plotsDir='plots/'):
    
    if not runFlags.noClean:
        print('making cleaning phase plot(s)')

        #find window ends
        window_start = runFlags.gps - runFlags.segment_start + runFlags.window / 2
        window_end = runFlags.gps - runFlags.segment_start - runFlags.window / 2
        
        color = set_color(runFlags.run_dirname())
        cleancolor = set_color('clean')
 
        for ifo in ifoNames:
            name = 'cleaning_phase_time_domain_plot_{0}'.format(ifo)
            f, ax = plt.subplots(dpi=200)        
            #get glitches found in cleaning phase & those found in run
            clean = get_waveform('post/clean/glitch_median_time_domain_waveform_{0}.dat'.format(ifo))
            #print('DEBUGGING')
            #file=open('/home/fsarandrea/git/BayesWave/testdir_17/test.txt','w+')
            #from tempfile import TemporaryFile
            #np.savetxt('/home/fsarandrea/git/BayesWave/testdir_17/x_test.dat',clean[0])
            #np.savetxt('/home/fsarandrea/git/BayesWave/testdir_17/y_test.dat',clean[1])
            #file.close()
            #from sys import exit
            #exit()

            if runFlags.cleanOnly_flag:
                run = clean
            else:
                run = get_waveform('post/{1}/{1}_median_time_domain_waveform_{0}.dat'.format(ifo, runFlags.run_dirname()))


            if runFlags.mdc:
                # plot injected signal 
                injected = np.genfromtxt('post/injected_whitened_waveform_{ifo}.dat'.format(ifo = ifo))
                ax.plot(clean[0], injected ,color = set_color('injected'), label = 'injected', alpha=0.7,zorder=2,linewidth=1)
            #plot cleaning phase waveform 
            ax.plot(clean[0], clean[1],color = cleancolor, label = 'clean', alpha=0.7,zorder=10,linewidth=1)
            ax.fill_between(clean[0], clean[4], clean[5],color = cleancolor, alpha=0.3,edgecolor=None,zorder=8)
            #plot run waveform
            ax.plot(run[0], run[1],color = color, label = 'run',linewidth=1)
            ax.fill_between(run[0], run[4], run[5],color = color, alpha=0.3,edgecolor=None)
            
            #plot window
            ax.axvspan(window_start, window_end, color='gray',alpha=0.2,zorder=1,label='window')
            
            #other plot elements
            ax.legend(fontsize=6); ax.set_xlim(0, runFlags.seglen)
            ax.set_xlabel('Time (s)'); ax.set_ylabel('Whitened Strain')
            ax.set_title(runFlags.run_dirname()+', '+ifo)
            plt.savefig(plotsDir+name); plt.close()
            
            

def plot_cleaning_phase_frequency_domain(runFlags, plotsDir='plots/'):
    
    if not runFlags.noClean:
        color = set_color(runFlags.run_dirname()) 
        for ifo in ifoNames:
            name = 'cleaning_phase_frequency_domain_plot_{0}'.format(ifo)
            f, ax = plt.subplots(dpi=200)     

            #get glitches found in cleaning phase & those found in run
            clean = get_waveform('post/clean/glitch_median_frequency_domain_waveform_spectrum_{0}.dat'.format(ifo))
            if not runFlags.cleanOnly_flag:
                run = get_waveform('post/{1}/{1}_median_frequency_domain_waveform_spectrum_{0}.dat'.format(ifo, runFlags.run_dirname()))
            else:
                run = clean
            
            #get psd in cleaning phase and full run
            clean_psd = get_waveform('post/clean/glitch_median_PSD_{0}.dat'.format(ifo))
            if not runFlags.cleanOnly_flag:
                run_psd = get_waveform('post/{1}/{1}_median_PSD_{0}.dat'.format(ifo, runFlags.run_dirname()))
            else:
                run_psd = clean_psd
            
            #plot cleaning phase waveform 
            ax.plot(clean[0], clean[1],color = set_color('clean'), label = 'clean', alpha=0.7,zorder=10,linewidth=1)
            ax.fill_between(clean[0], clean[4], clean[5],color = set_color('clean'), alpha=0.3,edgecolor=None,zorder=8)

            #plot run waveform
            ax.plot(run[0], run[1],color = color, label = 'run',linewidth=1)
            ax.fill_between(run[0], run[4], run[5],color = color, alpha=0.3,edgecolor=None)

            #plot PSDs
            ax.plot(clean_psd[0],clean_psd[1],color='darkred',linewidth=1,label='clean psd')
            ax.plot(run_psd[0],run_psd[1],color='k',linewidth=1,label='run psd')

            #other plot elements
            ax.legend(fontsize=6,ncol=2)
            ax.set_xlim(runFlags.flow, runFlags.srate/2); ax.set_ylim(1e-50,1e-42)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Power')
            ax.set_title(ifo)
            plt.savefig(plotsDir+name); plt.close()

######################################################################################################################
# BWB webpage production
#
######################################################################################################################
def whitened_residual_plots(model,ifoList,ifoNames,runFlags = RUN_FLAGS):
    
    if model == 'glitch' or model == 'clean':
        lineColors = ifoColors
    elif ((mod == 'signal') or (mod == 'cbc')):
        lineColors = signal_ifoColorList
    elif model == 'noise':
        colour = ncolor
        lineColors = ifoColors
    else:
        lineColors = ifoColors
    
    hashlist = ['solid','dashed','dashdot','solid','dashed','dashdot']

    plt.clf()

    # -- Plot N[0,1]
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(-4, 4, 100)
    cdf = [1.0-0.5*(1.+math.erf(xx/math.sqrt(2.))) for xx in x]
    
    residual = {}
    whitened = {}
    psd_info = {}

    local_runType, local_model = runType_and_model(model, runFlags = RUN_FLAGS)
    for ifo in ifoList:
        filename = 'post/{runType}/{model}_median_PSD_{ifo}.dat'.format(runType = local_runType, model = local_model, ifo = ifoNames[int(ifo)])
        print("whitened_residual_plots filename 1: " + filename)
        psd_info[ifo] = get_waveform(filename)
            
        imin = 0
        while psd_info[ifo][1][imin] >= 1.0:
            imin += 1

        filename = 'post/{runType}/fourier_domain_{model}_median_residual_{ifo}.dat'.format(runType = local_runType, model = local_model, ifo = ifoNames[int(ifo)])
        print("whitened_residual_plots filename 2: " + filename)
        residual[ifo] = np.genfromtxt(filename)
        
        whitened[ifo] = []
        for i in range(0,len(psd_info[ifo][1])):
            whitened[ifo].append(residual[ifo][i,1]/math.sqrt(psd_info[ifo][1][i]/2.))

    
    #### --- Histograms
    plt.plot(x,scs.norm.pdf(x, mu, sigma),color='k',lw=1.4,label='$\mathcal{N}[0,1]$')
    for ifo in ifoList:
        plt.hist(whitened[ifo][imin::],bins=50,density=True,color=lineColors[int(ifo)],label=ifoNames[int(ifo)],histtype='step')

    plt.title(model)
    plt.yscale('log')
    plt.ylim(10e-4,2)
    plt.legend(loc=1)
    plt.grid(True)
    
    plt.savefig('plots/{model}_whitened_residual_histograms.png'.format(model=model))

    #plt.show()
    plt.clf()

    #### --- CDFs
    plt.plot(x,cdf,color='k',lw=1.4,label='$\mathcal{N}[0,1]$')
    for ifo in ifoList:
        
        plt.hist(whitened[ifo][imin::],bins=50,density=True,color=lineColors[int(ifo)],label=ifoNames[int(ifo)],histtype='step',cumulative=-1)

    plt.title(model)
    plt.yscale('log')
    plt.ylim(10e-4,2)
    plt.legend(loc=1)
    plt.grid(True)

    plt.savefig('plots/{model}_whitened_residual_cdfs.png'.format(model=model))

    #plt.show()
    plt.clf()

def make_verbose_page(htmlDir, plotsDir, runFlags = RUN_FLAGS):
    subpage = open(htmlDir+'/verbose.html', 'w+')
    # -- Start of html subpage
    subpage.write('<html>\n')
    subpage.write('<head>\n')
    subpage.write('</head>\n')
    subpage.write('<body>\n')

    subpage.write('    <h2>Verbose Output</h2>\n')


    #for ifo in ifoList:
    #      plotsrc = './{plotsDir}{model}_frequence_domain_{ifo}.png'.format(model=model,plotsDir=plotsDir,ifo=ifoNames[int(ifo)])
    #      subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')

    if 'glitch' in runFlags.modelList:
        subpage.write('    <h3>Verbose Output for glitch model</h2>\n')

        # put in tf plots for glitch  
        for ifo in ifoNames:
            plotsrc = './' +  plotsDir + '{model}{ifo}_{param1}_{param2}_2dscatter_chains_{chainmin}_{chainmax}.png'.format(param1 = 't', 
                    param2 = 'f', model = 'glitch', chainmin = 0, chainmax = runFlags.Nchain, ifo = ifo) 
            subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a><br/>\n')
        # put in QA plots for glitch  
        for ifo in ifoNames:
            plotsrc = './' +  plotsDir + '{model}{ifo}_{param1}_{param2}_2dscatter_chains_{chainmin}_{chainmax}.png'.format(param1 = 'Q', 
                    param2 = 'logA', model = 'glitch', chainmin = 0, chainmax = runFlags.Nchain, ifo = ifo) 
            subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a><br/>\n')

    if 'cbc' in runFlags.modelList:
        subpage.write('    <h3>Verbose Output for cbc model</h2>\n')
        plotsrc = './' + plotsDir + '{0}_{1}_2dscatter.png'.format('ra', 'sin_dec')
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a><br/>\n')

        plotsrc = './' + plotsDir + '{0}_{1}_2dscatter.png'.format('distance', 'cos_iota')
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a><br/>\n')

        plotsrc = './' + plotsDir + '{0}_{1}_2dscatter.png'.format('mchirp', 'chi_eff')
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a><br/>\n')

        plotsrc = './' + plotsDir + '{0}_{1}_2dscatter.png'.format('spin1', 'spin2')
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a><br/>\n')

        plotsrc = './' + plotsDir + '{0}_{1}_2dscatter.png'.format('chi_eff', 'spin1')
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a><br/>\n')

        plotsrc = './' + plotsDir + '{0}_{1}_2dscatter.png'.format('chi_eff', 'spin2')
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a><br/>\n')

            
    subpage.write('  </div>\n')
    # -- End of html nubpage
    subpage.write('</body>\n')
    subpage.write('</html>\n')
    subpage.close()


def make_skymap_page(htmlDir, plotsDir, runFlags = RUN_FLAGS):
    #if not os.path.exists(plotsDir):
    #if 'waveform' in page: 
    #    subpage = open(htmlDir+'waveform.html', 'w')
    #else:
    #    subpage = open(htmlDir+html_dict[page][0]+'.html', 'w')
    subpage = open(htmlDir+'/skymap.html', 'w')
    # -- Start of html subpage
    subpage.write('<html>\n')
    subpage.write('<head>\n')
    subpage.write('</head>\n')
    subpage.write('<body>\n')

    subpage.write('    <h2>Sky map</h2>\n')

    # Display the skymap
    plotsrc = './'+plotsDir+'skymap.png'
    if not os.path.exists(plotsrc):
        subpage.write(' <h3>Sky map generation failed.</h3>\n')
    subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=650></a><br/>\n')

    if runFlags.CBCType:
        for chain in range(runFlags.Nchain):
            # Only make the 0th skymap unless verbose is on 
            if runFlags.verbose or chain == 0:
                plotsrc = './' + plotsDir + 'cbc_skymap_{0}.png'.format(chain)
                subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=650></a><br/>\n')

    # Display the other megaksy plots, if present
    plotsrc = './'+plotsDir+'ra_acf.png'
    if os.path.exists(plotsrc):
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    plotsrc = './'+plotsDir+'ra_chain.png'
    if os.path.exists(plotsrc):
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')

    subpage.write('  </div>\n')
    # -- End of html nubpage
    subpage.write('</body>\n')
    subpage.write('</html>\n')
    subpage.close()


def make_stokes_page(htmlDir, plotsDir):
    #if not os.path.exists(plotsDir):
    #if 'waveform' in page: 
    #    subpage = open(htmlDir+'waveform.html', 'w')
    #else:
    #    subpage = open(htmlDir+html_dict[page][0]+'.html', 'w')
    subpage = open(htmlDir+'/stokes.html', 'w')
    # -- Start of html subpage
    subpage.write('<html>\n')
    subpage.write('<head>\n')
    subpage.write('</head>\n')
    subpage.write('<body>\n')

    subpage.write('    <h2>Stokes parameters</h2>\n')

    # Display the useful combinations of stokes parameters
    plotsrc = './'+plotsDir+'stokes_combinations_zoomin.png'
    if not os.path.exists(plotsrc):
        subpage.write(' <h3>Stokes calculation failed.</h3>\n')
    subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=1000></a>\n')
    # -- Toggle for full plots
    subpage.write('<br/>See full axis: <a id="displayFullstokes" href="javascript:toggle(\'divFullstokes\',\'displayFullstokes\');">show</a>\n')
    subpage.write('  <div id="divFullstokes" style="display: none">\n')
    plotsrc = './'+plotsDir+'stokes_combinations.png'
    subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=1000></a>\n')
    subpage.write('  </div>\n')
    # -- End of html nubpage
    subpage.write('</body>\n')
    subpage.write('</html>\n')
    subpage.close()


def splash_page(plotsDir, ifoNames, snrList, bayeswaverunfile, sig_gl, sig_noise, postDir, runFlags = RUN_FLAGS):
    html_string = ''
    if not (runFlags.fullOnly_flag or runFlags.GlitchCBC_flag or runFlags.SignalCBC_flag):
        html_string += '    <a href="./'+plotsDir+'odds.png"><img src="./'+plotsDir+'odds.png" style="float: right;" width=600 alt="odds.png"></a>\n'
    elif runFlags.fullOnly_flag:
        html_string += '    <h3>This run was done with the --fullOnly flag, meaning glitch and signal were run simultaneously.</h3>\n'
    elif runFlags.GlitchCBC_flag:
        html_string += '    <h3>This run was done with the --GlitchCBC flag, meaning glitch and CBC were run simultaneously.</h3>\n'
    elif runFlags.SignalCBC_flag:
        html_string += '    <h3>This run was done with the --SignalCBC flag, meaning signal and CBC were run simultaneously.</h3>\n'
    html_string += '    <h3>Detector Names: {0}</h3>\n'.format(', '.join(ifoNames))
    html_string += '    <h3>Models used:</h3>\n'
    html_string += '        <b>Simultaneous signal+glitch fitting:</b> '
    if runFlags.fullOnly_flag:
        html_string += 'On</br>\n'
    else:
        html_string += 'Off</br>\n'

    html_string += '        <b>Signal model:</b> '
    if 'signal' in runFlags.modelList:
        html_string += 'On</br>\n'
    else:
        html_string += 'Off</br>\n'
    html_string += '        <b>Glitch model:</b> '
    if 'glitch' in runFlags.modelList:
        html_string += 'On</br>\n'
    else:
        html_string += 'Off</br>\n'
    html_string += '        <b>Noise model:</b> '
    if runFlags.noNoiseFlag or runFlags.fullOnly_flag:
        html_string += 'Off</br>\n'
    else:
        html_string += 'On</br>\n'
    if mdc:
        html_string += '    <h3>Matched Filter SNRs of Injections</h3>\n'
        for ifo, snr in zip(ifoNames, snrList):
            html_string += '    {0} injected with SNR {1:.1f}<br/>\n'.format(ifo, snr)
    html_string += '    <h3>Log Info</h3>\n'
    html_string += '    <a href="./'+bayeswaverunfile+'">See full log file</a>\n'
    if len(runFlags.modelList) == 1 and runFlags.modelList != ['clean']:
        html_string +='    <h3>This is a {0} only run.</h3>\n'.format(runFlags.modelList[0])
        html_string +='    There are no Bayes factors for this run since it\'s only a single model.\n'
    # elif runFlags.fullOnly_flag:
    #     html_string +='    <h3>Bayes factor for signal+glitch vs signal only:</h3>\n'
    #     html_string +='    {0} <br/>\n'.format(sig_noise)

    #     html_string +='    <h3>Bayes factor for signal+glitch vs glitch only:</h3>\n'
    #     html_string +='    {0} <br/>\n'.format(sig_gl)
    elif runFlags.GlitchCBC_flag:
        html_string +='    <h3>Bayes factor for cbc+glitch vs cbc only:</h3>\n'
        html_string +='    {0} <br/>\n'.format(sig_noise)

        html_string +='    <h3>Bayes factor for cbc+glitch vs glitch only:</h3>\n'
        html_string +='    {0} <br/>\n'.format(sig_gl)

    elif runFlags.SignalCBC_flag:
        html_string +='    <h3>Bayes factor for cbc+signal vs cbc only:</h3>\n'
        html_string +='    {0} <br/>\n'.format(sig_noise)

        html_string +='    <h3>Bayes factor for cbc+signal vs signal only:</h3>\n'
        html_string +='    {0} <br/>\n'.format(sig_gl)


    html_string +='    <h3>Evidence for Signal</h3>\n'
    if not runFlags.fullOnly_flag:
        if 'glitch' not in runFlags.modelList:
            html_string += '    log(Evidence_signal / Evidence_glitch) = N/A (no glitch model)<br/>\n'
        elif 'signal' not in runFlags.modelList:
            html_string += '    log(Evidence_signal / Evidence_glitch) = N/A (no signal model)<br/>\n'
        else:
            html_string +='    log(Evidence_signal / Evidence_glitch) = {0:.1f} &plusmn; {1:.1f} <br/>\n'.format(sig_gl,err_sig_gl)
        # if there is a noise flag 
        if not runFlags.noNoiseFlag:
            html_string +='    log(Evidence_signal / Evidence_noise) = {0:.1f} &plusmn; {1:.1f} <br/>\n'.format(sig_noise,err_sig_noise)
        else:
            html_string +='    log(Evidence_signal / Evidence_noise) = N/A (no noise model)\n'
    else:
        html_string +='    <b>log(Bayes factor for signal+glitch vs signal only):</b>\n'
        html_string +='    {0} <br/>\n'.format(np.log(sig_noise))

        html_string +='    <b>log(Bayes factor for signal+glitch vs glitch only):</b>\n'
        html_string +='    {0} <br/>\n'.format(np.log(sig_gl))
    html_string +='  </p>\n'

    return html_string


def make_homepage(htmlDir, plotsDir, model, ifoNames, snrList, bayeswaverunfile, sig_gl, sig_noise, splash_page_text, runFlags = RUN_FLAGS):
    summary = open(htmlDir+'summary.html', 'w')
    # -- Start of html page with basic info about the run
    summary.write('<html>\n')
    summary.write('<head>\n')
    summary.write('</head>\n')
    summary.write('<body>\n')
    summary.write('  <p>\n')
    summary.write('\n')
    # -- Odds plot
    summary.write(splash_page_text)
    # -- End of html page with basic info about the run
    summary.write('</body>\n')
    summary.write('</html>\n')
    summary.close()



def make_index(htmlDir, plotsDir, model, gps, ifoList, ifoNames, snrList, bayeswaverunfile, sig_gl, sig_noise, splash_page_text, runFlags = RUN_FLAGS):
    index = open('index.html', 'w')
    # -- Start of the BWB webpage
    index.write('<!DOCTYPE html/css>\n')
    index.write('<html>\n')
    index.write('<head>\n')
    index.write('<link rel="stylesheet" type="text/css" href="./html/BWBweb.css">\n')
    index.write('<!--<script src="http://ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js"></script>--!>\n')
    index.write('<script src="./html/secure_ajax.js"></script>\n')
    index.write('<script src="./html/navigate.js"></script>\n')
    index.write('</head>\n')
    # -- Title
    index.write('<body>\n')
    index.write('<div class="container wrapper">\n')
    index.write('  <div id="top">\n')
    index.write('    <h1>BayesWave Output Page</h1>\n')
    index.write('       <p>Results for trigger at GPS {0}</p>\n'.format(gps))
    index.write('  </div>\n')
    # -- Navigation bar
    index.write('  <div class="wrapper">\n')
    index.write('    <div id="menubar">\n')
    index.write('      <ul id="menulist">\n')
    index.write('        <li class="menuitem" id="summary">Model selection\n')

    # make links to model webpages with timeseries and the like
    for mod in runFlags.modelList:
        if mod == 'clean':
            continue
        index.write('        <li class="menuitem" id="{0}">{0} model\n'.format(mod))
    if runFlags.multi_type:
        index.write('        <li class="menuitem" id="{0}">{1} + {2} model\n'.format(runFlags.run_dirname(), 
                                                                                runFlags.modelList[0], runFlags.modelList[1]))
    for mod in runFlags.modelList:
        if mod == 'clean': 
            continue
        index.write('        <li class="menuitem" id="{0}moments">{0} waveform moments\n'.format(mod))
    if runFlags.CBCType:
        index.write('        <li class="menuitem" id="cbcparams">CBC Parameters\n'.format(mod))

    if not runFlags.noClean:
        index.write('        <li class="menuitem" id="{0}">{0} model\n'.format('clean'))
        index.write('        <li class="menuitem" id="{0}moments">{0} waveform moments\n'.format('clean'))

    if runFlags.noPolarization_flag:
        index.write('        <li class="menuitem" id="stokes">Stokes parameters\n')
    if 'signal' in runFlags.modelList:
        index.write('        <li class="menuitem" id="skymap">Skymap\n')
    if mdc:
        index.write('        <li class="menuitem" id="overlap">Overlap\n')

    index.write('        <li class="menuitem" id="diagnostics">Diagnostics\n')
    if runFlags.verbose:
        index.write('        <li class="menuitem" id="verbose">Verbose Output\n'.format(mod))
    
    if not runFlags.noClean:
        index.write('        <li class="menuitem" id="cleaning">Cleaning Phase\n')
    index.write('      </ul>\n')
    index.write('    </div>\n')
    # -- Main part
    index.write('    <div id="main">\n')
    index.write('      <p>\n')
    index.write('\n')
    index.write(splash_page_text)
    index.write('    </div>\n')

    # -- End of the BWB webpage
    index.write('</body>\n')
    index.write('</html>\n')
    index.close()



def make_model_page(htmlDir, plotsDir, model, ifoList, ifoNames, runFlags = RUN_FLAGS):
    subpage = open(htmlDir+model+'.html','w') # make html file

    # -- Start of html subpage
    subpage.write('<html>\n')
    subpage.write('<head>\n')
    subpage.write('</head>\n')
    subpage.write('<body>\n')
    
    # To Plot:
    #   - time domain waveform
    #   - power spectra
    #   - tf tracks
    #   - Q scan
    #   - (if applicable) injeciton Q scan
    #   - Residual Q scan

    subpage.write('     <h2>{0} model</h2>'.format(model))

    ######### Time domain waveform #######
    subpage.write('         <center><h3>Time Domain Waveforms</h3></center>\n')
    for ifo in ifoList:
        plotsrc = './{plotsDir}{model}_waveform_{ifo}.png'.format(model=model,plotsDir=plotsDir,ifo=ifoNames[int(ifo)])
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    # -- Toggle for full plots
    subpage.write('<br/>See full axis: <a id="displayFull'+model+'timedomain" href="javascript:toggle(\'divFull'+model+'timedomain\',\'displayFull'+model+'timedomain\');">show</a>\n')
    subpage.write('  <div id="divFull'+model+'timedomain" style="display: none">\n')
    for ifo in ifoList:
        plotsrc = './{plotsDir}{model}_waveform_{ifo}_full.png'.format(model=model,plotsDir=plotsDir,ifo=ifoNames[int(ifo)])
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    subpage.write("</div>\n")
    subpage.write('  <br/>\n')
    subpage.write('  <hr>\n')

    ######### Frequency domain waveform #######
    subpage.write('         <center><h3>Power Spectra</h3></center>\n')
    for ifo in ifoList:
        plotsrc = './{plotsDir}{model}_frequence_domain_{ifo}.png'.format(model=model,plotsDir=plotsDir,ifo=ifoNames[int(ifo)])
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    subpage.write('  <br/>\n')
    subpage.write('  <hr>\n')


    # only add tf tracks to non-combined pages:
    if not model in ('full', 'cbcglitch', 'cbcsignal'):


      ######### f(t) tracks #######
      subpage.write('         <center><h3>Frequency vs. Time</h3></center>\n')
      for ifo in ifoList:
          plotsrc = './{plotsDir}{model}_tf_{ifo}.png'.format(model=model,plotsDir=plotsDir,ifo=ifoNames[int(ifo)])
          subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
      subpage.write('  <br/>\n')
      subpage.write('  <hr>\n')

    ######### Spectrogram #######
    subpage.write('         <center><h3>Spectrogram of median waveform</h3></center>\n')
    for ifo in ifoList:
        plotsrc = './'+plotsDir+'{0}_spectrogram_Q8_{1}.png'.format(model,ifoNames[int(ifo)])
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    # -- Toggle for full plots
    subpage.write('<br/>Other Q resolutions: <a id="displayFull'+model+'spect\'" href="javascript:toggle(\'divFull'+model+'spect\',\'displayFull'+model+'spect\');">show</a>\n')
    subpage.write('  <div id="divFull'+model+'spect'+'" style="display: none">\n')
    for ifo in ifoList:
        plotsrc = './'+plotsDir+'{0}_spectrogram_Q4_{1}.png'.format(model,ifoNames[int(ifo)])
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    for ifo in ifoList:
        plotsrc = './'+plotsDir+'{0}_spectrogram_Q16_{1}.png'.format(model,ifoNames[int(ifo)])
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    subpage.write("</div>\n")
    subpage.write('  <br/>\n')
    subpage.write('  <hr>\n')

    ######### Data #######
    subpage.write('         <center><h3>Spectrogram of Data</h3></center>\n')
    for ifo in ifoList:
        plotsrc = './'+plotsDir+'data_spectrogram_Q8_{0}.png'.format(ifoNames[int(ifo)])
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    # -- Toggle for full plots
    subpage.write('<br/>Other Q resolutions: <a id="displayFull'+model+'data\'" href="javascript:toggle(\'divFull'+model+'data\',\'displayFull'+model+'data\');">show</a>\n')
    subpage.write('  <div id="divFull'+model+'data'+'" style="display: none">\n')
    for ifo in ifoList:
        plotsrc = './'+plotsDir+'data_spectrogram_Q4_{0}.png'.format(ifoNames[int(ifo)])
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    for ifo in ifoList:
        plotsrc = './'+plotsDir+'data_spectrogram_Q16_{0}.png'.format(ifoNames[int(ifo)])
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    subpage.write("</div>\n")
    subpage.write('  <br/>\n')
    subpage.write('  <hr>\n')


  ######### Residuals #######
    subpage.write('         <center><h3>Spectrogram of Residuals</h3></center>\n')
    for ifo in ifoList:
        plotsrc = './'+plotsDir+'{0}_residual_spectrogram_Q8_{1}.png'.format(model,ifoNames[int(ifo)])
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    # -- Toggle for full plots
    subpage.write('<br/>Other Q resolutions: <a id="displayFull'+model+'res\'" href="javascript:toggle(\'divFull'+model+'res\',\'displayFull'+model+'res\');">show</a>\n')
    subpage.write('  <div id="divFull'+model+'res'+'" style="display: none">\n')
    for ifo in ifoList:
        plotsrc = './'+plotsDir+'{0}_residual_spectrogram_Q4_{1}.png'.format(model,ifoNames[int(ifo)])
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    for ifo in ifoList:
        plotsrc = './'+plotsDir+'{0}_residual_spectrogram_Q16_{1}.png'.format(model,ifoNames[int(ifo)])
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    subpage.write("</div>\n")
    subpage.write('  <br/>\n')
    subpage.write('  <hr>\n')
    subpage.write('</body>\n')
    subpage.write('</html>\n')
    subpage.close()


def make_moments_page(htmlDir, plotsDir, model, ifoList, ifoNames, runFlags = RUN_FLAGS):

    subpage = open(htmlDir+model+'moments.html','w')

    # -- Start of html subpage
    subpage.write('<html>\n')
    subpage.write('<head>\n')
    subpage.write('</head>\n')
    subpage.write('<body>\n')


    for page in ['t0_rec', 'dur_rec', 'f0_rec', 'band_rec', 'snr']:
        subpage.write('  <br/>\n')
        subpage.write('  <center><h2>{0}</h2></center>'.format(html_dict[page][1]))
        
        for ifo in ifoList:
            plotsrc = './'+plotsDir+model+'_'+html_dict[page][2]+'_{0}.png'.format(ifoNames[int(ifo)])
            subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
        subpage.write('    </br>\n')


    subpage.write('  <br/>\n')
    subpage.write('  <center><h2>Cornerplot {0}</h2></center>'.format(model))

    if model == 'glitch':
        for ifo in ifoList:
            plotsrc = './'+plotsDir +  model + '_cornerplot_{0}.png'.format(ifoNames[int(ifo)])
            subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
        subpage.write('    </br>\n')
    elif model == 'cbc':
        plotsrc = './'+plotsDir +  model + '_cornerplot_0.png'.format(ifoNames[int(ifo)])
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
        subpage.write('    </br>\n')
    elif model == 'signal':
        plotsrc = './'+plotsDir +  model + '_cornerplot_extrinsic.png'
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
        plotsrc = './'+plotsDir +  model + '_cornerplot_wavelets.png'
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
        subpage.write('    </br>\n')


    subpage.write('</body>\n')
    subpage.write('</html>\n')
    subpage.close()

def make_cbc_params_page(htmlDir, plotsDir, runFlags = RUN_FLAGS):

    if not runFlags.CBCType:
        return

    subpage = open(htmlDir + 'cbcparams.html','w')

    # -- Start of html subpage
    subpage.write('<html>\n')
    subpage.write('<head>\n')
    subpage.write('</head>\n')
    subpage.write('<body>\n')

    param_names = ['distance', 'Mc', 'Q', 'mass1', 'mass2', 'chi_eff', 'Z']
    subpage.write('  <br/>\n')
    subpage.write('  <center><h2>Corner Plot</h2></center>')
    if runFlags.verbose:
        chains = np.arange(0, runFlags.Nchain)
    else:
        chains = [0]
    for chain in chains:
        plotsrc = './' + plotsDir + "cbc_cornerplot_{0}.png".format(chain)
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=1000></a>\n')

    subpage.write('  <br/>\n')
    subpage.write('  <center><h2>Sky Location</h2></center>')
    for chain in chains:
        plotsrc = './' + plotsDir + "cbc_skymap_{0}.png".format(chain)
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    subpage.write('    </br>\n')


    for page in param_names:
        subpage.write('  <br/>\n')
        subpage.write('  <center><h2>{0}</h2></center>'.format(html_dict[page][1]))
        
        plotsrc = './' + plotsDir + "cbc_" + html_dict[page][2]+'_chain.png'
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')

        plotsrc = './' + plotsDir + "cbc_" + html_dict[page][2]+'_hist.png'
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')

        subpage.write('    </br>\n')



    subpage.write('</body>\n')
    subpage.write('</html>\n')
    subpage.close()

    return 

def make_cleaning_page(htmlDir, plotsDir, runFlags = RUN_FLAGS):
    subpage = open(htmlDir+'/cleaning.html', 'w+')
    subpage.write('<html>\n')
    subpage.write('<head>\n')
    subpage.write('</head>\n')
    subpage.write('<body>\n')

    subpage.write('    <h2>Cleaning Phase</h2>\n')

    subpage.write('         <center><h3>Time Domain</h3></center>\n') 
    for ifo in ifoNames:
        plotsrc = './' + plotsDir + 'cleaning_phase_time_domain_plot_{0}.png'.format(ifo)
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    subpage.write('  <br/>\n')
    for ifo in ifoNames:
        plotsrc = './' + plotsDir + 'cleaning_phase_frequency_domain_plot_{0}.png'.format(ifo)
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')   
 
    subpage.write('  </div>\n')
    subpage.write('</body>\n')
    subpage.write('</html>\n')
    subpage.close()
####################


def make_diagnostics_page(htmlDir, plotsDir, model, ifoList, ifoNames, modelList, runFlags = RUN_FLAGS):
    
    subpage = open(htmlDir+'diagnostics.html','w+')
    subpage.write('<html>\n')
    subpage.write('<head>\n')
    subpage.write('</head>\n')
    subpage.write('<body>\n')
    
    for plot in ['likelihood', 'model_dimensions_histo', 'model_dimensions']:
        plotsrc = './'+plotsDir+plot+'.png'
        subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
        if plot == 'likelihood':
            subpage.write('     </br>')


    ######## Glitch, signal and cbc cornerplot parameters ###### 
    subpage.write('         <center><h3>Posterior distribution of models</h3></center>\n')
    for mod in modelList:
        if mod == 'cbc':
            plotsrc = './' + plotsDir + "cbc_cornerplot_0.png"
            subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=1000></a>\n')
        elif mod == 'glitch':
            for ifo in ifoList:
                plotsrc = './' + plotsDir + "{0}_cornerplot_{1}.png".format(mod, ifoNames[int(ifo)])
                subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
        elif mod == 'signal':
                plotsrc = './' + plotsDir + "{0}_cornerplot_extrinsic.png".format(mod)
                subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
                plotsrc = './' + plotsDir + "{0}_cornerplot_wavelets.png".format(mod)
                subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')

    subpage.write('         <center><h3>PDFs and CDFs of whitened (fourier domain) residuals, compared to N[0,1]</h3></center>\n')
    if runFlags.multi_type:
        plotsrc = './'+plotsDir+'{runType}_whitened_residual_histograms.png'.format(runType = runFlags.run_dirname())
        if os.path.exists(plotsrc):
            subpage.write('      <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
            plotsrc = './'+plotsDir+'{runType}_whitened_residual_cdfs.png'.format(runType = runFlags.run_dirname())
            subpage.write('      <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')

    for mod in modelList:
        plotsrc = './'+plotsDir+'{0}_whitened_residual_histograms.png'.format(mod)
        if not os.path.exists(plotsrc):
            continue
        subpage.write('      <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
        plotsrc = './'+plotsDir+'{0}_whitened_residual_cdfs.png'.format(mod)
        subpage.write('      <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    #-- noise
    plotsrc = './'+plotsDir+'noise_whitened_residual_histograms.png'
    if os.path.exists(plotsrc):
        subpage.write('      <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
        plotsrc = './'+plotsDir+'noise_whitened_residual_cdfs.png'
        subpage.write('      <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
    subpage.write('     </br>')

    # put anderson darling statistic on page
    # only exists if running with bayesLine flag
    print("Are we going to try AD statistics? {0}".format(runFlags.bayesLine))
    if runFlags.bayesLine:
        print("Printing Anderson Darling Statistics")
        subpage.write('    <center><h3>Anderson Darling Statistics</h3></center>\n')
        if runFlags.multi_type:
            models = [runFlags.run_dirname()]
            for m in runFlags.modelList:
                models.append(m)
        else:
            models = runFlags.modelList
        for m in models:
            subpage.write('<table style="width:50%">\n')
            subpage.write('\t<tr> <th colspan="4">{0} Model</th></tr>\n'.format(m))
            subpage.write('\t<tr>\n')
            subpage.write('\t\t<th>IFO</th>\n')
            subpage.write('\t\t<th>Sample Rate [Hz]</th>\n')
            subpage.write('\t\t<th>AD statistic</th>\n')
            subpage.write('\t\t<th>P value</th>\n')
            subpage.write('\t</tr>\n')

            runType_local, model_local = runType_and_model(m, runFlags)
            for ifo in ifoNames:
                try:
                    AD_file = postDir + "{runType}/{model}_anderson_darling_{ifo}.dat".format(runType = runType_local, model = model_local, ifo = ifo)
                    sample_rate, AD, p_value = np.loadtxt(AD_file, unpack = True, ndmin=2)
                except:
                    print("Could not find Anderson-Darling file. Skipping this part.\n")
                    continue
                for j in range(len(AD)):
                    subpage.write('\t<tr>\n')
                    if j == 0:
                        subpage.write('\t\t<td rowspan="{1}">{0}</td>\n'.format(ifo, len(AD)))
                    subpage.write('\t\t<td>{0}</td>\n'.format(sample_rate[j]))
                    subpage.write('\t\t<td>{0}</td>\n'.format(AD[j]))
                    subpage.write('\t\t<td>{0}</td>\n'.format(p_value[j]))
                    subpage.write('\t</tr>\n')
            subpage.write('</table>\n')
    subpage.write('\n</br>')
    subpage.write('</body>\n')
    subpage.write('</html>\n')
    subpage.close()
    return

def runType_and_model(model, runFlags = RUN_FLAGS):
    # returns runtype and model
    if model == 'clean':
        runType = 'clean'
        model = 'glitch'
    elif runFlags.multi_type:
        runType = runFlags.run_dirname()
    else:
        runType = model

    return runType, model

def make_overlap_page(htmlDir, plotsDir, modelList, ifoList, ifoNames):

    subpage = open(htmlDir+'overlap.html','w')
    subpage.write('<html>\n')
    subpage.write('<head>\n')
    subpage.write('</head>\n')
    subpage.write('<body>\n')

    for mod in modelList:
        
        subpage.write('<center><h2>Overlap Histogram between recovered {0} and injection</h2></center>'.format(mod))
        for ifo in ifoList:
            plotsrc = './'+plotsDir+mod+'_overlap_{0}.png'.format(ifoNames[int(ifo)])
            subpage.write('    <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
        
        plotsrc = './'+plotsDir+mod+'_network_overlap.png'
        
        subpage.write('<center><h2>Network Overlap Histogram between recovered {0} and injection</h2></center>'.format(mod))
        subpage.write(' <center>')
        subpage.write('     <a href="'+plotsrc+'"><img src="'+plotsrc+'" alt="'+plotsrc+'" width=500></a>\n')
        subpage.write(' </center>')
        subpage.write('<hr>')

    subpage.write('</body>\n')
    subpage.write('</html>\n')
    subpage.close()


def make_webpage(htmlDir, model, mdc, gps, ifoList, ifoNames, modelList, snrList, sig_gl, sig_noise, postprocesspath, postDir, runFlags = RUN_FLAGS):
    # TODO: move in separate function
    # -- Find out the path to the BayesWave executable

    os.system('cp '+postprocesspath+'/BWBweb.css '+htmlDir+'.')
    os.system('cp '+postprocesspath+'/secure_ajax.js '+htmlDir+'.')
    os.system('cp '+postprocesspath+'/navigate.js '+htmlDir+'.')

    splash_page_text = splash_page(plotsDir, ifoNames, snrList, bayeswaverunfile, sig_gl, sig_noise, postDir, runFlags)

    # -- Write the index
    make_index(htmlDir, plotsDir, model, gps, ifoList, ifoNames, snrList, bayeswaverunfile, sig_gl, sig_noise, splash_page_text, runFlags)
    # -- Write summary page (works as a homepage)
    make_homepage(htmlDir, plotsDir, model, ifoNames, snrList, bayeswaverunfile, sig_gl, sig_noise, splash_page_text, runFlags)

    # -- Makes cbc-parameter page if we have a cbc model 
    make_cbc_params_page(htmlDir, plotsDir, runFlags)
    for mod in runFlags.modelList:
        make_model_page(htmlDir, plotsDir, mod, ifoList, ifoNames)
        make_moments_page(htmlDir, plotsDir, mod, ifoList, ifoNames)

    if runFlags.multi_type:
        make_model_page(htmlDir, plotsDir, runFlags.run_dirname(), ifoList, ifoNames)
    make_diagnostics_page(htmlDir, plotsDir, model, ifoList, ifoNames,modelList)
    if mdc:
        make_overlap_page(htmlDir, plotsDir, modelList, ifoList, ifoNames)
    if 'signal' in runFlags.modelList:
        make_skymap_page(htmlDir, plotsDir)
    if runFlags.noPolarization_flag:
        make_stokes_page(htmlDir, plotsDir)
    if not runFlags.noClean:
         make_cleaning_page(htmlDir, plotsDir)
    make_verbose_page(htmlDir, plotsDir, runFlags = RUN_FLAGS)




######################################################################################################################
#
# Main
#
######################################################################################################################
# -- Parse command line arguments
parser = argparse.ArgumentParser(description='Produce html page for a bayeswave run.')

# -- Get basic info on the run
jobName, mdc, injFlag, bayeswaverunfile, ifoList, ifoNames, gps, snrList = RUN_FLAGS.jobName, RUN_FLAGS.mdc, RUN_FLAGS.injFlag, RUN_FLAGS.bayeswaverunfile, RUN_FLAGS.ifoList, RUN_FLAGS.ifoNames, RUN_FLAGS.gps, RUN_FLAGS.snrList


# make this into a local filename
bayeswaverunfile = bayeswaverunfile.replace(RUN_FLAGS.trigdir + '/', "")
print("The mdc status is: {0}\n".format(mdc))
if injFlag:
    print("The injection was performed via an xml table\n")

# -- Create directory that will contain all plots
if not os.path.exists(plotsDir):
    os.makedirs(plotsDir)

# -- Plot Stokes parameters:
if RUN_FLAGS.noPolarization_flag:
        stokes_plots(postDir, plotsDir, RUN_FLAGS.flow, RUN_FLAGS.srate)

# -- Read in time vector
time = np.loadtxt(str(jobName)+postDir+'timesamp.dat')
freq = np.loadtxt(str(jobName)+postDir+'freqsamp.dat')

# -- Don't worry about printing the colored stuff any more, so set worc to whitened only
worc = 'whitened'

for mod in RUN_FLAGS.modelList:
    print("\nAnalyzing the {0} model".format(mod))
    
    # -- Loop over interferometers to create plots
    for ifo in ifoList:
        #print('skipping most plot creation, remove this later')
        #continue
        
        print('Processing {0} data'.format(ifoNames[int(ifo)]))
        
        # --------------------
        # Make waveform plots
        # --------------------

        # -- Determine axes for waveform plots
        axwinner = get_axes(jobName, postDir, ifoList, mod, time, RUN_FLAGS)

        # -- Read in the reconstructed time domain waveform
        local_runType, local_model = runType_and_model(mod, RUN_FLAGS)
        filename = str(jobName)+postDir+'{runType}/{model}_median_time_domain_waveform_{ifo}.dat'.format(ifo = ifoNames[int(ifo)], 
                model = local_model, runType = local_runType)
        timesamp,median_waveform, high_50, low_50, high_90, low_90 = get_waveform(filename)

        # -- Plot the median waveform and injected waveform (if present)
        #np.savetxt("/home/fsarandrea/git/BayesWave/testdir_17/outputDir2/post/test_median_waveform.dat", median_waveform)
        
        clean = get_waveform('post/clean/glitch_median_time_domain_waveform_{ifo}.dat'.format(ifo = ifoNames[int(ifo)]))
        #shifting the time axis so that the glitch is centered arounf the 0s point
        time_shift=float(event_time)-float(start_time)
        xvals=clean[0]-time_shift
        # identifying the x-coordinates of the glitch
        event_points=math.floor(1024*time_shift)
        yvals=clean[1]
        
        #selecting only 938 points of the time signal
        xvals=xvals[event_points-469:event_points+469]
        yvals=yvals[event_points-469:event_points+469]
        plt.figure()
        fig, ax = plt.subplots()
        plt.plot(xvals, yvals, linewidth=1, alpha=1)
        plt.savefig("test_plot.png")


        
