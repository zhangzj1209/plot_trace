import numpy as np
import matplotlib.pyplot as plt
import obspy
import os
from matplotlib.dates import date2num, DateFormatter

def plot_multi_trace_waveform_energy(path):
    '''
    Place each .sac file of the multi-station waveform in the separate file path, 
        and then plot the multi-station waveform & multi-station energy

    Parameters
    ----------
    path : file storage path

    Returns
    -------
    None.

    '''
    
    SECONDS_PER_DAY = 3600.0*24.0
    JET_LAG = 3600.0*8.0        # the difference between Beijing time & universal time
    files = os.listdir(path)    # make a list of all the files in path
    NUM_STATION = len(files)    # the number of stations
    for file in files:
        LENGTH_DATA = obspy.read(path + file)[0].data.shape[0]  # the length of a trace
        break
    DATA = np.empty((NUM_STATION, LENGTH_DATA)) # record the raw data
    for i, file in enumerate(files):
        DATA[i] = obspy.read(path + file)[0].data
    DATA_plot = DATA/abs(DATA.max())    # normalization
    
    fig_multitrace = plt.figure(figsize=(16, 8))
    ax = plt.subplot(111)
    tr = obspy.read(path + files[0])[0]
    time = np.arange(0, tr.stats.npts/tr.stats.sampling_rate, tr.stats.delta)
    x_values = ((tr.times()/SECONDS_PER_DAY) + date2num((tr.stats.starttime + JET_LAG).datetime))
    off = 2     # the distance of each trace for plotting 
    for i in range(DATA_plot.shape[0]):
        ax.plot(x_values, DATA_plot[i, :] + i*off, 'k', linewidth=0.5)
    formatter = DateFormatter('%y%m%d %H:%M:%S')
    ax.xaxis.set_major_formatter(formatter)
    yticks_serial = [x*off for x in range(NUM_STATION)]
    yticks_name = []
    for i, file in enumerate(files):
        yticks_name.append(file[14: 22])
    
    plt.title('The raw multi-station records', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Stations', fontsize=14)
    plt.tick_params(labelsize=12)
    plt.yticks(yticks_serial, size=12)
    plt.yticks(yticks_serial, yticks_name)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    fig_multitrace.savefig('./fig/Multi_station_waveform.png', dpi=600, bbox_inches='tight')
    
    Record_second = int(x_values.shape[0]/tr.stats.sampling_rate)
    ENERGY = np.empty((NUM_STATION, Record_second))
    Time_Window = 1     # the window length of self-stacking
    for i in range(NUM_STATION):
        for j in range(Record_second):
            ENERGY[i, j] = np.sum((DATA[i, round(j*tr.stats.sampling_rate): round((j + Time_Window)*tr.stats.sampling_rate)])**2)
    ENERGY_norm = ENERGY/ENERGY.max()
    
    fig_energy = plt.figure(figsize=(16, 8))
    ax = plt.subplot(111)
    cbar = ax.imshow(ENERGY_norm, extent=[x_values.min(), x_values.max(), 0, NUM_STATION], 
                     cmap='Greens', aspect='auto', interpolation='bicubic', origin='lower')
    yticks_serial_energy = [x for x in range(NUM_STATION)]
    plt.title('The multi-station self-stacking energy', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Stations', fontsize=14)
    plt.tick_params(labelsize=12)
    plt.yticks(yticks_serial_energy, size=12)
    plt.yticks(yticks_serial_energy, yticks_name)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    formatter = DateFormatter('%y%m%d %H:%M:%S')
    ax.xaxis.set_major_formatter(formatter)
    ax.figure.colorbar(cbar)
    fig_energy.savefig('./fig/Multi_station_energy.png', dpi=600, bbox_inches='tight')