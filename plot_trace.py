import numpy as np
import matplotlib.pyplot as plt
import scipy
import pywt

# In[]
def oneD_FFT(data, nt, sampling_fre=100):
    '''
    Fast Fourier transform

    Parameters
    ----------
    data : the input of 1-D data
    nt : the length of the signal (sec)
    sampling_fre : sampling frequency

    Returns
    -------
    freq1 : the x-axis after FFT, 0-sampling_fre/2 (Hz)
    fft_amp1 : the y-axis after FFT, amplitude infomation

    '''
    N = nt*sampling_fre   # sampling point
    data_fft = np.fft.fft(data)
    fft_amp0 = np.array(np.abs(data_fft)/N*2)   # calculate two-sided spectrum
    direct = fft_amp0[0]
    fft_amp0[0] = 0.5*direct
    N_2 = int(N/2)
    
    fft_amp1 = fft_amp0[0: N_2]     # calculate single-sided specturm
    list1 = np.array(range(0, N_2))
    freq1 = sampling_fre*list1/N    # the frequency axis of single-sided spectrum
    
    fft_amp1[0] = 0
    
    fig = plt.figure(figsize=(8, 4))
    plt.plot(freq1, fft_amp1, linewidth=1)
    plt.title('Single-sided spectrum', fontsize=16)
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    fig.savefig('./fig/FFT_single_sided_spectrum.png', dpi=600, bbox_inches='tight')
    return freq1, fft_amp1
    
    
# In[]
def oneD_filter(data, nt, fre_min, fre_max, filter_order, sampling_fre=100):
    '''
    1-D filter

    Parameters
    ----------
    data : the input of 1-D data
    nt : the length of the signal (sec)
    fre_min : the low-frequency cutoff of bandpass
    fre_max : the high-frequency cutoff of bandpass
    filter_order : the order of filter
    sampling_fre : sampling frequency

    Returns
    -------
    data_filter : filtered waveform

    '''
    b, a = scipy.signal.butter(filter_order, [2*fre_min/sampling_fre, 2*fre_max/sampling_fre], 'bandpass')
    data_filter = scipy.signal.filtfilt(b, a, data)
    
    fig = plt.figure(figsize=(8, 4))
    plt.plot(np.arange(0, nt, 1/sampling_fre), data_filter, 'k', linewidth=1)
    plt.title('Filtered waveform', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    fig.savefig('./fig/Filtered_waveform.png', dpi=600, bbox_inches='tight')
    return data_filter
    

# In[]
def plot_trace_waveform(data, nt, sampling_fre=100):
    '''
    Plot a single trace waveform (time series)
    
    Parameters
    ----------
    data : the input of 1-D data
    nt : the length of the signal (sec)
    sampling_fre : sampling frequency

    Returns
    -------
    None.

    '''
    sampling_interval = 1/sampling_fre
    t = np.arange(0, nt, sampling_interval)
    
    fig = plt.figure(figsize=(8, 4))
    plt.plot(t, data, 'k', linewidth=1)
    plt.title('Waveform', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    fig.savefig('./fig/Waveform.png', dpi=600, bbox_inches='tight')


# In[]
def plot_envelope(data, delta, sampling_fre=100):
    '''
    Plot the envelope of the data (time series)

    Parameters
    ----------
    data : the input of 1-D data
    delta : the interval of the envelope (sec)
    sampling_fre : sampling frequency

    Returns
    -------
    xx : the x-axis of the envelope
    yy : the y-axis of the envelope (values of envelope)

    '''    
    data_envelope = data.reshape(int(len(data)/(delta*sampling_fre)), int(delta*sampling_fre))
    data_envelope = np.max(data_envelope, axis=1)
    x = np.linspace(0, len(data), int(len(data)/(delta*sampling_fre)))
    f = scipy.interpolate.interp1d(x, abs(data_envelope), kind='cubic')
    xx = np.linspace(0, len(data), len(data))
    yy = f(xx)
    
    fig = plt.figure(figsize=(8, 4))
    plt.plot(xx, data, 'k', linewidth=1)
    plt.plot(xx, yy, 'r', linewidth=1)
    plt.title('Envelope', fontsize=16)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    fig.savefig('./fig/Envelope.png', dpi=600, bbox_inches='tight')
    return xx, yy
    
    
# In[]
def plot_STFT(data, nt, sampling_fre=100):
    '''
    Plot the Short time Fourier transform
    
    Parameters
    ----------
    data : the input of 1-D data
    nt : the length of the signal (sec)
    sampling_fre : sampling frequency

    Returns
    ----------
    fre : frequency (Hz)
    ts : time (sec)
    z : the time-frequency data of STFT 

    '''  
    fre, ts, amp = scipy.signal.stft(data, fs=sampling_fre, window='hann', nperseg=256)
    z = np.abs(amp.copy())
    
    fig = plt.figure(figsize=(8, 4))
    plt.pcolormesh(ts, fre, z)
    plt.colorbar(extend='both', aspect=20)
    plt.ylim(max(fre), 0)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Frequency (Hz)', fontsize=14)
    plt.title('Short time Fourier transform', fontsize=16)
    fig.savefig('./fig/STFT.png', dpi=600, bbox_inches='tight')
    return fre, ts, z


# In[]
def plot_WT(data, nt, sampling_fre=100):
    '''
    Plot the Wavelet transform
    
    Parameters
    ----------
    data : the input of 1-D data
    nt : the length of the signal (sec)
    sampling_fre : sampling frequency

    Returns
    -------
    cwtmatr : the time-frequency data of WT
    
    '''
    totalscal = sampling_fre/2  # totalscal is fixed at half of the sampling_fre 
    cf = pywt.central_frequency('cgau8')    # central frequency
    cparam = 2*cf*totalscal
    scales = cparam/np.arange(1, totalscal+1)
    [cwtmatr, frequencies] = pywt.cwt(data/np.max(abs(data)), scales, 'cgau8', 1/sampling_fre)
    
    fig = plt.figure(figsize=(8, 4))
    plt.imshow(abs(cwtmatr), aspect='auto')
    plt.ylim(max(frequencies), 0)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Frequency (Hz)', fontsize=14)
    plt.title('Wavelet transform', fontsize=16)
    plt.colorbar(extend='both', aspect=20)
    fig.savefig('./fig/WT.png', dpi=600, bbox_inches='tight')
    return abs(cwtmatr)
    

# In[]    
def plot_ST(data):
    '''
    S-transform

    Parameters
    ----------
    data : real matrix

    Returns
    -------
    complex matrix

    '''
    H = np.fft.fft(data)
    n = len(data)
    t = np.append(np.arange(np.ceil(n/2)), np.arange(-np.floor(n/2), 0))
    t2 = np.reciprocal(t[1:])[None]
    t = t[None].T
    t3 = np.matmul(t, t2)
    t4 = np.exp(-2*np.pi*np.pi*np.power(t3, 2))
    t5 = np.zeros([n, 1])
    t5[0] = 1
    t6 = np.append(t5, t4, axis=1)
    t7 = H[None]
    tt = np.arange(0, n)
    for i in range(1, n):
        t7 = np.append(t7, H[np.roll(tt, -i)][None], axis=0)
    return np.fft.fft(np.fft.ifft2(t6*t7)).T