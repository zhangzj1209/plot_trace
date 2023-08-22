import obspy
from plot_trace import *
from plot_multi_trace import *

input = obspy.read('./seis.sac')[0]
data = input.data
sampling_fre = input.stats.sampling_rate    # 100 Hz
nt = input.stats.npts*input.stats.delta     # 10s

oneD_FFT(data, nt, sampling_fre)
oneD_filter(data, nt, fre_min=2, fre_max=20, filter_order=2, sampling_fre=sampling_fre)
plot_trace_waveform(data, nt, sampling_fre)
plot_envelope(data, delta=0.5, sampling_fre=sampling_fre)
plot_STFT(data, nt, sampling_fre)
plot_WT(data, nt, sampling_fre)
plot_ST(data)

#plot_multi_trace_waveform_energy('./data/')