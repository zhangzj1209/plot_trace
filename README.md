# Processing of One-trace & Multi-traces Seismic Signals Based on Python

## Installation (via Anaconda)
```
conda install numpy==1.23.5 matplotlib==3.7.2 obspy==1.3.0 scipy==1.11.1 PyWavelets==1.4.1
```

## Description
- In `plot_trace.py`:
  - `plot_trace_waveform`: plot a single trace waveform (time series)
    ![image](https://github.com/zhangzj1209/Plot_trace/blob/main/fig/Waveform.png)
  
  - `oneD_FFT`: 1-D Fast Fourier Transform (*FFT*)
    ![image](https://github.com/zhangzj1209/Plot_trace/blob/main/fig/FFT_single_sided_spectrum.png)
    
  - `oneD_filter`: 1-D filter
    ![image](https://github.com/zhangzj1209/Plot_trace/blob/main/fig/Filtered_waveform.png)
 
  - `plot_envelope`: plot the envelope of the data (time series)
    ![image](https://github.com/zhangzj1209/Plot_trace/blob/main/fig/Envelope.png)
  
  - `plot_STFT`: plot the Short Time Fourier Transform
    ![image](https://github.com/zhangzj1209/Plot_trace/blob/main/fig/STFT.png)
 
  - `plot_WT`: plot the Wavelet Transform
    ![image](https://github.com/zhangzj1209/Plot_trace/blob/main/fig/WT.png)
 
  - `plot_ST`: plot the S-Transform (***uncompleted***)


- In `plot_multi_trace.py`:
  - `plot_multi_trace_waveform_energy`: use `.sac` files placed in a separate folder `./data/` to plot the multi-station waveform & multi-station energy.
    
    ***Note***: We did **not** include an example of multi-station seismic signals here.
