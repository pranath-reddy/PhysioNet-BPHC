import numpy as np
from scipy.signal import butter, lfilter
from scipy import stats
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

def detect_peaks(ecg_measurements,signal_frequency,gain):

        """
        Method responsible for extracting peaks from loaded ECG measurements data through measurements processing.
        This implementation of a QRS Complex Detector is by no means a certified medical tool and should not be used in health monitoring. 
        It was created and used for experimental purposes in psychophysiology and psychology.
        You can find more information in module documentation:
        https://github.com/c-labpl/qrs_detector
        If you use these modules in a research project, please consider citing it:
        https://zenodo.org/record/583770
        If you use these modules in any other project, please refer to MIT open-source license.
        If you have any question on the implementation, please refer to:
        Michal Sznajder (Jagiellonian University) - technical contact (msznajder@gmail.com)
        Marta lukowska (Jagiellonian University)
        Janko Slavic peak detection algorithm and implementation.
        https://github.com/c-labpl/qrs_detector
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        
        MIT License
        Copyright (c) 2017 Michal Sznajder, Marta Lukowska
    
        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:
        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.
        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.
        """


        filter_lowcut = 0.001
        filter_highcut = 15.0
        filter_order = 1
        integration_window = 30  # Change proportionally when adjusting frequency (in samples).
        findpeaks_limit = 0.35
        findpeaks_spacing = 100  # Change proportionally when adjusting frequency (in samples).
        refractory_period = 240  # Change proportionally when adjusting frequency (in samples).
        qrs_peak_filtering_factor = 0.125
        noise_peak_filtering_factor = 0.125
        qrs_noise_diff_weight = 0.25


        # Detection results.
        qrs_peaks_indices = np.array([], dtype=int)
        noise_peaks_indices = np.array([], dtype=int)


        # Measurements filtering - 0-15 Hz band pass filter.
        filtered_ecg_measurements = bandpass_filter(ecg_measurements, lowcut=filter_lowcut, highcut=filter_highcut, signal_freq=signal_frequency, filter_order=filter_order)

        filtered_ecg_measurements[:5] = filtered_ecg_measurements[5]

        # Derivative - provides QRS slope information.
        differentiated_ecg_measurements = np.ediff1d(filtered_ecg_measurements)

        # Squaring - intensifies values received in derivative.
        squared_ecg_measurements = differentiated_ecg_measurements ** 2

        # Moving-window integration.
        integrated_ecg_measurements = np.convolve(squared_ecg_measurements, np.ones(integration_window)/integration_window)

        # Fiducial mark - peak detection on integrated measurements.
        detected_peaks_indices = findpeaks(data=integrated_ecg_measurements,
                                                     limit=findpeaks_limit,
                                                     spacing=findpeaks_spacing)

        detected_peaks_values = integrated_ecg_measurements[detected_peaks_indices]

        return detected_peaks_values,detected_peaks_indices



"""QRS detection methods."""

def detect_qrs(channel_number, ecg_data_raw, detected_peaks_indices, detected_peaks_values):
    """
    Method responsible for classifying detected ECG measurements peaks either as noise or as QRS complex (heart beat).
    """

    filter_lowcut = 0.001
    filter_highcut = 15.0
    filter_order = 1
    integration_window = 30  # Change proportionally when adjusting frequency (in samples).
    findpeaks_limit = 0.35
    findpeaks_spacing = 100  # Change proportionally when adjusting frequency (in samples).
    refractory_period = 240  # Change proportionally when adjusting frequency (in samples).
    qrs_peak_filtering_factor = 0.125
    noise_peak_filtering_factor = 0.125
    qrs_noise_diff_weight = 0.25

    # Measured and calculated values.
    qrs_peak_value = 0.0
    noise_peak_value = 0.0
    threshold_value = 0.0

    # Detection results.
    qrs_peaks_indices = np.array([], dtype=int)
    noise_peaks_indices = np.array([], dtype=int)
    
    for detected_peak_index, detected_peaks_value in zip(detected_peaks_indices, detected_peaks_values):

        try:
            last_qrs_index = qrs_peaks_indices[-1]
        except IndexError:
            last_qrs_index = 0

        # After a valid QRS complex detection, there is a 200 ms refractory period before next one can be detected.
        if detected_peak_index - last_qrs_index > refractory_period or not qrs_peaks_indices.size:
            # Peak must be classified either as a noise peak or a QRS peak.
            # To be classified as a QRS peak it must exceed dynamically set threshold value.
            if detected_peaks_value > threshold_value:
                qrs_peaks_indices = np.append(qrs_peaks_indices, detected_peak_index)

                # Adjust QRS peak value used later for setting QRS-noise threshold.
                qrs_peak_value = qrs_peak_filtering_factor * detected_peaks_value + \
                                      (1 - qrs_peak_filtering_factor) * qrs_peak_value
            else:
                noise_peaks_indices = np.append(noise_peaks_indices, detected_peak_index)

                # Adjust noise peak value used later for setting QRS-noise threshold.
                noise_peak_value = noise_peak_filtering_factor * detected_peaks_value + \
                                        (1 - noise_peak_filtering_factor) * noise_peak_value

            # Adjust QRS-noise threshold value based on previously detected QRS or noise peaks value.
            threshold_value = noise_peak_value + \
                                   qrs_noise_diff_weight * (qrs_peak_value - noise_peak_value)

    # Create array containing both input ECG measurements data and QRS detection indication column.
    # We mark QRS detection with '1' flag in 'qrs_detected' log column ('0' otherwise).
    measurement_qrs_detection_flag = np.zeros([len(ecg_data_raw[channel_number]), 1])
    measurement_qrs_detection_flag[qrs_peaks_indices] = 1
    #print(measurement_qrs_detection_flag)
    #print(np.reshape(self.ecg_data_raw[self.channel_number, :].transpose(), (7500,1)))
    ecg_data_detected = np.append(np.reshape(ecg_data_raw[channel_number].transpose(), (ecg_data_raw.shape[1],1)), measurement_qrs_detection_flag, 1)

    return ecg_data_detected, qrs_peaks_indices, noise_peaks_indices




def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
        """
        Method responsible for creating and applying Butterworth filter.
        :param deque data: raw data
        :param float lowcut: filter lowcut frequency value
        :param float highcut: filter highcut frequency value
        :param int signal_freq: signal frequency in samples per second (Hz)
        :param int filter_order: filter order
        :return array: filtered data
        """
        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y

def findpeaks(data, spacing=1, limit=None):
        """
        Janko Slavic peak detection algorithm and implementation.
        https://github.com/jankoslavic/py-tools/tree/master/findpeaks
        Finds peaks in `data` which are of `spacing` width and >=`limit`.
        :param ndarray data: data
        :param float spacing: minimum spacing to the next peak (should be 1 or more)
        :param float limit: peaks should have value greater or equal
        :return array: detected peaks indexes array
        """
        len = data.size
        x = np.zeros(len + 2 * spacing)
        x[:spacing] = data[0] - 1.e-6
        x[-spacing:] = data[-1] - 1.e-6
        x[spacing:spacing + len] = data
        peak_candidate = np.zeros(len)
        peak_candidate[:] = True
        for s in range(spacing):
            start = spacing - s - 1
            h_b = x[start: start + len]  # before
            start = spacing
            h_c = x[start: start + len]  # central
            start = spacing + s + 1
            h_a = x[start: start + len]  # after
            peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

        ind = np.argwhere(peak_candidate)
        ind = ind.reshape(ind.size)
        if limit is not None:
            ind = ind[data[ind] > limit]
        return ind

def segments_extract(data, qrs_peaks_indices):
    segment = 0
    peaks_length = 0

    for peaks in qrs_peaks_indices:
        if((peaks-125 < 0) or (peaks+250 > len(data[0]))):
            continue
        else:
            peaks_length = peaks_length + 1
    signal = np.zeros((1,peaks_length,375,12))
    for peaks in qrs_peaks_indices:
        
        peaks = int(peaks)
        if((peaks-125 < 0) or (peaks+250 > len(data[0]))):
            #print("segment = ", segment)
            continue
        for channelNo in range(0,12):
            #print(signalChannel[peaks-125:peaks+250])
            signal[:,segment,:,channelNo] = data[channelNo][peaks-125:peaks+250]
            #print("segment = ", segment)
            #print("channelNo = ", channelNo)
            #print("signal shape ", signal.shape)
            #print("signal ", signal[:,segment,:,channelNo])
        segment = segment + 1
        
    #print("Final Signal ", signal[:,:,:,:])


    return signal


def get_12ECG_features(data, header_data):

    tmp_hea = header_data[0].split(' ')
    ptID = tmp_hea[0]
    #print("ptID = ", ptID)
    num_leads = int(tmp_hea[1])
    sample_Fs= int(tmp_hea[2])
    gain_lead = np.zeros(num_leads)
    
    for ii in range(num_leads):
        tmp_hea = header_data[ii+1].split(' ')
        gain_lead[ii] = int(tmp_hea[2].split('/')[0])

    # for testing, we included the mean age of 57 if the age is a NaN
    # This value will change as more data is being released
    for iline in header_data:
        if iline.startswith('#Age'):
            tmp_age = iline.split(': ')[1].strip()
            age = int(tmp_age if tmp_age != 'NaN' else 57)
        elif iline.startswith('#Sex'):
            tmp_sex = iline.split(': ')[1]
            if tmp_sex.strip()=='Female':
                sex =1
            else:
                sex=0
        elif iline.startswith('#Dx'):
            labels = iline.split(': ')[1].split(',')

    
#   We are only using data from lead1
    peaks,idx = detect_peaks(data[0],sample_Fs,gain_lead[0])

    ecg_data_detected, qrs_peaks_indices, noise_peaks_indices = detect_qrs(0, data, idx, peaks)

    #print("detected qrs indices ", qrs_peaks_indices)
    #Generating Signals from detected peaks [shape of signal(1, no_of_segments, 375, 12)]
    signal = segments_extract(data, qrs_peaks_indices)
    signal = signal[0]
    data = signal.reshape(-1,12,375)
    
    a = data.shape[0]
    b = data.shape[1]

    data_temp = []
    for i in range(a):
        x = []
        for j in range(b):
        
            sample = data[i,j,:]
            min = np.amin(sample)
            max = np.amax(sample)
            range_val = max - min
            if range_val != 0:
                x_norm = (sample-min)/range_val
                x.append(x_norm)
            
        if np.asarray(x).shape[0] == 12:
            data_temp.append(x)

    data = np.asarray(data_temp).reshape(-1,375,12)

    #Code for feature extraction
    model1 = load_model('./feature_weights/AF_features.h5')
    model2 = load_model('./feature_weights/AVB_features.h5')
    model3 = load_model('./feature_weights/LBBB_features.h5')
    model4 = load_model('./feature_weights/Normal_features.h5')
    model5 = load_model('./feature_weights/PAC_features.h5')
    model6 = load_model('./feature_weights/PVC_features.h5')
    model7 = load_model('./feature_weights/RBBB_features.h5')
    model8 = load_model('./feature_weights/STD_features.h5')
    model9 = load_model('./feature_weights/STE_features.h5')
    
    featureModel1 = Model(inputs=model1.inputs, outputs=model1.layers[10].output)
    feature_maps1 = featureModel1.predict(data)

    featureModel2 = Model(inputs=model2.inputs, outputs=model2.layers[10].output)
    feature_maps2 = featureModel2.predict(data)

    featureModel3 = Model(inputs=model3.inputs, outputs=model3.layers[10].output)
    feature_maps3 = featureModel3.predict(data)

    featureModel4 = Model(inputs=model4.inputs, outputs=model4.layers[10].output)
    feature_maps4 = featureModel4.predict(data)

    featureModel5 = Model(inputs=model5.inputs, outputs=model5.layers[10].output)
    feature_maps5 = featureModel5.predict(data)

    featureModel6 = Model(inputs=model6.inputs, outputs=model6.layers[10].output)
    feature_maps6 = featureModel6.predict(data)

    featureModel7 = Model(inputs=model7.inputs, outputs=model7.layers[10].output)
    feature_maps7 = featureModel7.predict(data)

    featureModel8 = Model(inputs=model8.inputs, outputs=model8.layers[10].output)
    feature_maps8 = featureModel8.predict(data)

    featureModel9 = Model(inputs=model9.inputs, outputs=model9.layers[10].output)
    feature_maps9 = featureModel9.predict(data)

    features = np.concatenate((feature_maps1, feature_maps2, feature_maps3, feature_maps4, feature_maps5, feature_maps6, feature_maps7, feature_maps8, feature_maps9), axis=1)
    
    return features
