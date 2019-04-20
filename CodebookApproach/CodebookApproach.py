import numpy as np
from sklearn.cluster import KMeans
import sys
from sklearn.preprocessing import MinMaxScaler


class CodebookApproach:
    def __init__(self, window_length, overlap_length, cluster_number):
        self.window_length = window_length
        self.overlap_length = overlap_length
        self.cluster_number = cluster_number
    
    def extract_windows(self, X):
        '''Windows are extracted from each sensor channel by using window length and overlap length.

        Args:
          X: One-axis sensor data
        Returns:
          windows (list): Extracted windows.
        '''
        windows = []
    
        max_window_index = len(X) - self.window_length + 1
        i = 0
    
        while i < max_window_index:
            w = X[i : i + self.window_length]
            windows.append(w)
            i += self.overlap_length
        
        return windows

    def get_windows(self, X):
        '''Windows are extracted from each sensor channel.

        Args:
          X: Three-axis sensor data with 3d shape. (samples, 3-axis, data)
        Returns:
          windows_seperate (numpy.array): Extracted windows which are appended in seperate list for each samples.
          windows_combined (numpy.array): Extracted windows which are are appended in one list for all samples.
        '''
    
        windows_seperate = [[],[],[]]
        windows_combined = [[],[],[]]
    
        for i in range(0,len(X)):
            for channel in range(3):
                windows = self.extract_windows(X[i][channel])
                windows_seperate[channel].append(windows)
                windows_combined[channel].extend(windows)
            
        return np.array(windows_seperate), np.array(windows_combined) 

    def extract_subsequences(self, dataset):
        '''Subsequences are extracted from each sensor.
        
        Args:
            dataset (list): Three-axis sensors data with 4d shape. (sensors, samples, 3-axis, data)
        Returns:
            windows_seperate (list): Extracted windows which are appended in seperate list for each samples.
            windows_combined (list): Extracted windows which are are appended in one list for all samples.
        '''
        windows_seperate = []
        windows_combined = []
    
        for i in range(len(dataset)):
            windows_s, windows_c = self.get_windows(dataset[i])
            windows_seperate.append(windows_s)
            windows_combined.append(windows_c)
        
        return windows_seperate, windows_combined     

    def get_codebook(self, windows):
        '''Calculates cluster centers by using k-means clustering algorithm.

        Args:
          windows: Extracted windows.
        Returns:
          cluster_centers : Cluster centers(codebook)
        '''
        kmeans = KMeans(n_clusters=self.cluster_number).fit(windows)
        return kmeans.cluster_centers_

    def get_codebooks(self, windows):
        '''Calculates cluster centers(codebook) for each sensor channel.

        Args:
          windows: Extracted windows from each sensor.
        Returns:
          codewords: Calculated codebooks.
        '''
        codewords = []
    
        for sensor in range(len(windows)):
            sensor_codewords = [[],[],[]]
            windows_s = windows[sensor]
        
            for channel in range(windows_s.shape[0]):
                sensor_codewords[channel] = self.get_codebook(windows_s[channel])
        
            codewords.append(sensor_codewords) 
        
        return np.array(codewords)

    def normalize(self, X):
        '''Normalizes by using MinMaxScaler.

        Args:
          X: data
        Returns:
          X_norm: normalized data
        '''
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_norm = scaler.fit_transform(X)
        return X_norm

    def hard_codeword_assignment(self, window, codebook, histogram):
        '''Assigns a window to most simimlar codeword by incrementing of its frequency.
        
        Args:
          window: data
          codebook: Constructed codebook. Codebook and window args should be acquired from the same sensor channel.
          histogram: Histogram-type feature of a sequence.
        Returns:
          histogram: Histogram-type feature of a sequence.
        '''
        min_distance = sys.float_info.max
        codeword_index = -1  
        
        for i in range(0,len(codebook)):
            distance = np.linalg.norm(window - codebook[i])
                    
            if distance < min_distance:
                min_distance  = distance
                codeword_index = i

        histogram[codeword_index] += 1
        return histogram

    def assign_codewords(self, codebooks, windows):
        '''Represents each sequence as a histogram-type feature.

        Args:
          codebooks: Calculated codebooks.
          windows: Extracted windows.
        Returns:
          histogram_sensors: Normalized histogram-type features.
        '''
        histogram_sensors = []
        for sensor in range(len(windows)):
            codebook = codebooks[sensor]
        
            histogram_sensor = [[],[],[]]
            for channel in range(3):
            
                histogram_channel = []
            
                for i in range(windows[sensor].shape[1]):
                    windows_for_activity = windows[sensor][channel][i]
                    histogram = np.zeros(codebook.shape[1])
                
                    #Encodes a sequence as a histogram-type feature which is a distirbution of codewords.
                    for j in range(len(windows_for_activity)):
                        histogram = self.hard_codeword_assignment(windows_for_activity[j], codebook[channel], histogram)
                
                    histogram_channel.append(histogram)
            
                histogram_sensor[channel] = self.normalize(histogram_channel)
        
            histogram_sensors.append(histogram_sensor)
        
        return np.array(histogram_sensors)

    def concatenate_histograms(self, histograms):
        '''Concatenates histograms for each sequence.

        Args:
          histograms: Normalized histogram-type features.
        Returns:
          histogram_sensors: Concatenated histograms for each sequence.
        '''
        histogram_sensors = []

        for i in range(histograms.shape[0]):
            histogram_sensor = []
            
            for j in range(histograms.shape[2]):
                histogram_channel = []
                histogram_channel.extend(histograms[i][0][j])
                histogram_channel.extend(histograms[i][1][j])
                histogram_channel.extend(histograms[i][2][j])
                histogram_sensor.append(histogram_channel)
                
            histogram_sensors.append(histogram_sensor)
         
        histograms_tuple = ()
        for j in range(len(histogram_sensors)):
            histograms_tuple = histograms_tuple + (histogram_sensors[j],)
            
        return np.hstack(histograms_tuple)