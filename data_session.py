#################################################################################################
## Overall explanation
##
## This class Session is meant to load all the data from a Session, for one monkey, for a brain area.
##
## Once the instance of the class is created, the data are not loaded yet, and will be loaded only when it's needed.
## Once the data are loaded, they are stored in a numpy multidirectional array build with the following structures :
## DATA -> Key (Seen, Missed, Correct Rejections, False Alarm) -> 3 dimentional numpy array (Time, Channels, Trials)
##
## The class has two main graphs it can produced.
## Figure 1 will be a classical ploting of MUA over time
## Figure 2 will plot MUA over time for different condition (Trials, electrodes, contrasts, etc...)
##
## Both graph can be averaged over trials and electrodes, not contrast yet.
##
## A function section has been created, with usefull tools for working with the data. They are not meant
## to be used by the users.
##
## It's important to note that Channels and Electrodes are synomyms here



#################################################################################################
## Import Section
from scipy import io
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from math import isnan
import numpy.ma as ma

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)





from abstract_session import AbstractSession



class Session(AbstractSession):
	


	def __init__(self,file_name,directory_path="data/",normalized=True):
		self.raw_data = io.loadmat(directory_path+file_name)["DATA_SESSION"]
		self.data = {}	
		self.file_name = file_name
		self.session = file_name.split("SESSION_")[1].split(".")[0]
		self.directory_path = directory_path
		if normalized:
			self.get_data("seen",normalized=True)
			self.get_data("missed",normalized=True)
			self.get_data("correct_rejections",normalized=True)
			self.get_data("false_alarm",normalized=True)


#################################################################################################
## Loading Data session

	def get_data(self,data_type,low_contrast=True,medium_contrast=True,high_contrast=True,average_over_trials=False,normalized=False):
		if data_type in self.data.keys():
			data = self.data[data_type]
			if not (low_contrast and medium_contrast and high_contrast):
				index = self.get_contrast_index(low_contrast,medium_contrast,high_contrast)
				data = data[index]
			return data
		full_data = self.raw_data
		if average_over_trials:
			text="AverageTrials"
		else:
			text="AllTrials"
		if data_type == "correct_rejections":
			#We add here a default contrast for the correct_rejections trials
			data = full_data["Conditions"][0][0][text][0][0]["CR"][0]
			self.data[data_type] = data
			self.remove_corrupted_channels(data_type)
		elif data_type == "false_alarm":
			#We add here a default contrast for the false_alarm trials
			data = full_data["Conditions"][0][0][text][0][0]["FA"][0]
			self.data[data_type] = data
			self.remove_corrupted_channels(data_type)
		elif data_type == "seen":
			data = full_data["Conditions"][0][0][text][0][0]["LUM"][0][0]["Seen"][0]
			self.data[data_type] = data
			self.remove_corrupted_channels(data_type)
		elif data_type == "missed":
			data = full_data["Conditions"][0][0][text][0][0]["LUM"][0][0]["Missed"][0]
			self.data[data_type] = data
			self.remove_corrupted_channels(data_type)
		elif data_type == "texture":
			#We add here a default contrast for the texture trials
			data = full_data["Conditions"][0][0][text][0][0]["Text"][0]
			self.data[data_type] = data
			self.remove_corrupted_channels(data_type)
		elif data_type == "time":
			data = full_data["SessionInfo"][0][0]["Time"][0][0][0]*1000
			self.data[data_type] = data
		elif data_type == "session":
			data = np.asarray([self.session]*self.get_contrast_number())
			self.data[data_type] = data
		elif data_type == "best_channel":
			data = np.asarray([self.get_best_snr("channel")]*self.get_contrast_number())
			self.data[data_type] = data
		elif data_type == "contrast_snr":
			data = np.asarray(self.get_snr("contrast"))
			self.data[data_type] = data
		elif data_type == "contrast":
			data = full_data["SessionInfo"][0][0]["Contrasts"][0][0][0]
			self.data[data_type] = data
		elif data_type == "contrast_performance":
			data = []
			miss_trials = self.get_data("missed")
			hit_trials = self.get_data("seen")
			for i in range(len(hit_trials)):
				if hit_trials[i].ndim == 3:
					hit = len(hit_trials[i][0][0])
				elif hit_trials[i].ndim == 2:
					hit = len(hit_trials[i][0])
				else:
					print("Warning, data shape not recognized")
				if miss_trials[i].ndim == 3:
					miss = len(miss_trials[i][0][0])
				elif hit_trials[i].ndim == 2:
					miss = len(miss_trials[i][0])
				else:
					print("Warning, data shape not recognized")
				data.append(hit/(hit+miss))
			self.data[data_type] = np.asarray(data)
		elif data_type == "contrast_category":
			mua_file_name = self.session + "_MUA.mat" 
			raw_data = io.loadmat(self.directory_path+mua_file_name)
			all_contrast_type = raw_data["MUA"][0][0]["INFO"][0][0]["Categorie"][0]
			hit = raw_data["MUA"][0][0]["LUM"]["psMS_S"][0]
			miss = raw_data["MUA"][0][0]["LUM"]["psMS_M"][0]

			def get_trial_number(x):
				if x.ndim == 3:
					return len(x[0,0,:])
				elif x.ndim == 2:
					return 1
				else:
					print("Unexpected data shape")

			hit = np.asarray(list(map(get_trial_number,hit)))
			miss = np.asarray(list(map(get_trial_number,miss)))

			mask = (hit >= 3) & (miss >= 3)
			contrast_type = all_contrast_type[mask]
			data = np.asarray(contrast_type)
			self.data[data_type] = data
		else:
			raise(BaseException("Data Type not found"))
		if normalized:
			"""
			#Getting the baseline
			electrode_averaged = self.average_over(data,trials=True,contrast=True)
			electrode_averaged = electrode_averaged[0:self.stimulus_offset]
			electrode_baseline = self.average_over(electrode_averaged,time=True)
			"""
			
			#Getting the normalisation factor
			text = self.get_data("texture")[0]
			electrode_texture_response = self.average_over(text,trials=True)
			electrode_baseline = self.average_over(electrode_texture_response[:self.stimulus_offset],time=True)
			electrode_max_peak = np.max(electrode_texture_response[self.stimulus_offset:self.stimulus_offset+300]-electrode_baseline,axis=0)
			#Normalizing the data
			for i in range(len(data)):
				try:
					data[i] = np.swapaxes((np.swapaxes(data[i],1,2)-electrode_baseline)/electrode_max_peak,1,2)
				except np.AxisError:
					data[i] = (data[i]-electrode_baseline)/electrode_max_peak
		if not (low_contrast and medium_contrast and high_contrast):
			index = self.get_contrast_index(low_contrast,medium_contrast,high_contrast)
			data = data[index]
		return data

