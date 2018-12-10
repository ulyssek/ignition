########################################################################################################
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



########################################################################################################
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







class Session():


	def __init__(self,file_name,directory_path="data/"):
		self.raw_data = io.loadmat(directory_path+file_name)["DATA_SESSION"]
		self.data = {}	

########################################################################################################
## Loading Data session

	def get_data(self,data_type,average_over_trials=False):
		if data_type in self.data.keys():
			return self.data[data_type]
		full_data = self.raw_data
		if average_over_trials:
			text="AverageTrials"
		else:
			text="AllTrials"
		if data_type == "correct_rejections":
			#We add here a default contrast for the correct_rejections trials
			data = [full_data["Conditions"][0][0][text][0][0]["CR"][0][0]]
			self.data[data_type] = data
			self.remove_corrupted_channels(data_type)
		elif data_type == "false_alarm":
			#We add here a default contrast for the false_alarm trials
			data = [full_data["Conditions"][0][0][text][0][0]["FA"][0][0]]
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
			data = [full_data["Conditions"][0][0][text][0][0]["Text"][0][0]]
			self.data[data_type] = data
			self.remove_corrupted_channels(data_type)
		elif data_type == "time":
			data = full_data["SessionInfo"][0][0]["Time"][0][0][0]*1000
			self.data[data_type] = data
		elif data_type == "contrast":
			data = full_data["SessionInfo"][0][0]["Contrasts"][0][0][0]
			self.data[data_type] = data
		elif data_type == "contrast_performance":
			data = []
			miss_trials = self.get_data("missed")
			hit_trials = self.get_data("seen")
			for i in range(len(hit_trials)):
				hit = len(hit_trials[i][0][0])
				miss = len(miss_trials[i][0][0])
				data.append(hit/(hit+miss))
			self.data[data_type] = np.asarray(data)
		else:
			raise(BaseException("Data Type not found"))
		return data

########################################################################################################
## Ploting functions


	def plot_figure_2(self,data_type,contrast=0,title="Title not set yet",ao_channels=True,ao_trials=False,smooth=1):
		if ao_channels:
			averaged_over = "channels"
			displayed_over = "trials"
		else:
			averaged_over = "trials"
			displayed_over = "channels"
		title = "MUA over time over " + displayed_over + " averaged over " + averaged_over + " for " 
		if data_type in ("seen","missed"):
			title += "contrast #" + str(contrast+1) + " and " 
		title += data_type + " condition"
		data = self.get_data(data_type)
		data_averaged = self.average_over(data[contrast],time=False,channels=ao_channels,trials=ao_trials)
		final_data = self.array_smoother(data_averaged,smooth)
		fig = self._core_figure_2(final_data,title,displayed_over)
		iplot(fig, filename='basic-line')

	def plot_figure_1(self,data_type,contrast=0,title="Title not set yet",ao_channels=True,ao_trials=False,smooth=1,show=True):
		title = "MUA over time "
		if ao_channels:
			title += "averaged over channels "
		if ao_trials:
			title += "averaged over trials "
		title += "for contrast #" + str(contrast+1) + " and " + data_type + " condition"
		data = self.get_data(data_type)
		data_averaged = self.average_over(data[contrast],time=False,channels=ao_channels,trials=ao_trials)
		self._core_figure_1(data_averaged,title,smooth,show)


	def plot_figure_3(self,data_type1,data_type2,contrast=0,title="Title not set yet",ao_channels=True,smooth=1,show=True):
		d1 = self.get_data(data_type1)[contrast]
		d2 = self.get_data(data_type2)[contrast]
		d2 = self.average_over(d2,time=False,channels=ao_channels,trials=True)
		d1 = self.average_over(d1,time=False,channels=ao_channels,trials=True)

		data = d1-d2

		title = "Differences of MUA over time "
		if ao_channels:
			title += "averaged over channels "
		title += "for contrast #" + str(contrast+1) + " and " + data_type1 + " - " + data_type2 + " condition"
		self._core_figure_1(data,title,smooth,show)

	def _core_figure_2(self,full_data,title,ylabel):
		timing_step = self.get_data("time")
		data = [
		go.Heatmap(z=full_data,
			x=np.asarray(timing_step),
			y=np.asarray(range(len(full_data[0]))),
			)
		]
		layout = go.Layout(
			title=title,
			autosize=False,
			width=1000,
			height=500,
			margin=dict(
				l=65,
				r=50,
				b=65,
				t=90
			),
			xaxis=dict(
				title='Time',
				titlefont=dict(
					family='Courier New, monospace',
					size=18,
					color='#7f7f7f'
				)
			),
			yaxis=dict(
				title=ylabel,
				titlefont=dict(
					family='Courier New, monospace',
					size=18,
					color='#7f7f7f'
				)
			),
		)

		fig = go.Figure(data=data, layout=layout)
		return fig

	def _core_figure_1(self,full_data,title,smooth,show):
		dim = len(full_data.shape)
		self._loop_figure_1(full_data,dim-1,smooth)
		plt.title(title)
		plt.xlabel("Time (ms)")
		plt.ylabel("Multi Unit Activity")
		if show:
			plt.show()

	def _loop_figure_1(self,data,i,smooth):
		if i == 0:
			data = self.smoother(data,smooth)
			plt.plot(self.get_data("time"),data)
		else:
			for j in range(len(data[0])):
				self._loop_figure_1(data[:,j],i-1,smooth)
			
		

########################################################################################################
## Miscallaneous functions

	def average_over(self,data,time=False,channels=False,trials=False,contrast=False):
		if contrast:
			averaged_data = []
			for trial_data in data:
				trial_data_averaged = average_over(trial_data,time=time,channels=channels,trials=trials)
				averaged_data.append(trial_data_averaged)

			data = np.asarray(averaged_data)
			data = np.nanmean(averaged_data,0)
		else:
			if trials:
				data = np.nanmean(data,2)
			if channels:
				data = np.nanmean(data,1)
			if time:
				data = np.nanmean(data,0)
		return data

	def smoother(self,data,window=10):
		result = []
		for i in range(len(data)):
			a = data[int(max(0,i-window/2)):int(min(len(data-1),i+window/2))]
			result.append(np.mean(a))
		return result

	def array_smoother(self,data,window=50):
		smoothed_data = []
		for i in range(len(data[0])):
			smoothed_data.append(self.smoother(data[:,i]-np.nanmean(data[:,i],axis=0),window))
		return np.asarray(smoothed_data)

	def remove_corrupted_channels(self,data_type):
		for i in range(len(self.data[data_type])):
			data_averaged = self.average_over(self.data[data_type][i],time=True,channels=False,trials=True)
			corrupted_channels = np.isnan(data_averaged)
			self.data[data_type][i] = self.data[data_type][i][:,~corrupted_channels,:]

	def get_contrast_value(self,contrast_index,round_number=2):
		contrasts = self.get_data("contrast")
		contrast = contrasts[contrast_index]
		contrast = round(contrast,round_number)

