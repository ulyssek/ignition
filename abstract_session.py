
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








class AbstractSession():


	
#################################################################################################
## Ploting functions


	def plot_figure_2(self,data_type,contrast=0,title="Title not set yet",ao_channels=True,ao_trials=False,ao_contrast=False,smooth=1):
		if ao_channels:
			averaged_over = "channels"
			displayed_over = "trials"
		else:
			averaged_over = "trials"
			displayed_over = "channels"
		title = "MUA over time over " + displayed_over + " averaged over " + averaged_over 
		if data_type in ("seen","missed"):
			if ao_contrast:
				title += "averaged over contrast "
			else:
				title += " for contrast #" + str(contrast+1) + " and " 
		title += data_type + " condition"
		data = self.get_data(data_type)
		if ao_contrast:
			data_averaged = self.average_over(data,time=False,channels=ao_channels,trials=ao_trials,contrast=True)
		else:
			data_averaged = self.average_over(data[contrast],time=False,channels=ao_channels,trials=ao_trials)
		final_data = self.array_smoother(data_averaged,smooth)
		fig = self._core_figure_2(final_data,title,displayed_over)
		iplot(fig, filename='basic-line')

	def plot_figure_1(self,data_type,contrast=0,title="Title not set yet",ao_channels=True,ao_trials=False,ao_contrast=False,low_contrast=True,medium_contrast=True,high_contrast=True,smooth=1,show=True):
		title = "MUA over time "
		if ao_channels or ao_contrast:
			title += "averaged over channels "
		if ao_trials or ao_contrast:
			title += "averaged over trials "
		if ao_contrast:
			title += "averaged over contrast"
		else:
			title += "for contrast #" + str(contrast+1) 
		title += " and " + data_type + " condition"
		data = self.get_data(data_type,low_contrast=low_contrast,medium_contrast=medium_contrast,high_contrast=high_contrast)
		if ao_contrast:
			data_averaged = self.average_over(data,time=False,channels=True,trials=True,contrast=True)
		else:
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

	def _core_figure_1(self,full_data,title,smooth,show,time_window=(0,None)):
		if time_window[1] is None:
			time_window = (time_window[0],len(self.get_data("time")))
		dim = len(full_data.shape)
		if type(full_data)==np.float64:
			print("Couldn't print the graph, empty data")
			return
		self._loop_figure_1(full_data,dim-1,smooth,time_window)
		plt.title(title)
		plt.xlabel("Time (ms)")
		plt.ylabel("Multi Unit Activity")
		if show:
			plt.show()

	def _loop_figure_1(self,data,i,smooth,time_window):
		if i == 0:
			data = self.smoother(data,smooth)
			plt.plot(self.get_data("time")[time_window[0]:time_window[1]],data)
		else:
			for j in range(len(data[0])):
				self._loop_figure_1(data[:,j],i-1,smooth,time_window)
			
#################################################################################################
## Miscallaneous functions

	def average_over(self,data,time=False,channels=False,trials=False,contrast=False):
		if contrast:
			averaged_data = []
			for trial_data in data:
				trial_data_averaged = self.average_over(trial_data,time=time,channels=channels,trials=trials)
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
	
	def var_over(self,data,time=False,channels=False,trials=False,contrast=False):
		axis = ()
		if trials:
			axis += (2,)
		if channels:
			axis += (1,)
		if time:
			axis += (0,)
		return np.nanstd(data,axis)


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
		return contrast

	def get_contrast_index(self,low_contrast=True,medium_contrast=True,high_contrast=True):
		contrast_performance = self.get_data("contrast_performance")
		contrast_index = []
		for i in range(len(contrast_performance)):
			if low_contrast and contrast_performance[i] <= 0.2:
				contrast_index.append(i)
			elif medium_contrast and contrast_performance[i] <= 0.8 and contrast_performance[i] > 0.2:
				contrast_index.append(i)
			elif high_contrast and contrast_performance[i] > 0.8:
				contrast_index.append(i)
		return contrast_index
			

