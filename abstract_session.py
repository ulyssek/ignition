
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

	stimulus_offset = 228 #Magic number defining the stimulus offset
	stimulus_end = 428 #Magic number defining the end of the stimulus peak response

	
#################################################################################################
## Ploting functions

	def plot_figure_1(self,data_type,contrast=0,title="Title not set yet",ao_channels=True,ao_trials=False,ao_contrast=False,low_contrast=True,medium_contrast=True,high_contrast=True,smooth=1,show=True,clip=False):
		if data_type in ("false_alarm","correct_rejections"):
			contrast = 0
			print("No contrast for false alarms or correct rejections")
		title = "MUA over time "
		if ao_channels or ao_contrast:
			title += "averaged over channels "
		if ao_trials or ao_contrast:
			title += "averaged over trials "
		if ao_contrast:
			title += "averaged over contrast"
		else:
			title += "for contrast #" + str(contrast+1) 
		title += " and " + str(data_type) + " condition"
		if type(data_type) == type("str"):
			data = self.get_data(data_type,low_contrast=low_contrast,medium_contrast=medium_contrast,high_contrast=high_contrast)
		else:
			data = np.asarray(list(map(lambda x : self.get_data(x,low_contrast=low_contrast,medium_contrast=medium_contrast,high_contrast=high_contrast),data_type)))
			data = np.concatenate(data)
		if ao_contrast:
			data_averaged = self.average_over(data,time=False,channels=True,trials=True,contrast=True)
		else:
			data_averaged = self.average_over(data[contrast],time=False,channels=ao_channels,trials=ao_trials)
		if clip:
			data_averaged=np.clip(data_averaged,-10,1)	
		self._core_figure_1(data_averaged,title,smooth,show)
		return data_averaged



	def plot_figure_2(self,data_type,contrast=0,title="Title not set yet",ao_channels=True,ao_trials=False,ao_contrast=False,smooth=1,clip=None):
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
		if clip is not None:
			final_data = np.clip(final_data,-10,clip)
		fig = self._core_figure_2(final_data,title,displayed_over)
		iplot(fig, filename='basic-line')
		return final_data
	

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

	def plot_figure_4(self,data_type,title="Title not set yet",smooth=1,clip=False,ylabel=""):
		full_data = self.get_data(data_type)
		display_order = list(range(len(self.get_data("contrast"))))
		display_order.sort(key=lambda x : self.get_data("contrast")[x])
		full_data = [full_data[i] for i in display_order]
		contrasts = [self.get_data("contrast")[i] for i in display_order]
		data_averaged = []
		for data in full_data:
			data_averaged.append(self.average_over(data,time=False,channels=True,trials=True))
		data_averaged = np.transpose(np.asarray(data_averaged))
		final_data = self.array_smoother(data_averaged,smooth)
		if clip:
			final_data = np.clip(final_data,-10,1)
		fig = self._core_figure_2(final_data,title,ylabel,contrasts)
		iplot(fig, filename='basic-line')

	def plot_figure_5(self,data_type,contrast=0,title="Title not set yet",ao_channels=False,low_contrast=True,medium_contrast=True,high_contrast=True,smooth=1,show=True,clip=False,baseline=False,ylabel="Variance"):
		data = self.get_data(data_type,low_contrast=low_contrast,medium_contrast=medium_contrast,high_contrast=high_contrast)
		#data_averaged = self.average_over(data[contrast],time=False,channels=False,trials=True)
		data = data[contrast]
		if clip:
			data=np.clip(data,-10,1)
		final_data = np.nanstd(data,2)
		if baseline:
			electrode_baseline = np.nanmean(final_data[0:self.stimulus_offset],0)
			final_data = final_data/electrode_baseline
		if ao_channels:
			final_data = np.nanmean(final_data,1)
		self._core_figure_1(final_data,title,smooth,show,ylabel=ylabel)
		return final_data 

	def plot_figure_6(self,data_type,contrast=0,channel=0,mmin=-0.3,mmax=1,array_size=1000,smoothing=30):
		if data_type in ("false_alarm","correct_rejections"):
			contrast = 0
			print("No contrast for false alarms or correct rejections")
		a = self.get_data(data_type)[contrast][:,channel,:]
		step = (mmax-mmin)/array_size
		vect = np.arange(mmin,mmax+step,step)[:array_size+1]
		"""
		c = np.zeros((763,array_size))

		for i,x in enumerate(a):
			a,b = np.histogram(x,vect)
			c[i] = a
		d = self.array_smoother(np.transpose(c),smoothing,2)
		"""
		d = self.get_smoothed_hist(a,smoothing,array_size,mmin,mmax)
		fig = self._core_figure_2(d,"Distribution over trials","MUA",vect)
		iplot(fig, filename='basic-line')
		return d

	
	def _core_figure_1(self,full_data,title,smooth,show,time_window=(0,None),ylabel="Multi Unie Activity"):
		if time_window[1] is None:
			time_window = (time_window[0],len(self.get_data("time")))
		dim = len(full_data.shape)
		if type(full_data)==np.float64:
			print("Couldn't print the graph, empty data")
			return
		self._loop_figure_1(full_data,dim-1,smooth,time_window)
		plt.title(title)
		plt.xlabel("Time (ms)")
		plt.ylabel(ylabel)
		if show:
			plt.show()

	def _loop_figure_1(self,data,i,smooth,time_window):
		if i == 0:
			data = self.smoother(data,smooth)
			plt.plot(self.get_data("time")[time_window[0]:time_window[1]],data)
		else:
			for j in range(len(data[0])):
				self._loop_figure_1(data[:,j],i-1,smooth,time_window)
	
	def _core_figure_2(self,full_data,title,ylabel,yaxis=None):
		timing_step = self.get_data("time")
		if yaxis is None:
			yaxis = range(len(full_data[0]))
		data = [
		go.Heatmap(z=full_data,
			x=np.asarray(timing_step),
			y=np.asarray(yaxis),
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
				try:
					data = np.nanmean(data,2)
				except np.AxisError:
					print(data.shape)
					print("Warning, data shape unexpected")
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
		#TO BE FIXED, nice way to smooth still have edge effects
		box = np.ones(window)/window
		result = np.convolve(data,box,mode='same') 
		return result


	def smoother2(self,data,window=10):
		#TO BE FIXED, nice way to smooth still have edge effects
		box = np.ones(window)/window
		result = np.convolve(data,box,mode='same') 
		return result

	def array_smoother(self,data,window=50,b=0):
		#Function smoothing all the vector of a 2d numpy array
		if b == 0:
			smoothed_data = np.apply_along_axis(lambda x : self.smoother(x,window),0,data)
		else:
			smoothed_data = np.apply_along_axis(lambda x : self.smoother2(x,window),0,data)
			
		return smoothed_data

	def remove_corrupted_channels(self,data_type):
		for i in range(len(self.data[data_type])):
			data_averaged = self.average_over(self.data[data_type][i],time=True,channels=False,trials=True)
			corrupted_channels = np.isnan(data_averaged)
			try:
				self.data[data_type][i] = self.data[data_type][i][:,~corrupted_channels,:]
			except IndexError:
				self.data[data_type][i] = self.data[data_type][i][:,~corrupted_channels]
				

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
	
	def get_contrast_number(self):
		return len(self.get_data("contrast"))
	
	def get_channel_number(self):
		return len(self.get_data("false_alarm")[0][0])

	def get_time_step(self):
		t = self.get_data("time")
		return t[1]-t[0]

	def get_snr(self,data_type,talkative=False):
		#Ratio is computed for seen trials
		seen_data = self.get_data("seen")
		result = []
		if data_type == "contrast":
			item_number = self.get_contrast_number()
			foo = lambda i : self.average_over(seen_data[i],time=False,channels=True,trials=True)
		elif data_type == "channel":
			item_number = self.get_channel_number()
			foo = lambda i : self.average_over(seen_data,time=False,channels=False,trials=True,contrast=True)[:,i]
		else:
			print("data type not recognized")
			return
		for i in range(item_number):
			averaged_data = foo(i)
			noise_amplitude = np.std(averaged_data[:self.stimulus_offset])
			signal_peak = np.max(averaged_data[self.stimulus_offset:self.stimulus_end])
			snr = signal_peak/noise_amplitude
			if talkative:
				print("SNR for " + data_type + " #" + str(i) + " : " + str(round(snr,2)))
			result.append(snr)	
		return result	
			
	def get_smoothed_hist(self,data,smoothing,array_size,mmin,mmax):

		step = (mmax-mmin)/array_size
		vect = np.arange(mmin,mmax+step,step)[:array_size+1]

		#Building histogram
		c = np.zeros((763,array_size))
		for i,x in enumerate(data):
			a,b = np.histogram(x,vect)
			c[i] = a

		#Smoothing
		d = self.array_smoother(np.transpose(c),smoothing,2)
		return d
