
#################################################################################################
## Import Section
from scipy import io
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from math import isnan
import numpy.ma as ma
import seaborn as sns
import pandas as pd

from timeit import time

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)








class AbstractSession():

	stimulus_offset = 228 #Magic number defining the stimulus offset
	stimulus_end = 428 #Magic number defining the end of the stimulus peak response

	
#################################################################################################
## Ploting functions

	def plot_figure_1(self,data_type,contrast=0,title=None,ao_channels=True,ao_trials=False,ao_contrast=False,low_contrast=True,medium_contrast=True,high_contrast=True,smooth=1,show=True,clip=False,channel=None,talkative=True,label=None,time_window=(0,None),trial_slice=None):
		none_slice = slice(None,None,None)
		if trial_slice is None:
			trial_slice = none_slice
		if data_type in ("false_alarm","correct_rejections"):
			contrast = 0
			if talkative:
				print("No contrast for false alarms or correct rejections")
		if title is None:
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
		#data = np.array(data)
		for c in range(len(data)):
			data[c] = self.array_smoother(data[c][none_slice,none_slice,trial_slice],smooth,b=1)
		if ao_contrast:
			data_averaged = self.average_over(data,time=False,channels=True,trials=True,contrast=True)
		else:
			if channel is not None:
				data = data[contrast][:,channel,:]
				#Must make an ugly trick in this case
				if ao_trials:
					data_averaged = np.nanmean(data,1)
			else:
				data_averaged = self.average_over(data[contrast],time=False,channels=ao_channels,trials=ao_trials)
		if clip:
			data_averaged=np.clip(data_averaged,-10,1)	
		if show:
			self._core_figure_1(data_averaged[time_window[0]:time_window[1]],title,1,show,label=label,time_window=time_window)
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

	def plot_figure_4(self,data_type,title="Title not set yet",smooth=1,clip=False,ylabel="",sort=True,best_channel=False,bound=None,add_false_alarms=False,add_correct_rejections=False,add_absent=False):
		full_data = self.get_data(data_type)
		contrast_number = len(self.get_data(data_type))
		display_order = list(range(contrast_number))
		if sort:
			display_order = self.get_display_order()
		else:
			display_order = list(range(contrast_number))
		full_data = [full_data[i] for i in display_order]
		n = 0
		if add_absent:
			full_data = list(self.get_data("absent")) + full_data
			n += len(self.get_data("absent"))
		else:
			if add_false_alarms:
				full_data = list(self.get_data("false_alarm")) + full_data
				n += len(self.get_data("false_alarm"))
			if add_correct_rejections:
				full_data = list(self.get_data("correct_rejections")) + full_data
				n += len(self.get_data("correct_rejections"))
		contrasts = [self.get_data("contrast")[i] for i in display_order]
		contrasts = list(range(len(full_data))) #TO BE FIXED
		if best_channel:
			best_channels = [self.get_data("best_channel")[i] for i in display_order]
		data_averaged = []
		for i,data in enumerate(full_data):
			if best_channel:
				temp = self.average_over(data,time=False,channels=False,trials=True)
				temp = temp[:,best_channels[i]]
			else:
				temp = self.average_over(data,time=False,channels=True,trials=True)
			temp_smoothed = self.smoother(temp,smooth)
			data_averaged.append(temp_smoothed)#self.average_over(data,time=False,channels=True,trials=True))
		data_averaged = np.asarray(data_averaged)
		final_data = data_averaged
		a = len(self.get_data("false_alarm"))
		b,c,d = self.get_display_bound() 
		lines = [n+b,n+c,n+d]
		if clip:
			final_data = np.clip(final_data,-10,1)
		fig = self._core_figure_2(final_data,title,ylabel,contrasts,bound=bound,lines=lines)
		iplot(fig, filename='basic-line')
		return final_data

	def plot_figure_5(self,data_type,contrast=None,title="Title not set yet",ao_channels=False,low_contrast=True,medium_contrast=True,high_contrast=True,smooth=1,show=True,clip=False,baseline=False,ylabel="Variance",smoothing_window=1,best_channel=False,time_window=(0,None),label=None):
		## Plotting the variance at each time spot (allow smoothing before computing variance)
		if contrast is not None:
			data = [data[contrast]]
		else:
			data = self.get_data(data_type,low_contrast=low_contrast,medium_contrast=medium_contrast,high_contrast=high_contrast)
		if clip:
			data=[np.clip(d,-10,1) for d in data]
		variances = []
		best_channels = self.get_data("best_channel",low_contrast=low_contrast,medium_contrast=medium_contrast,high_contrast=high_contrast)
		for i,d in enumerate(data):
			if best_channel:
				temp = self.array_smoother(d[:,best_channels[i],:],smoothing_window,b=1)
				temp = np.nanstd(temp,1)
			else:
				temp = self.array_smoother(d,smoothing_window,b=1)
				temp = np.nanstd(temp,(1,2))
			variances.append(temp)
		final_data = np.nanmean(variances,0)
		if baseline:
			electrode_baseline = np.nanmean(final_data[0:self.stimulus_offset],0)
			final_data = final_data/electrode_baseline
		self._core_figure_1(final_data[time_window[0]:time_window[1]],title,smooth,show,ylabel=ylabel,time_window=time_window,label=label)
		return final_data 

	def plot_figure_6(self,data_type,contrast=None,channel=None,mmin=-0.3,mmax=1,array_size=1000,smoothing=30,low_contrast=True,medium_contrast=True,high_contrast=True,time_smoothing_window=1,timing_window=[0,None],title="Distibution over trials",three_d=False,trials_slice=None,plot_average=False,show=True):
		none_slice = slice(None,None,None)
		if trials_slice is None:
			trials_slice = slice(None,None,None)
		if data_type in ("false_alarm","correct_rejections"):
			contrast = 0
			print("No contrast for false alarms or correct rejections")
		if contrast is not None:
			a = [self.get_data(data_type)[contrast]]
		else:
			a = self.get_data(data_type,low_contrast=low_contrast,medium_contrast=medium_contrast,high_contrast=high_contrast)	
		if channel is not None:
			a = list(map(lambda x : x[:,channel:channel+1,:],a))

		for c in range(len(a)):
			a[c] = self.array_smoother(a[c][none_slice,none_slice,trials_slice],time_smoothing_window,b=1)

		a = self.merge_contrast_and_channels(a)
		timing_slice = slice(timing_window[0],timing_window[1],None)

		if plot_average:
			plt.plot(np.nanmean(a,1)[timing_slice])
			plt.show()
		step = (mmax-mmin)/array_size
		vect = np.arange(mmin,mmax+step,step)[:array_size+1]
		d = self.get_smoothed_hist(a,smoothing,array_size,mmin,mmax)
		if show:
			fig = self._core_figure_2(d,title,"MUA",vect,timing_window=timing_window,three_d=three_d)
			iplot(fig, filename='basic-line')
		return d

	def plot_figure_6_bis(self,data_type,contrast=None,channel=None,mmin=-0.3,mmax=1,array_size=1000,smoothing=30,low_contrast=True,medium_contrast=True,high_contrast=True,time_smoothing_window=1,timing_window=[0,None],title="Distibution over trials",three_d=False,trials_slice=None,show=True):
		none_slice = slice(None,None,None)
		if trials_slice is None:
			trials_slice = slice(None,None,None)
		if data_type in ("false_alarm","correct_rejections"):
			contrast = 0
			print("No contrast for false alarms or correct rejections")
		if contrast is not None:
			a = [self.get_data(data_type)[contrast]]
		else:
			a = self.get_data(data_type,low_contrast=low_contrast,medium_contrast=medium_contrast,high_contrast=high_contrast)	
		if channel is not None:
			a = list(map(lambda x : x[:,channel:channel+1,:],a))

		for c in range(len(a)):
			a[c] = self.array_smoother(a[c][none_slice,none_slice,trials_slice],time_smoothing_window,b=1)

		a = self.merge_contrast_and_channels(a)
		timing_slice = slice(timing_window[0],timing_window[1],None)
		return np.nanmean(a,1)[timing_slice]

	def plot_figure_7(self,data_type,title="Title not set yet",smooth=1,clip=False,ylabel="",sort=True,best_channel=False,bound=None,add_false_alarms=False,add_correct_rejections=False,add_absent=False,smoothing_window=50):
		full_data = self.get_data(data_type)
		contrast_number = len(self.get_data(data_type))
		if sort:
			display_order = self.get_display_order()
		else:
			display_order = list(range(contrast_number))
		full_data = [full_data[i] for i in display_order]
		n = 0
		if add_absent:
			full_data = list(self.get_data("absent")) + full_data
			n += len(self.get_data("absent"))
		else:
			if add_false_alarms:
				full_data = list(self.get_data("false_alarm")) + full_data
				n += len(self.get_data("false_alarm"))
			if add_correct_rejections:
				full_data = list(self.get_data("correct_rejections")) + full_data
				n += len(self.get_data("correct_rejections"))
		contrasts = [self.get_data("contrast")[i] for i in display_order]
		contrasts = list(range(len(full_data))) #TO BE FIXED
		if best_channel:
			best_channels = [0]*n + [self.get_data("best_channel")[i] for i in display_order]
		data_averaged = []
		for i,data in enumerate(full_data):
			if best_channel:
				temp = data[:,best_channels[i],:]
				temp = self.array_smoother(temp,smoothing_window,b=1)
				temp = np.nanstd(temp,1)
			else:
				#temp = self.average_over(data,time=False,channels=True,trials=False)
				temp = self.array_smoother(data,smoothing_window,b=1)
				temp = np.nanstd(temp,(1,2))
			temp_smoothed = self.smoother(temp,smooth)
			data_averaged.append(temp_smoothed)
		data_averaged = np.asarray(data_averaged)
		final_data = data_averaged
		if clip:
			final_data = np.clip(final_data,-10,1)
		b,c,d = self.get_display_bound() 
		lines = [n+b,n+c,n+d]
		fig = self._core_figure_2(final_data,title,ylabel,contrasts,bound=bound,lines=lines)
		iplot(fig, filename='basic-line')
		return final_data

	def plot_figure_8(self,low_contrast=True,medium_contrast=True,high_contrast=True,studied_window=(328,428),bins=None,xlim=(-0.5,1.75),ylim=(0,3),title="",plot_absent=False):
		kwargs= {
			"low_contrast":low_contrast,
			"medium_contrast":medium_contrast,
			"high_contrast":high_contrast,
		    }

		ax = plt.axes((0,0,1,1))

		self.plot_distrib(label="Missed",bins=bins,ax=ax,studied_window=studied_window,data_type="missed",**kwargs)
		self.plot_distrib(label="Seen",bins=bins,ax=ax,studied_window=studied_window,data_type="seen",**kwargs)
		self.plot_distrib(label="Present",bins=bins,ax=ax,studied_window=studied_window,data_type="present",**kwargs)
		if plot_absent:
			self.plot_distrib(label="Absent",bins=bins,ax=ax,studied_window=studied_window,data_type="absent")

		ax.legend()
		ax.set(xlabel="Mean response (MUA)",title=title,xlim=xlim,ylim=ylim)
		plt.show()


	def plot_figure_9(self,studied_window=(328,428),bins=None,xlim=(-0.5,1.75),ylim=(0,3),title="",plot_absent=False):
		ax = plt.axes((0,0,1,1))

		kwargs = {
			"low_contrast":True,
			"medium_contrast":False,
			"high_contrast":False,
		}
		self.plot_distrib(label="Low contrast",bins=bins,ax=ax,studied_window=studied_window,data_type="present",**kwargs)
		kwargs = {
			"low_contrast":False,
			"medium_contrast":True,
			"high_contrast":False,
		}
		self.plot_distrib(label="Medium contrast",bins=bins,ax=ax,studied_window=studied_window,data_type="present",**kwargs)
		kwargs = {
			"low_contrast":False,
			"medium_contrast":False,
			"high_contrast":True,
		}
		self.plot_distrib(label="High contrast",bins=bins,ax=ax,studied_window=studied_window,data_type="present",**kwargs)
		if plot_absent:
			self.plot_distrib(label="Stimulus absent",bins=bins,ax=ax,studied_window=studied_window,data_type="absent")

		ax.legend()
		ax.set(xlabel="Mean response (MUA)",title=title,xlim=xlim,ylim=ylim)
		plt.show()

	def plot_figure_10(self,studied_window=(328,428),title=""):
		so,se = studied_window
		kwargs = {
		    "low_contrast":True,
		    "medium_contrast":False,
		    "high_contrast":False,
		}
		a = self.get_var(so,se,"present",**kwargs)

		kwargs = {
		    "low_contrast":False,
		    "medium_contrast":True,
		    "high_contrast":False,
		}
		b = self.get_var(so,se,"present",**kwargs)

		kwargs = {
		    "low_contrast":False,
		    "medium_contrast":False,
		    "high_contrast":True,
		}
		c = self.get_var(so,se,"present",**kwargs)

		l = [[a,"low"],[b,"medium"],[c,"high"]]
		df = pd.DataFrame(l,columns=("Variance","Contrast"))
		
		ax = sns.catplot(y="Variance",x="Contrast",data=df,kind='bar')
		
		return df

	def plot_figure_11(self,data_type,smoothing=100,mmin=-0.3,mmax=0.6,channel=0,time_smoothing_window=50,low_contrast=True,medium_contrast=True,high_contrast=True,timing_window=[0,None],plot_absent=False,three_d=False,title=""):
		kwargs = {
			'smoothing':smoothing,
			'mmin':mmin,
			'mmax':mmax,
			'channel':channel,
			'time_smoothing_window':time_smoothing_window,
			'timing_window':timing_window,
			'three_d':three_d,
		}
		cwargs = {
		    	'low_contrast':True,
		    	'medium_contrast':False,
		    	'high_contrast':False,
			'title':title+" (Low contrast trials)",
		}
		self.plot_figure_6(data_type,**{**kwargs,**cwargs})
		cwargs = {
		    	'low_contrast':False,
		    	'medium_contrast':True,
		    	'high_contrast':False,
			'title':title+" (Medium contrast trials)",
		}
		self.plot_figure_6(data_type,**{**kwargs,**cwargs})
		cwargs = {
		    	'low_contrast':False,
		    	'medium_contrast':False,
		    	'high_contrast':True,
			'title':title+" (High contrast trials)",
		}
		self.plot_figure_6(data_type,**{**kwargs,**cwargs})
		if plot_absent:
			self.plot_figure_6("absent",**kwargs,title=title+" (Stimulus absent trials)")


	def plot_distrib(self,data_type,label=None,bins=None,studied_window=(228,328),ax=None,**kwargs):
		so,se = studied_window
		if bins is None:
			bins = np.arange(-0.5, 1.5, 0.05)

		a = list(map(lambda x : x[so:se,:,:],self.get_data(data_type,**kwargs)))
		b = list(map(lambda x : self.average_over(x,time=True,channels=True),a))
		if len(b) !=0:
			c = np.concatenate(b)
		else:
			c = np.empty((0))
		c = np.clip(c[~np.isnan(c)],-10,3)
		try:
			result = sns.distplot(c,label=label,bins=bins,ax=ax).get_lines()[0].get_data()
			return result
		except KeyError:
			return np.empty((0))	


	def plot_contrast_vs_contrast_performance(self):
		plt.plot(self.get_data("contrast"),self.get_data("contrast_performance"))
		plt.title("Contrast vs Contrast Performance")
		plt.xlabel("Contrast value")
		plt.ylabel("Contrast performance") 
		plt.show()

	def plot_average_MUA(self,fig_size=12,smooth=1,clip=False,low_contrast=True,medium_contrast=True,high_contrast=True,cc=None,subplot=True,title=None):
		# cc is the contrast category
		if cc is not None:
			low_contrast,medium_contrast,high_contrast = self.compute_contrast_category(cc)
		if subplot:
			fig=plt.figure(figsize=(4, 4))
			fig.set_figheight(fig_size)
			fig.set_figwidth(fig_size)
		columns = 2
		rows = 2
		data_types = ("seen","false_alarm","missed","correct_rejections")
		for i in range(1,columns*rows+1):
			if subplot:
				fig.add_subplot(rows, columns,i)
			kwargs = {
				"ao_channels":True,
				"ao_trials":True,
				"ao_contrast":True,
				"show":False,
				"talkative":False,
				"smooth":smooth,
				"clip":clip,
				"low_contrast":low_contrast,
				"medium_contrast":medium_contrast,
				"high_contrast":high_contrast,
				"label":data_types[i-1],
			}
			if title is not None:
				kwargs["title"] = title
			else:
				kwargs["title"] = data_types[i-1]
			self.plot_figure_1(data_types[i-1],**kwargs)
		plt.legend()
		plt.show()

	def plot_best_snr_MUA(self,fig_size=12):
		channel = self.get_best_snr("channel")
		contrast = self.get_best_snr("contrast")
		fig = plt.figure(figsize=(4,4))
		fig.set_figheight(fig_size)
		fig.set_figwidth(fig_size)
		columns = 2
		rows = 2
		data_types = ("seen","false_alarm","missed","correct_rejections")
		for i in range(1,columns*rows+1):
			fig.add_subplot(rows,columns,i)
			a = self.plot_figure_1(data_types[i-1],contrast=contrast,ao_channels=False,ao_contrast=False,ao_trials=True,show=False,title=data_types[i-1],channel=channel,talkative=False) 
		plt.show()
	
	def plot_trial_number(self):
		hit_trials = list(map(lambda x : len(x[0,0,:]),self.get_data("seen")))
		miss_trials = list(map(lambda x : len(x[0,0,:]),self.get_data("missed")))
		contrast = self.get_data("contrast")
		plt.plot(contrast,hit_trials,label="Hit trials")
		plt.plot(contrast,miss_trials,label="Missed trials")
		plt.title("Trial counts")
		plt.legend()
		plt.show()

	
	def _core_figure_1(self,full_data,title,smooth,show,time_window=(0,None),ylabel="Multi Unie Activity",label=None):
		if time_window[1] is None:
			time_window = (time_window[0],len(self.get_data("time")))
		dim = len(full_data.shape)
		if type(full_data)==np.float64:
			print("Couldn't print the graph, empty data")
			return
		self._loop_figure_1(full_data,dim-1,smooth,time_window,label)
		plt.title(title)
		plt.xlabel("Time (ms)")
		plt.ylabel(ylabel)
		if show:
			if label is not None:
				plt.legend()
			plt.show()

	def _loop_figure_1(self,data,i,smooth,time_window,label):
		if i == 0:
			data = self.smoother(data,smooth)
			plt.plot(self.get_data("time")[time_window[0]:time_window[1]],data,label=label)
		else:
			for j in range(len(data[0])):
				self._loop_figure_1(data[:,j],i-1,smooth,time_window,label)
	
	def _core_figure_2(self,full_data,title,ylabel,yaxis=None,bound=None,lines=[],timing_window=[0,None],three_d=False):
		timing_step = self.get_data("time")
		if timing_window[1] is None:
			timing_window[1] = len(full_data)
		if yaxis is None:
			yaxis = range(len(full_data[0]))
		kwargs = {
			"z":full_data[:,timing_window[0]:timing_window[1]],
			"x":np.asarray(timing_step[timing_window[0]:timing_window[1]]),
			"y":np.asarray(yaxis),
		}
		if bound is not None:
			kwargs["zmin"]=bound[0]
			kwargs["zmax"]=bound[1]
		if three_d:
			data = [
				go.Surface(**kwargs)
			]
		else:
			data = [
			go.Heatmap(**kwargs)
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
			shapes=list(map(lambda x : 
				{
				    'type': 'line',
				    'x0': timing_step[0],
				    'y0': x-0.5,
				    'x1': timing_step[-1],
				    'y1': x-0.5,
				    'line': {
					'color': 'rgb(55, 128, 55)',
					'width': 3,
				    },
				},lines)),
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
			to_mean = []
			if time:
				to_mean.append(0)
			if channels:
				to_mean.append(1)
			if trials:
				to_mean.append(2)
			data = np.nanmean(data,tuple(to_mean))
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
			result.append(np.nanmean(a))
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

	def format_data(self,data_type):
		for i in range(len(self.data[data_type])):
			if len(self.data[data_type][i].shape) == 2:
				self.data[data_type][i] = np.expand_dims(self.data[data_type][i],2)
				

	def get_contrast_value(self,contrast_index,round_number=2):
		contrasts = self.get_data("contrast")
		contrast = contrasts[contrast_index]
		contrast = round(contrast,round_number)
		return contrast

	def get_contrast_index(self,low_contrast=True,medium_contrast=True,high_contrast=True):
		contrast_performance = self.get_data("contrast_performance")
		contrast_category = self.get_data("contrast_category")
		contrast_index = []
		for i in range(len(contrast_performance)):
			if low_contrast and contrast_category[i] == 1:
				contrast_index.append(i)
			elif medium_contrast and contrast_category[i] == 2: 
				contrast_index.append(i)
			elif high_contrast and contrast_category[i] == 3:
				contrast_index.append(i)
		return contrast_index
	
	def get_contrast_number(self):
		return len(self.get_data("contrast"))
	
	def get_channel_number(self):
		return len(self.get_data("seen")[0][0])

	def get_time_step(self):
		t = self.get_data("time")
		return t[1]-t[0]

	def get_step_number(self):
		t = self.get_data("time")
		return len(t)

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
		step_number = self.get_step_number()
		c = np.zeros((step_number,array_size))
		for i,x in enumerate(data):
			a,b = np.histogram(x,vect,density=True)
			c[i] = a

		#Smoothing
		d = self.array_smoother(np.transpose(c),smoothing,2)
		return d

	def get_best_snr(self,data_type):
		return np.argmax(self.get_snr(data_type,False))

	def session_overview(self):
		##############################################
		## Printing snr

		self.get_snr("channel",talkative=True)
		self.get_snr("contrast",talkative=True)
		##############################################
		## Printing contrast vs contrast performance

		self.plot_contrast_vs_contrast_performance()	

		self.plot_trial_number()


		##############################################
		## Plotting average MUA
		
		print("Average MUA")
		self.plot_average_MUA()
		print("Best SNR (contrast and channel) MUA")
		self.plot_best_snr_MUA()

	def get_display_order(self):
		display_order = list(range(len(self.get_data("contrast"))))
		display_order.sort(key=lambda x : 100*self.get_data("contrast_category")[x]+self.get_data("contrast_performance")[x])
		return display_order

	def get_display_bound(self):
		categories = self.get_data("contrast_category")
		categories.sort()
		categories = np.asarray(categories)
		return np.argmin(abs(categories-1)),np.argmin(abs(categories-2)),np.argmin(abs(categories-3))

	def compute_contrast_category(self,cc):
		cc = int(cc)
		low_contrast = (cc % 2 == 1)
		medium_contrast = (int(cc/10) % 2 == 1)
		high_contrast = (int(cc/100) % 2 == 1)
		return low_contrast, medium_contrast, high_contrast



	def get_var(self,so,se,data_type,**kwargs):
		a = list(map(lambda x : x[so:se,:,:],self.get_data(data_type,**kwargs)))
		b = list(map(lambda x : self.average_over(x,time=True,channels=True),a))
		if len(b) !=0:
			c = np.concatenate(b)
		else:
			c = np.empty((0))
		c = np.clip(c[~np.isnan(c)],-10,3)
		return np.var(c)


	def merge_contrast_and_channels(self,data):
		partly_flatten_data = list(map(lambda x : x.reshape(x.shape[0],x.shape[1]*x.shape[2]),data))
		
		if len(partly_flatten_data) != 0:
			fully_flatten_data = np.concatenate(partly_flatten_data,1)
		else:
			fully_flatten_data = np.ones((self.get_step_number(),0)) 

		return fully_flatten_data
