




#################################################################################################
## Import Section
from scipy import io
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from math import isnan
import numpy.ma as ma

import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

from data_session import Session
from abstract_session import AbstractSession

import tool_box

import os
import re





class MultipleSessions(AbstractSession):


	def __init__(self,cortical_area=None,file_names=[],directory_path="data/",excluded=[],normalized=False,monkey="",auto_exclude=False,new_norm=False):
		if auto_exclude:
			excluded_files = tool_box.excluded_files(cortical_area)
			excluded.extend(excluded_files)
		if cortical_area is not None:
			files_to_add = tool_box.file_list((cortical_area,),directory_path,monkey)
			for file in files_to_add:
				if file not in excluded:
					file_names.append(file)
		self.sessions = list(map(lambda x : Session(x,directory_path,normalized,new_norm=new_norm),file_names))
		self.data = {}
		if normalized:
			self.get_data("seen",normalized=True)
			self.get_data("missed",normalized=True)
			self.get_data("correct_rejections",normalized=True)
			self.get_data("false_alarm",normalized=True)
		self.time_steps = None
		self.get_data("time")
		file_names.clear()
		excluded.clear()
		





	def get_data(self,data_type,low_contrast=True,medium_contrast=True,high_contrast=True,average_over_trials=False,normalized=False):
		if data_type == "time":
			if self.time_steps is not None:
				return self.data["time"]
			else:
				d = 0
				for i in range(1,len(self.sessions)+1):
					session = self.sessions[-i+d]
					time_data = session.get_data("time")
					if self.time_steps is None:
						self.data["time"] = time_data
						self.time_steps = len(time_data)
					else:
						if len(time_data) != self.time_steps:
							print("Session ignored, different time scale")
							self.sessions.pop(-i+d)
							d += 1
							
				
		else:
			#data = np.asarray(list(map(lambda x : x.get_data(data_type,low_contrast,medium_contrast,high_contrast,average_over_trials,normalized),self.sessions)))
			#self.data[data_type] = np.concatenate(data)
			self.data[data_type] = [item for session in self.sessions for item in session.get_data(data_type,low_contrast,medium_contrast,high_contrast,average_over_trials,normalized)]
		return self.data[data_type]



