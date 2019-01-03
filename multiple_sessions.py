




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

from data_session import Session
from abstract_session import AbstractSession

import os
import re





class MultipleSessions(AbstractSession):


	def __init__(self,cortical_area=None,file_names=[],directory_path="data/"):
		if cortical_area is not None:
			file_names = []
			files = os.listdir("./"+directory_path)
			for file_name in files:
				pattern = re.compile('^'+cortical_area+'_SESSION_*')
				if pattern.match(file_name):
					file_names.append(file_name)
		self.sessions = list(map(lambda x : Session(x,directory_path),file_names))
		self.data = {}
		self.time_steps = None
		self.get_data("time")
		





	def get_data(self,data_type,low_contrast=True,medium_contrast=True,high_contrast=True,average_over_trials=False):
		if data_type == "time":
			if self.time_steps is not None:
				return self.data["time"]
			else:
				for i in range(1,len(self.sessions)+1):
					session = self.sessions[-i]
					time_data = session.get_data("time")
					if self.time_steps is None:
						self.data["time"] = time_data
						self.time_steps = len(time_data)
					else:
						if len(time_data) != self.time_steps:
							print("Session ignored, different time scale")
							self.sessions.pop(-i)
							
				
		else:
			data = np.asarray(list(map(lambda x : x.get_data(data_type,low_contrast,medium_contrast,high_contrast,average_over_trials),self.sessions)))
			self.data[data_type] = np.concatenate(data)
		return self.data[data_type]

