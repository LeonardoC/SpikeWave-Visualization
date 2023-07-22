# EEG Spike-Wave visualization tool
# Copyright (C) 2023  Leonardo Cañete-Sifuentes, Azul Salmerón Ruiz, Lizbeth Becerril Belio 

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Repository: https://github.com/LeonardoC/SpikeWave-Visualization

import pandas as pd
from linelength_event_detector.lleventdetector import *
from linelength_event_detector.lltransform import *
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def detect_hill(x):
    a, b = x
    return a <= 0 and b > 0

def detect_events(data, subject, sfx, mel, llw, prc):
  channel0 = subject * 2
  channel1 = channel0 + 1
  actual = lltransform(data[channel0], sfx, llw)
  events = lleventdetector([actual], sfx, prc, mel, llw)[0]
  channels = [1 for i in range(events.shape[0])]
  channels = np.array(channels, ndmin=2).transpose()
  events = np.hstack((events, channels))
  
  actual = lltransform(data[channel1], sfx, llw)
  new_events = lleventdetector([actual], sfx, prc, mel, llw)[0]
  channels = [2 for i in range(new_events.shape[0])]
  channels = np.array(channels, ndmin=2).transpose()
  new_events = np.hstack((new_events, channels))
  events = np.vstack((events, new_events))
  events = events[events[:, 0].argsort()]
  return events

def split_events(events):
  windows = []
  channels = []
  large_window = False

  windows.append([events[0][0], events[0][1]])
  channels.append([events[0][2],])

  i = 1
  while i < events.shape[0]:
    if events[i][0] > windows[-1][1]:
      windows.append([events[i][0], events[i][1]])
      channels.append([events[i][2],])
    else:
      windows[-1][1] = events[i][1]
      set_channels = set(channels[-1])
      set_channels.add(events[i][2])
      channels[-1] = list(set_channels)
    i += 1

  return windows, channels


def print_results(data, res):
  for i in range(len(res[0])):
    chls = res[1][i]
    chls = [int(i) for i in chls]
    avgs = []
    for chl in chls:
      start =  int(res[0][i][0])
      end = int(res[0][i][1]) + 1
      avg = data[chl, start:end].mean()
      avgs.append(avg)
      time = (res[0][i][1] - res[0][i][0]) / 500
    print(i, res[0][i] / 500, time, "s, ", res[1][i], avgs)
    
def plot_window(data, sfx, windows, window):
    channels = windows[1][window]
    channels = [int(channel) for channel in channels]
    channels

    nchannels = len(channels)
    nsamples = data.shape[1]
    time = [i / sfx for i in range(nsamples)]

    start = int(windows[0][window][0])
    end = int(windows[0][window][1]) + 1
    
    fig = make_subplots(rows=nchannels, cols=1)
    for i in range(nchannels):
        fig.append_trace(go.Scatter(
            x=time[start:end],
            y=data[channels[i]][start:end],
            name= channels[i]
        ), row=i + 1, col=1)
    fig.update_layout(height=300*nchannels, width=600, showlegend= True)
    fig.show()