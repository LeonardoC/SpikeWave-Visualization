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


# Import packages
from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from linelength_event_detector.lleventdetector import *
from linelength_event_detector.lltransform import *
import json
from process_data import *
from scipy.signal import find_peaks, peak_prominences

# Incorporate data
data = pd.read_csv(r"C:\Users\L03109567\Desktop\O4-E10,112,13.csv", 
                   header = None)
data = data.to_numpy()
data = data.transpose()

channels = data.shape[0]
mean_prominences = []
mean_prominences_reflected = []

for channel in range(channels):
    peaks, _ = find_peaks(-data[channel])
    mean_prominence = peak_prominences(-data[channel], peaks)[0].mean()
    mean_prominences.append(mean_prominence)
    # For reflected channels
    peaks, _ = find_peaks(data[channel])
    mean_prominence = peak_prominences(data[channel], peaks)[0].mean()
    mean_prominences_reflected.append(mean_prominence)

def main(data, subject, sfx, mel, llw, prc):
    events = [[],[]]
    subject_events = detect_events(data, subject, sfx, mel, llw, prc)
    subject_windows, subject_channels = split_events(subject_events)
    events[0] += subject_windows
    events[1] += subject_channels
    return events


sfx = 500 # Sampling frequency
mel = 500 # Minimum event length (milliseconds)
prc = 99
llw = 1
subject = 0
windows = main(data, subject, sfx, mel, llw, prc)
# graphs = generate_graphs(data)

# Initialize the app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.H1("Detector de Espigas"),
    dcc.Dropdown(id = "input_subject",
       options=[
          {'label': 'Rata 1', 'value': 0},
          {'label': 'Rata 2', 'value': 1},
          {'label': 'Rata 3', 'value': 2},
          {'label': 'Rata 4', 'value': 3}
          ],
          value=0
          ),
    html.Div([html.Span("Prc:"),
              dcc.Input(id='input_prc', type='number', value = 99), 
              html.Span("LLW:"),
              dcc.Input(id='input_llw', type='number', value = 1)]),
    html.Hr(),
    dcc.Slider(1, len(windows[0]), 1, value=1, marks=None,
               tooltip={"placement": "bottom", "always_visible": True},
               id="input_window"
               ),
    html.Div([
        html.Div([html.Span("Reflejar canal?"),
                  dcc.Checklist(id = "input_reflect",options=[1, 2], value=[], inline=True)],
                  style={'display': 'inline-block'})
        # html.Div([html.Span("Umbral: "),
        #           dcc.Input(id='input_picos', type='number', value = 0.1)],
        #           style={'display': 'inline-block', 'margin-left': '2em'})
    ]),
    html.Div([
        html.P("", id = "output_tiempo"),
        html.P("", id = "output_duracion"),
        html.P("", id = "output_picos"),
        html.P("", id = "output_frecuencia")
    ]),
    dcc.Graph(figure={}, id='controls-and-graph'),
    
    dcc.Store(id='windows', data=json.dumps(windows))
])


@callback(
    Output('windows', 'data'),
    Output(component_id='controls-and-graph', component_property='figure',
           allow_duplicate=True),
    Output(component_id='output_tiempo', component_property='children', allow_duplicate = True),
    Output(component_id='output_duracion', component_property='children', allow_duplicate = True),
    Output(component_id='output_picos', component_property='children', allow_duplicate = True),
    Output(component_id='output_frecuencia', component_property='children', allow_duplicate = True),
    Output(component_id='input_window', component_property='min'),
    Output(component_id='input_window', component_property='max'),
    Output(component_id='input_window', component_property='value'),
    Output(component_id='input_reflect', component_property='value'),
    Input(component_id='input_subject', component_property='value'),
    Input(component_id='input_prc', component_property='value'),
    Input(component_id='input_llw', component_property='value'),
    prevent_initial_call=True
)
def update_graph(subject, prc, llw):
   windows = main(data, subject, sfx, mel, llw, prc)
   windows_json = json.dumps(windows)
   fig, tiempo, duracion, picos, frecuencia= update_window(windows_json, 1, [], subject)
   #print(len(windows[0]), len(windows[1]))
   min = 1
   max = len(windows[0])
   return windows_json, fig, tiempo, duracion, picos, frecuencia, min, max, 1, []


@callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    Output(component_id='output_tiempo', component_property='children'),
    Output(component_id='output_duracion', component_property='children'),
    Output(component_id='output_picos', component_property='children'),
    Output(component_id='output_frecuencia', component_property='children'),
    Input('windows', 'data'),
    Input(component_id='input_window', component_property='value'),
    Input(component_id='input_reflect', component_property='value'),
    Input(component_id='input_subject', component_property='value')
)
def update_window(windows_json, window, reflect, subject):
    windows = json.loads(windows_json)
    if not window:
       window = 0
    else:
        window -= 1
    channels = windows[1][window]
    channels = sorted([int(channel) for channel in channels])

    nsamples = data.shape[1]
    time = [i / sfx for i in range(nsamples)]

    start = int(windows[0][window][0])
    end = int(windows[0][window][1]) + 1

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Espiga detectada" if 1 in channels else "SIN espiga",
                        "Espiga detectada" if 2 in channels else "SIN espiga"))
    n_hills = []    
    for i in range(2):
        channel = subject*2 + i
        fig_data = data[channel][start:end]
        if reflect and (i + 1) in reflect:
            fig_data = -fig_data
            peaks, _ = find_peaks(-fig_data, prominence = mean_prominences_reflected[channel])
        else:
            peaks, _ = find_peaks(-fig_data, prominence = mean_prominences[channel])

        n_hills.append(len(peaks))

        fig.append_trace(go.Scatter(
            x=time[start:end],
            y=fig_data,
            name=f'Channel {i + 1}',
            line=dict(color='blue' if i == 0 else 'red')
        ), row=i + 1, col=1)

        fig.append_trace(go.Scatter(
            x=[time[start:end][i] for i in peaks],
            y=fig_data[peaks],
            name=f'Peaks {i + 1}',
            mode='markers',
            visible='legendonly'
        ), row=i + 1, col=1)
        

    fig.update_layout(height=300*2, showlegend= True)
    fig.update_xaxes(range=(time[start], time[end]), row = 1, col = 1)
    fig.update_xaxes(range=(time[start], time[end]), row = 2, col = 1)
    timespan = time[end] - time[start]
    return (
        fig, 
        f'Tiempo: [{time[start]:.2f}, {time[end]:.2f}]', 
        f'Duración: {timespan:.2f}',
        f'Picos: {n_hills}',
        f'Frecuencia: [{n_hills[0] / timespan:.2f}, {n_hills[1] / timespan:.2f}]')

# def update_graph(window):
#     if not window:
#        window = 0
#     channels = windows[1][window].strip(',').split(',')
#     channels = [int(channel) for channel in channels]
#     nchannels = len(channels)
    
#     nsamples = data.shape[1]
#     time = [i / sfx for i in range(nsamples)]

#     start = int(windows[0][window][0])
#     end = int(windows[0][window][1])

#     fig = make_subplots(rows=nchannels, cols=1)
#     for i in range(nchannels):
#         graph = graphs[channels[i]]
#         fig.append_trace(graph, row=i + 1, col=1)
#     fig.update_xaxes(range = [time[start], time[end]])
#     fig.update_layout(height=300*nchannels, showlegend= True)
#     return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
