import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import numpy as np
import sys, os

class PlotsWComparison:
  def __init__(self, plot_refresh_rate, visible_duration, data_per_second):
    self.plot_refresh_rate = plot_refresh_rate
    self.visible_duration = visible_duration
    self.data_per_second = data_per_second
    self.num_samples_per_call = int(self.data_per_second*self.plot_refresh_rate) # #number of samples coming at each call to plot_live function

  def init_common_params (self, y_label):
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#bcbd22', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#17becf', '#d62728']) 
    mpl.rcParams['axes.edgecolor'] = 'black'
    mpl.rcParams['axes.linewidth'] = 0.5

    self.fig = plt.figure()
    self.ax1 = self.fig.add_subplot(1,1,1)
    self.ax1.set_xlabel("Time", labelpad=2, fontsize=7)
    self.ax1.set_ylabel(y_label, labelpad=2, fontsize=7)
    self.ax1.set_xlim(-2, self.visible_duration+2)
    self.ax1.tick_params(axis='both', which='major', labelsize=6)
    self.ax1.grid(False)

    self.xs = np.zeros(1)
    self.ys = np.zeros((2,1))

  def term_common_params (self):
    self.fig.set_size_inches(1.75, 1.5) #width, height
    plt.tight_layout()

  def plot_live(self, i, start_time, meas_line, pred_line, meas_list, pred_list, plot_name):
    #Here vel_meas_list and vel_pred_list contains lists of shape (plot_time/plot_refresh_rate,). This is (3000,) for 300 seconds of data at 10 hz.
    t0 = time.time()
    cur_frame = abs(int((t0-start_time)/self.plot_refresh_rate*self.num_samples_per_call))
    
    meas_data = np.mean(meas_list[cur_frame:cur_frame+self.num_samples_per_call])
    pred_data = np.mean(pred_list[cur_frame:cur_frame+self.num_samples_per_call])
    
    if (i%int(self.visible_duration/self.plot_refresh_rate) == 0): #Reset data once the period is filled.
      self.xs = np.zeros(1)
      self.ys = np.array([[meas_data], [pred_data]])

    self.xs = np.append(self.xs, i%int(self.visible_duration/self.plot_refresh_rate) * self.plot_refresh_rate)
    self.ys = np.append(self.ys, np.zeros((2,1)), axis=1)
    self.ys[0,-1] = meas_data
    self.ys[1,-1] = pred_data
    
    meas_line.set_xdata(self.xs)
    pred_line.set_xdata(self.xs)
    meas_line.set_ydata(self.ys[0])
    pred_line.set_ydata(self.ys[1])

    return meas_line, pred_line
  
class AirspeedPlot (PlotsWComparison):
  def __init__(self, plot_refresh_rate, visible_duration, data_per_second):
    super().__init__(plot_refresh_rate, visible_duration, data_per_second)
  
  def init_common_params(self, y_label):
    super().init_common_params(y_label)
  
  def term_common_params(self):
    super().term_common_params()

  def plot_airspeed_wcomparison(self):
    self.ax1.set_ylim(-2, 22) #This scale is m/s
    self.ax1.set_xticklabels([])
    self.vel_meas_line, = self.ax1.plot(self.xs, self.ys[0], linewidth=1, animated=True, label="Measured") 
    self.vel_pred_line, = self.ax1.plot(self.xs, self.ys[1], linewidth=1, animated=True, label="Predicted")

  def plot_airspeed_live(self, i, vel_meas_list, vel_pred_list, start_time):
    vel_meas_list = np.asarray(vel_meas_list)
    vel_pred_list = np.asarray(vel_pred_list)
    self.vel_meas_line, self.vel_pred_line = super().plot_live(i, start_time, self.vel_meas_line, self.vel_pred_line, vel_meas_list, vel_pred_list, 'airspeed')
    return list((self.vel_meas_line, self.vel_pred_line))
  
class AoaPlot (PlotsWComparison):
  def __init__(self, plot_refresh_rate, visible_duration, data_per_second):
    super().__init__(plot_refresh_rate, visible_duration, data_per_second)
  
  def init_common_params(self, y_label):
    super().init_common_params(y_label)
  
  def term_common_params(self):
    super().term_common_params()

  def plot_aoa_wcomparison(self):
    self.ax1.set_ylim(-2, 18) #This scale is degree
    self.ax1.set_xticklabels([])
    self.aoa_meas_line, = self.ax1.plot(self.xs, self.ys[0], linewidth=1, animated=True, label="Measured") 
    self.aoa_pred_line, = self.ax1.plot(self.xs, self.ys[1], linewidth=1, animated=True, label="Predicted")

  def plot_aoa_live(self, i, aoa_meas_list, aoa_pred_list, start_time):
    aoa_meas_list = np.asarray(aoa_meas_list)
    aoa_pred_list = np.asarray(aoa_pred_list)
    self.aoa_meas_line, self.aoa_pred_line = super().plot_live(i, start_time, self.aoa_meas_line, self.aoa_pred_line, aoa_meas_list, aoa_pred_list, 'aoa')
    return list((self.aoa_meas_line, self.aoa_pred_line))