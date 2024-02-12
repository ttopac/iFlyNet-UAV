import os
import time
import sys
import pathlib
from tkinter import Frame, Label, Canvas
from tkinter import LEFT, RIGHT, BOTTOM, NW
from PIL import ImageTk, Image
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

FILE_PATH = pathlib.Path(__file__).absolute()
GUI_LOC = FILE_PATH.parent.parent
sys.path.insert(0, str(GUI_LOC))

from gui_helpers.plot_helper import AirspeedPlot
from gui_helpers.plot_helper import AoaPlot
from gui_helpers.plot_helper import LiftPlot
from gui_helpers.plot_helper import DragPlot
from gui_helpers.process_video_helper import DrawTKOfflineVideo

class iFlynetWindows(Frame):
  def __init__(self, parent):
    Frame.__init__(self,parent)
    self.parent = parent
    
    self.stall_cond = "No"
    self.est_airspeed = 0
    self.est_aoa = 0
    self.truth_lift_val = 0
    self.truth_drag_val = 0
    self.est_lift_val = 0
    self.est_drag_val = 0
    self.init_angle = 0

  def draw_midrow (self, title_label, img_path):
    img = Image.open(img_path)
    width, height = img.size
    res_width, res_height = int(width/7.6), int(height/7.6)
    img = img.resize((res_width, res_height)) # (height, width)
    imgtk = ImageTk.PhotoImage(img)

    self.midrow_frame_top = Frame(self.parent)
    self.midrow_frame_bottom = Frame(self.parent)
    midrow_label = Label(self.midrow_frame_top, text=title_label, font=("Helvetica", 21, 'bold', 'underline'))
    midrow_label.pack(side=LEFT)
    midrow_legend = Label(self.midrow_frame_top, image=imgtk)
    midrow_legend.image = imgtk
    midrow_legend.pack(side=LEFT)

  def draw_video(self, video_label, video_path, experiment_name):
    self.video = DrawTKOfflineVideo(self.parent, video_label, video_path, experiment_name)
    return self.video

  def draw_airspeed_plot_wcomparison(self, plot_refresh_rate, visible_duration, data_per_second):
    airspeed_plot = AirspeedPlot(plot_refresh_rate, visible_duration, data_per_second)
    airspeed_plot.init_common_params("V (m/s)")
    airspeed_plot.plot_airspeed_wcomparison()
    airspeed_plot.term_common_params()

    self.airspeed_plot_wcomparison_cvs = FigureCanvasTkAgg(airspeed_plot.fig, master=self.parent)
    return airspeed_plot
  
  def draw_aoa_plot_wcomparison(self, plot_refresh_rate, visible_duration, data_per_second):
    aoa_plot = AoaPlot(plot_refresh_rate, visible_duration, data_per_second)
    aoa_plot.init_common_params("AoA (deg)")
    aoa_plot.plot_aoa_wcomparison()
    aoa_plot.term_common_params()

    self.aoa_plot_wcomparison_cvs = FigureCanvasTkAgg(aoa_plot.fig, master=self.parent)
    return aoa_plot
  
  def draw_lift_plot_wcomparison(self, plot_refresh_rate, visible_duration, data_per_second):
    lift_plot = LiftPlot(plot_refresh_rate, visible_duration, data_per_second)
    lift_plot.init_common_params("Lift (lbf)")
    lift_plot.plot_lift_wcomparison()
    lift_plot.term_common_params()

    self.lift_plot_wcomparison_cvs = FigureCanvasTkAgg(lift_plot.fig, master=self.parent)
    return lift_plot
  
  def draw_drag_plot_wcomparison(self, plot_refresh_rate, visible_duration, data_per_second):
    drag_plot = DragPlot(plot_refresh_rate, visible_duration, data_per_second)
    drag_plot.init_common_params("Drag (lbf)")
    drag_plot.plot_drag_wcomparison()
    drag_plot.term_common_params()

    self.drag_plot_wcomparison_cvs = FigureCanvasTkAgg(drag_plot.fig, master=self.parent)
    return drag_plot
  

  def draw_cartoon_cvs (self, imgpath):
    self._draw_cartoon_lbl()
    
    self.cartoon_cvs = Canvas(self.parent, width=640, height=360) #(0,0) is the top left corner
    self.cartoon_cvs.create_rectangle(3,3,640,360, width=2)

    self._draw_stall_lbl()
    self.cartoon_cvs.create_window (20, 310, window=self.stall_cond_lbl_incartoon, anchor=NW) #x,y
    
    self._draw_UAV_cartoon(imgpath)
    self.cartoon_cvs.create_window (150, 10, window=self.wing_cartoon, anchor=NW)

    self._draw_state_lbl()
    self.cartoon_cvs.create_window (20, 185, window = self.airspeed_icon, anchor=NW)
    self.cartoon_cvs.create_window (20, 210, window = self.airspeed_lbl_incartoon, anchor=NW)
    self.cartoon_cvs.create_window (20, 250, window = self.aoa_icon, anchor=NW)
    self.cartoon_cvs.create_window (20, 280, window = self.aoa_lbl_incartoon, anchor=NW)

  def ingest_cartoon_data (self, plot_refresh_rate, pred_stall_list, pred_airspeed_list, pred_aoa_list, data_per_second):
    self.plot_refresh_rate = plot_refresh_rate
    self.pred_stall_list = pred_stall_list
    self.pred_airspeed_list = pred_airspeed_list
    self.pred_aoa_list = pred_aoa_list
    self.num_samples_per_call = int(data_per_second*self.plot_refresh_rate) # #number of samples coming at each call to plot_live function
  

  def _draw_cartoon_lbl (self):
    self.cartoon_lbl = Label(self.parent, text='Estimated Flight Conditions', font=("Helvetica", 21, 'bold', 'underline'), justify='center')

  def _draw_stall_lbl (self):
    self.stall_lbl = Label(self.parent, text='Stall?', font=("Helvetica", 18), justify='center')
    self.stall_cond_lbl = Label(self.parent, text=self.stall_cond, font=("Arial", 26), justify='center')
    stall_cond = "No" if self.stall_cond == 0 else "Yes"
    self.stall_cond_lbl_incartoon = Label(self.parent, text='Stall: {}'.format(self.stall_cond), fg='green', font=("Arial", 22), justify='center')

  def _draw_UAV_cartoon(self, imgpath):
    self.wing_images_tk = dict()
    # aoas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] #OLD_Sept1
    aoas = np.arange(0, 16.1, 0.1) #NEW_Sept1
    for aoa in aoas:
      # img = Image.open(os.path.join(imgpath,f"{aoa}deg.png")) #OLD_Sept1
      img = Image.open(os.path.join(imgpath,"0deg.png")) #NEW_Sept1
      width, height = img.size
      res_width, res_height = int(width/7.0), int(height/7.0)
      img = img.resize((res_width, res_height)) # (height, width)
      img = img.rotate(-1.0 * aoa) # (height, width)
      imgtk = ImageTk.PhotoImage(img)
      self.wing_images_tk[aoa] = imgtk

    self.wing_cartoon = Label(self.parent, image=self.wing_images_tk[0])
    self.wing_cartoon.image = self.wing_images_tk[0]
    self.wing_cartoon.config(image=self.wing_images_tk[self.init_angle])

    img = Image.open(imgpath+"/airspeed.png")
    width, height = img.size
    res_width, res_height = int(width/24), int(height/24)
    img = img.resize((res_width, res_height)) # (height, width)
    imgtk = ImageTk.PhotoImage(img)
    self.airspeed_icon = Label(self.parent, image=imgtk)
    self.airspeed_icon.image = imgtk
    
    img = Image.open(imgpath+"/aoa.png")
    width, height = img.size
    res_width, res_height = int(width/24), int(height/24)
    img = img.resize((res_width, res_height)) # (height, width)
    imgtk = ImageTk.PhotoImage(img)
    self.aoa_icon = Label(self.parent, image=imgtk)
    self.aoa_icon.image = imgtk

  def _draw_state_lbl (self):
    self.state_lbl = Label(self.parent, text='Flight state', font=("Helvetica", 18), justify='center')
    self.state_est_lbl = Label(self.parent, text="Airspeed = {} m/s \n AoA = {} deg".format(self.est_airspeed, self.est_aoa), font=("Arial", 26), justify='center')
    self.airspeed_lbl_incartoon = Label(self.parent, text='Airspeed = {} m/s'.format(self.est_airspeed), font=("Arial", 18), justify='center')
    self.aoa_lbl_incartoon = Label(self.parent, text='AoA = {} deg'.format(self.est_aoa), font=("Arial", 18), justify='center')


  def update_uav_cartoon(self, old_aoa, start_time):
    new_aoa = 0
    try:
      t0 = time.time()
      # cur_frame = int((t0-start_time)/self.plot_refresh_rate)
      # aoa_est = int(self.pred_aoa_list[cur_frame])
      cur_frame = abs(int((t0-start_time)/self.plot_refresh_rate*self.num_samples_per_call))
      # aoa_est = round((np.mean(self.pred_aoa_list[cur_frame:cur_frame+self.num_samples_per_call]))) #OLD_Sept1
      aoa_est = np.around((np.mean(self.pred_aoa_list[cur_frame:cur_frame+self.num_samples_per_call])), decimals=1) #NEW_Sept1
      old_aoa = old_aoa
      new_aoa = aoa_est
      self.wing_cartoon.config(image=self.wing_images_tk[aoa_est])

    except:
      pass
    self.parent.after(int(self.plot_refresh_rate*1000), self.update_uav_cartoon, new_aoa, start_time)

  def update_stallest_lbls (self, start_time):
    try:
      t0 = time.time()
      # cur_frame = int((t0-start_time)/self.plot_refresh_rate)
      # stall_cond = self.pred_stall_list[cur_frame]
      cur_frame = abs(int((t0-start_time)/self.plot_refresh_rate*self.num_samples_per_call))
      stall_cond = np.mean(self.pred_stall_list[cur_frame:cur_frame+self.num_samples_per_call])
      if stall_cond:
        self.stall_cond_lbl.config(text="Yes", fg='red')
        self.stall_cond_lbl_incartoon.config(text='Stall: Yes', fg='red')
      else:
        self.stall_cond_lbl.config(text="No", fg='green')
        self.stall_cond_lbl_incartoon.config(text='Stall: No', fg='green')
    except:
      pass
    self.parent.after(int(self.plot_refresh_rate*1000), self.update_stallest_lbls, start_time)

  def update_stateest_lbls (self, start_time):
    try:
      t0 = time.time()
      # cur_frame = int((t0-start_time)/self.plot_refresh_rate)
      # airspeed_est = self.pred_airspeed_list[cur_frame]
      # aoa_est = self.pred_aoa_list[cur_frame]
      cur_frame = abs(int((t0-start_time)/self.plot_refresh_rate*self.num_samples_per_call))
      airspeed_est = np.mean(self.pred_airspeed_list[cur_frame:cur_frame+self.num_samples_per_call])
      aoa_est = np.mean(self.pred_aoa_list[cur_frame:cur_frame+self.num_samples_per_call])

      self.state_est_lbl.config(text= f"Airspeed = {float(airspeed_est):.1f} m/s \n AoA = {float(aoa_est):.1f} deg") #Predictions are shape:(pred_count, output_count)
      self.airspeed_lbl_incartoon.config(text = f"Airspeed = {float(airspeed_est):.1f} m/s")
      self.aoa_lbl_incartoon.config(text = f"AoA = {float(aoa_est):.1f} deg  ")
    except:
      pass
    self.parent.after(int(self.plot_refresh_rate*1000), self.update_stateest_lbls, start_time)



  def place_on_grid(self):
    #Top row
    self.video.videolbl.grid(row=1, column=0, rowspan=1, columnspan=4)
    self.video.videocvs.grid(row=2, column=0, rowspan=1, columnspan=4)
    self.cartoon_lbl.grid(row=1, column=4, rowspan=1, columnspan=4)
    self.cartoon_cvs.grid(row=2, column=4, rowspan=1, columnspan=4)

    #Mid. row
    self.midrow_frame_top.grid(row=3, column=0, rowspan=1, columnspan=8, pady=(6,2))
    self.midrow_frame_bottom.grid(row=4, column=0, rowspan=1, columnspan=8, pady=(2,6))

    #Bottom row
    self.airspeed_plot_wcomparison_cvs.get_tk_widget().grid(row=5, column=0, rowspan=1, columnspan=2, padx=(10,5))
    self.aoa_plot_wcomparison_cvs.get_tk_widget().grid(row=5, column=2, rowspan=1, columnspan=2, padx=(5,5))
    self.lift_plot_wcomparison_cvs.get_tk_widget().grid(row=5, column=4, rowspan=1, columnspan=2, padx=(5,5))
    self.drag_plot_wcomparison_cvs.get_tk_widget().grid(row=5, column=6, rowspan=1, columnspan=2, padx=(5,10))
  
