import os
import sys
import pathlib
from tkinter import Frame, Label, Canvas
from tkinter import LEFT, RIGHT, BOTTOM
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

FILE_PATH = pathlib.Path(__file__).absolute()
GUI_LOC = FILE_PATH.parent.parent
sys.path.insert(0, str(GUI_LOC))

from gui_helpers.plot_helper import AirspeedPlot
from gui_helpers.plot_helper import AoaPlot
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
    res_width, res_height = int(width/1.6), int(height/1.6)
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

  def draw_cartoon_cvs(self):
    pass

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
  
  def place_on_grid(self):
    #Top row
    self.video.videolbl.grid(row=1, column=0, rowspan=1, columnspan=4)
    self.video.videocvs.grid(row=2, column=0, rowspan=1, columnspan=4)
    # self.cartoon_lbl.grid(row=1, column=4, rowspan=1, columnspan=4)
    # self.cartoon_cvs.grid(row=2, column=4, rowspan=1, columnspan=4)

    #Mid. row
    self.midrow_frame_top.grid(row=3, column=0, rowspan=1, columnspan=8, pady=(6,2))
    self.midrow_frame_bottom.grid(row=4, column=0, rowspan=1, columnspan=8, pady=(2,6))

    #Bottom row
    self.airspeed_plot_wcomparison_cvs.get_tk_widget().grid(row=5, column=0, rowspan=1, columnspan=2, padx=(10,5))
    self.aoa_plot_wcomparison_cvs.get_tk_widget().grid(row=5, column=2, rowspan=1, columnspan=2, padx=(5,5))
    # self.lift_plot_wcomparison_cvs.get_tk_widget().grid(row=5, column=4, rowspan=1, columnspan=2, padx=(5,5))
    # self.drag_plot_wcomparison_cvs.get_tk_widget().grid(row=5, column=6, rowspan=1, columnspan=2, padx=(5,10))