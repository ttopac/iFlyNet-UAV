from threading import Thread
import time
from matplotlib.animation import FuncAnimation

class StreamOffline():
  def __init__(self, streamhold_queue, GUIapp):
    self.streamhold_queue = streamhold_queue
    self.airspeed_plot = None
    self.aoa_plot = None
    self.lift_plot = None
    self.drag_plot = None
    self.GUIapp = GUIapp

    self.meas_airspeed_list = list()
    self.meas_aoa_list = list()
    self.pred_airspeed_list = list()
    self.pred_aoa_list = list()
    self.uav_airspeed_list = list()
    self.uav_aoa_list = list()
    self.meas_lift_list = list()
    self.meas_drag_list = list()
    self.pred_lift_list = list()
    self.pred_drag_list = list()
    self.pred_stall_list = list()

  def prep_data_state(self, airspeed_truth, aoa_truth, airspeed_preds, aoa_preds, airspeed_uav, aoa_uav):
    # All of these are of shape (16114,) for dynamic15 (30 datapoints per second for ~8 minutes 57 seconds)
    self.meas_airspeed_list = airspeed_truth
    self.meas_aoa_list = aoa_truth
    self.pred_airspeed_list = airspeed_preds
    self.pred_aoa_list = aoa_preds
    self.uav_airspeed_list = airspeed_uav
    self.uav_aoa_list = aoa_uav

  def prep_data_stall(self, stall_preds):
    # Of shape (16114,) for dynamic15 (30 datapoints per second for ~8 minutes 57 seconds)
    self.pred_stall_list = stall_preds

  def prep_data_liftdrag(self, lift_truth, drag_truth, lift_preds, drag_preds):
    # All of these are of shape (16114,) for dynamic15 (30 datapoints per second for ~8 minutes 57 seconds)
    self.meas_lift_list = lift_truth
    self.meas_drag_list = drag_truth
    self.pred_lift_list = lift_preds
    self.pred_drag_list = drag_preds


  def initialize_video(self, video_label, video_path, experiment_name):
    self.video = self.GUIapp.draw_video(video_label, video_path, experiment_name)
    videostream_thr = Thread(target=self.stream_video, args=(0,))
    videostream_thr.start()
    print ("Initialized videos")

  def stream_video(self, video_id):
    while True: #Wait
      if not self.streamhold_queue.empty():
        print ("Started streaming video {}".format(video_id))
        time_delay = -0.6 #This time delay is here to fix the delay between video and data timing. There's usually some time offset between video and data.
        self.start_time = time.time() + time_delay
        self.video.stream_images(self.start_time - time_delay, 1/30)
        break
      else:
        pass

  def initialize_plots_wcomparison(self, include_state_pred, include_liftdrag_pred, plot_refresh_rate, visible_duration, data_per_second):
    self.plot_refresh_rate = plot_refresh_rate
    self.data_per_second = data_per_second

    if include_state_pred:
      self.airspeed_plot = self.GUIapp.draw_airspeed_plot_wcomparison(plot_refresh_rate, visible_duration, data_per_second)
      self.aoa_plot = self.GUIapp.draw_aoa_plot_wcomparison(plot_refresh_rate, visible_duration, data_per_second)
    if include_liftdrag_pred:
      self.lift_plot = self.GUIapp.draw_lift_plot_wcomparison(plot_refresh_rate, visible_duration, data_per_second)
      self.drag_plot = self.GUIapp.draw_drag_plot_wcomparison(plot_refresh_rate, visible_duration, data_per_second)

    plots_wcomparison_thr = Thread(target=self.stream_plots_wcomparison)
    plots_wcomparison_thr.start()
    print ("Initialized plotting w_comparisons")

  def stream_plots_wcomparison(self):
    while True: #Wait
      if not self.streamhold_queue.empty():
        print ("Started streaming plotting w_comparisons")
        if self.airspeed_plot is not None:
          time.sleep(0.01)
          _ = FuncAnimation(self.airspeed_plot.fig, self.airspeed_plot.plot_airspeed_live, fargs=(self.meas_airspeed_list, self.pred_airspeed_list, self.start_time), interval=self.plot_refresh_rate*1000, blit=True)
        if self.aoa_plot is not None:
          time.sleep(0.01)
          _ = FuncAnimation(self.aoa_plot.fig, self.aoa_plot.plot_aoa_live, fargs=(self.meas_aoa_list, self.pred_aoa_list, self.start_time), interval=self.plot_refresh_rate*1000, blit=True)
        if self.lift_plot is not None:
          time.sleep(0.01)
          _ = FuncAnimation(self.lift_plot.fig, self.lift_plot.plot_lift_live, fargs=(self.meas_lift_list, self.pred_lift_list, self.start_time), interval=self.plot_refresh_rate*1000, blit=True)
        if self.drag_plot is not None:
          time.sleep(0.01)
          _ = FuncAnimation(self.drag_plot.fig, self.drag_plot.plot_drag_live, fargs=(self.meas_drag_list, self.pred_drag_list, self.start_time), interval=self.plot_refresh_rate*1000, blit=True)
        self.GUIapp.update()
        break
      else:
        pass

  def initialize_cartoon(self, imgpath):
    self.GUIapp.draw_cartoon_cvs(imgpath)
    self.GUIapp.ingest_cartoon_data(self.plot_refresh_rate, self.pred_stall_list, self.pred_airspeed_list, self.pred_aoa_list, self.data_per_second)

    update_uavcartoon_thr = Thread(target=self.update_cartoon_elements)
    update_uavcartoon_thr.start()
    print ("Initialized cartoon graphic")

    
  def update_cartoon_elements (self):
    while True: #Wait
      if not self.streamhold_queue.empty():
        print ("Started updating the cartoon elements")
        time.sleep(0.01)
        self.GUIapp.update_stallest_lbls(self.start_time)

        time.sleep(0.01)
        self.GUIapp.update_stateest_lbls(self.start_time)

        self.GUIapp.update_uav_cartoon(0, self.start_time)
        time.sleep(0.01)
        self.GUIapp.update()
        break
      else:
        pass
    
