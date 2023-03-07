from threading import Thread
import time
from matplotlib.animation import FuncAnimation

class StreamOffline():
  def __init__(self, streamhold_queue):
    self.streamhold_queue = streamhold_queue
    self.airspeed_plot = None
    self.aoa_plot = None
    self.lift_plot = None
    self.drag_plot = None

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

  def initialize_video(self, GUIapp, video_label, video_path, experiment_name):
    self.video = GUIapp.draw_video(video_label, video_path, experiment_name)
    videostream_thr = Thread(target=self.stream_video, args=(0,))
    videostream_thr.start()
    print ("Initialized videos")

  def stream_video(self, video_id):
    while True: #Wait
      if not self.streamhold_queue.empty():
        print ("Started streaming video {}".format(video_id))
        time_delay = 0.6 #This time delay is here because it takes a bit that video actually starts after stream_images command. Increasing this makes video go earlier than plots.
        self.start_time = time.time() + time_delay
        self.video.stream_images(self.start_time - time_delay, 1/30)
        break
      else:
        pass

  def initialize_plots_wcomparison(self, GUIapp, include_state_pred, include_liftdrag_pred, plot_refresh_rate, visible_duration, data_per_second):
    self.GUIapp = GUIapp
    self.plot_refresh_rate = plot_refresh_rate

    if include_state_pred:
      self.airspeed_plot = GUIapp.draw_airspeed_plot_wcomparison(plot_refresh_rate, visible_duration, data_per_second)
      self.aoa_plot = GUIapp.draw_aoa_plot_wcomparison(plot_refresh_rate, visible_duration, data_per_second)

    plots_wcomparison_thr = Thread(target=self.stream_plots_wcomparison)
    plots_wcomparison_thr.start()
    print ("Initialized plotting w_comparisons")

  def stream_plots_wcomparison(self):
    while True: #Wait
      if not self.streamhold_queue.empty():
        print ("Started streaming plotting w_comparisons")
        if self.airspeed_plot is not None:
          time.sleep(0.05)
          _ = FuncAnimation(self.airspeed_plot.fig, self.airspeed_plot.plot_airspeed_live, fargs=(self.meas_airspeed_list, self.pred_airspeed_list, self.start_time), interval=self.plot_refresh_rate*1000, blit=True)
        if self.aoa_plot is not None:
          time.sleep(0.05)
          _ = FuncAnimation(self.aoa_plot.fig, self.aoa_plot.plot_aoa_live, fargs=(self.meas_aoa_list, self.pred_aoa_list, self.start_time), interval=self.plot_refresh_rate*1000, blit=True)
        self.GUIapp.update()
        break
      else:
        pass