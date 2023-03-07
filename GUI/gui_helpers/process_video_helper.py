import os
import time
import subprocess
import tkinter as tk
from tkinter import Frame, Canvas, Label
import PIL.Image, PIL.ImageTk
import numpy as np
import cv2

class DrawTKOfflineVideo(Frame):
  def __init__(self, parent, video_label, videopath, experiment_name):
    Frame.__init__(self, parent)
    self.parent = parent
    self.video_label = video_label
    self.experiment_name = experiment_name
    self.videopath = videopath
    self.videocvs = Canvas(parent, width=640, height=360)
    self.videolbl = Label(parent, text=video_label, font=("Helvetica", 21, 'bold', 'underline'))
    self.draw_and_process_image()

  def draw_and_process_image(self):
    step = 0
    self.frame_photos = list()
    path = os.path.join(self.videopath, self.experiment_name+"_auxvideo_480p.mov")
    print (path)
    
    #Do with openCV
    # self.cap = cv2.VideoCapture(path)
    # if (self.cap.isOpened() == False): 
    #   print("Error opening the video file") 
    
    #Do with ffmpeg
    args = ["ffmpeg", "-i", path, "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"]
    pipe = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=640 * 360 * 3)

    while True:
      try:
        #Do with openCV
        # _, frame = self.cap.read()
        # recolored_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #Do with ffmpeg
        frame = pipe.stdout.read(640 * 360 * 3)
        
        array = np.frombuffer(frame, dtype="uint8").reshape((360, 640, 3))

        frame_photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(array))
        if step == 0:
          self.vidframe = self.videocvs.create_image(0,0,image = frame_photo, anchor = tk.NW)
        self.frame_photos.append (frame_photo)
        step += 1
      except:
        break

  def stream_images(self, start_time, delay=1/30):
    call_delay = 30
    try:
      t0 = time.time()
      cur_frame = int((t0-start_time)/delay)
      frame_photo = self.frame_photos[cur_frame]
      self.videocvs.itemconfig (self.vidframe, image=frame_photo)
      self.videocvs.image = frame_photo
      if time.time() - start_time < 1:
        call_delay = 100
      
    except Exception as e:
      print (e)
    
    self.parent.after (call_delay, self.stream_images, start_time, delay)