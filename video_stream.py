# video_stream.py
import cv2
import threading
import time
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

class LiveStream:
    def __init__(self, sursa):
        self.sursa = sursa
        self.cap = cv2.VideoCapture(self.sursa)
        self.is_live = isinstance(self.sursa, str) and (self.sursa.startswith('rtsp') or self.sursa.startswith('http'))
        
        if self.is_live:
           
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) 
            
        self.ret, self.frame = self.cap.read()
        self.frame_id = 0 
        self.running = True
        self.lock = threading.Lock()
        
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
        print("[Video Stream] Conectat. Mod: Fluență Naturală (Echilibrat).")

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                
                with self.lock:
                    if ret:
                        self.ret = ret
                        self.frame = frame
                        self.frame_id += 1
                    else:
                        if not self.is_live:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        else:
                            time.sleep(0.5)
                
                
                if not self.is_live:
                    time.sleep(0.033) 
                else:
                    time.sleep(0.01)  

    def citeste_cadru(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy(), self.frame_id
            return self.ret, None, self.frame_id

    def opreste(self):
        self.running = False
        self.thread.join()
        self.cap.release()