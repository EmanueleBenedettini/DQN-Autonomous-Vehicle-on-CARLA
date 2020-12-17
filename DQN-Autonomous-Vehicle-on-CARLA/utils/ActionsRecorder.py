import time
import datetime


class VideoRecording:
    def __init__(self, cap, car, out):
        super(VideoRecording, self).__init__()
        self.cap = cap
        self.car = car
        self.out = out

        self.video_recording = True

    def run(self):
        global current_frame
        while(video_recording):
            current_frame = self.cap.read()
            self.out.append( (current_frame, self.car.get_current_control(), self.car.has_crashed()) )
            time.sleep(0.015)
            

class ActionRecorder:

    def __init__(self, camera, car):
        self.cap = camera
        self.car = car
        self.list = []


    def start_recording(self):
        self.thread = VideoRecording(self.cap, self.car, self.list)
        self.thread.start()
        self.start_time = datetime.datetime.now()
    
    def stop_recording(self):
        global video_recording
        video_recording = False

    def get_recorded_data(self):
        global video_recording
        if video_recording:
            raise RuntimeError('Cannot read directly while recorder is running')
        return self.list