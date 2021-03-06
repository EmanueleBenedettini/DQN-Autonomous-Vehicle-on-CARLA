import threading

from .camera import Camera


class CarlaCamera(Camera):
    def __init__(self, car):
        super(CarlaCamera, self).__init__()
        self.car = car
        self.width, self.height = self.car.get_im_sizes()
        self.thread = threading.Thread(target=self._capture_frames)

    def _read(self):
        image = self.car.get_image()
        #image[self.car.get_road_highlight() < 128] = 0 #This trims out everything except road
        self.value = image
        return image
