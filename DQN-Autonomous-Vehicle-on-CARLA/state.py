import numpy as np
import blosc
import cv2 as cv


def canny_filter(image, sizex, sizey, soglia1, soglia2):    # Edge enhancer
    blurred = cv.GaussianBlur(image, (sizex, sizey), 0)
    edges = cv.Canny(blurred, soglia1, soglia2)
    res = image.copy()
    res[edges!=0] = 255
    return res

def transform_to_tf_input(img, width=84, height=84):    # Transforms image to tf input
    dim = (width, height)
    resized = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    resized = cv.resize(resized, dim, interpolation = cv.INTER_AREA)
    resized = canny_filter(resized, 5, 5, 68, 88) # highlight borders
    cv.equalizeHist(resized)    # equalize histogram
    return resized


class State:
    IMAGE_HEIGHT = 84
    IMAGE_WIDHT = 84
    useCompression = False
    step_frames = 4

    @staticmethod
    def setup(args):
        State.useCompression = args.compress_replay
        State.step_frames = args.history_length
        IMAGE_HEIGHT = args.image_height
        IMAGE_WIDHT = args.image_width

    def state_by_adding_screen(self, screen, frameNumber):

        screen = transform_to_tf_input(screen, State.IMAGE_HEIGHT, State.IMAGE_WIDHT)  
        screen.resize((State.IMAGE_HEIGHT, State.IMAGE_WIDHT, 1)) #adapt image to 3 dimensions

        if State.useCompression:
            screen = blosc.compress(
                np.reshape(screen, State.IMAGE_HEIGHT * State.IMAGE_WIDHT).tobytes(), typesize=1)

        newState = State()
        if hasattr(self, 'screens'):
            newState.screens = self.screens[:State.step_frames - 1]
            newState.screens.insert(0, screen)
        else:
            newState.screens = []
            for i in range(State.step_frames):
                newState.screens.append(screen)
        return newState

    def get_screens(self):
        if State.useCompression:
            s = []
            for i in range(State.step_frames):
                s.append(np.reshape(np.fromstring(
                    blosc.decompress(
                        self.screens[i]), dtype=np.uint8), (State.IMAGE_HEIGHT, State.IMAGE_WIDHT, 1)))
        else:
            s = self.screens
        if State.step_frames == 1:
            return s[0]
        return np.concatenate(s, axis=2)
