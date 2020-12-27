import numpy as np
import blosc
import cv2 as cv


def transform_to_tf_input(img, width=84, height=84):
    dim = (width, height)
    resized = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    resized = cv.resize(resized, dim, interpolation = cv.INTER_AREA) 
    return resized


class State:
    IMAGE_SIZE = 84
    useCompression = False
    step_frames = 4

    @staticmethod
    def setup(args):
        State.useCompression = args.compress_replay
        State.step_frames = args.frame

    def state_by_adding_screen(self, screen, frameNumber):
        #screen = np.dot(screen, np.array([.299, .587, .114])).astype(np.uint8)  #trasforma in grigi

        #ridimensiona usando numpy
        #y_resize = State.IMAGE_SIZE / screen.shape[0]
        #x_resize = State.IMAGE_SIZE / screen.shape[1]
        #screen = cv.resize(screen, (0, 0), fx=x_resize, fy=y_resize)

        screen = transform_to_tf_input(screen, State.IMAGE_SIZE, State.IMAGE_SIZE)

        screen.resize((State.IMAGE_SIZE, State.IMAGE_SIZE, 1))

        if State.useCompression:
            screen = blosc.compress(
                np.reshape(screen, State.IMAGE_SIZE * State.IMAGE_SIZE).tobytes(), typesize=1)

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
                        self.screens[i]), dtype=np.uint8), (State.IMAGE_SIZE, State.IMAGE_SIZE, 1)))
        else:
            s = self.screens
        if State.step_frames == 1:
            return s[0]
        return np.concatenate(s, axis=2)
