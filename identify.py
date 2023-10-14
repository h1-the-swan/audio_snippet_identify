import librosa
import numpy as np
from scipy import signal

class Identify:
    # https://github.com/hiisi13/audio-offset-finder/blob/30fda3d9f9b02c753c659a6468636d52a01f5466/audio_offset_finder.py#L9-L21
    def __init__(self, find_file, window=10) -> None:
        self.find_file = find_file
        self.y_find, self.sr_find = librosa.load(self.find_file, sr=None)
        self.window = window

    def get_offset(self, within_file):
        # gets the offset in seconds of the snippet within within_file
        y_within, sr_within = librosa.load(within_file, sr=self.sr_find)
        c = signal.correlate(y_within, self.y_find[:sr_within*self.window], mode='valid', method='fft')
        peak = np.argmax(c)
        offset = round(peak / sr_within, 2)
        return offset

