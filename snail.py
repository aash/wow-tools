import ahk as autohotkey
from win_tools import get_window_rect
import logging
from d3dshot_stub import D3DShot, CaptureOutputs
import time
from common import Rect
import numpy as np

class Snail:

    WOW_WINDOW_NAME = 'World of Warcraft'
    COLOR_BLACK = (0, 0, 0)
    COLOR_GREEN = (0, 255, 0)

    def __init__(self):
        window_name = self.WOW_WINDOW_NAME
        self.ahk = autohotkey.AHK()
        self.ahk.set_coord_mode('Mouse', 'Client')
        self.window = self.ahk.find_window(title=window_name)
        self.window_id = int(self.window.id, 16)
        self.window.activate()
        self.window_rect = get_window_rect(self.ahk, window_name)
        self.d3d_fps = 30
        pass

    def __enter__(self):
        logging.info('Starting snail')
        self.d3d = D3DShot(capture_output=CaptureOutputs.NUMPY, fps = self.d3d_fps, roi = self.window_rect)
        self.d3d.capture(target_fps=self.d3d_fps, region=self.window_rect.xyxy())
        logging.info(f'snail started {self.window_rect}')
        self.ensure_next_frame()
        time.sleep(0.2)
        return self

    def __exit__(self, *exc_details):
        time.sleep(0.1)
        logging.info('Stopping snail')
        self.d3d.stop()
        del self.ahk

    def get_diff_image(self, action, initialize = None, finalize = None, roi = None):

        '''
        Get two consequtive images of a window, taken before action and after,
        with the option of initialization and finalization.
        '''
        if initialize:
            initialize()
        im1, _ = self.d3d.wait_next_frame()
        action()
        im2, _ = self.d3d.wait_next_frame()
        if finalize:
            finalize()
        return im1, im2

    def wait_next_frame(self, roi: Rect = None) -> np.ndarray:
        f, *_ = self.d3d.wait_next_frame(roi=roi)
        return f

    def wait_next_frame_with_time(self, roi: Rect = None):
        f, t = self.d3d.wait_next_frame(roi=roi)
        return f, t
    
    def ensure_next_frame(self):
        im = None
        while im is None:
            im = self.wait_next_frame()
        return
