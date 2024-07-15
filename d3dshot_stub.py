import time
import numpy as np
from common import Rect
import logging
import cv2
import cv2 as cv
import os, sys
import dxgi_screen_capture
from contextlib import contextmanager

__version__ = '1.0.0'

class CaptureOutputs:
    NUMPY = 1

class D3DShot:
    FULL_SCREEN = (0, 0, 1980*2, 1080*2)

    # capture_output is here for compatibility with original D3DShot implementation
    def __init__(self, capture_output = CaptureOutputs.NUMPY, fps = 30, roi: Rect = Rect(*FULL_SCREEN)):
        self.fps = fps
        self.roi = roi
        self.d3d = dxgi_screen_capture.dxgisc()
        self.d3d.Init()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self

    def capture(self, *argv, **argvd):
        pass

    def stop(self):
        del self.d3d
    
    def wait_next_frame(self, t = 0, roi: Rect = None) -> np.ndarray:
        sc = self.d3d
        fr1 = fr0 = sc.GetFrameNo()
        t0 = time.time()
        timeout = 0.3
        while fr1 == fr0:
            z = sc.Capture(33)
            fr1 = sc.GetFrameNo()
            if fr0 < fr1:
                r = self.roi if not roi else self.roi.sub_rect(roi) 
                c = sc.Dump(*r.xywh())
                im = c.astype(dtype=np.uint8).reshape((r.height(), r.width(), 4))
                im = cv.cvtColor(im, cv.COLOR_BGRA2BGR)
            if time.time() - t0 > timeout:
                raise TimeoutError('frame capture timeout')
        return im, fr1
