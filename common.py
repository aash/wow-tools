import numpy as np
import cv2 as cv
from enum import Enum
from dataclasses import dataclass
from typing import List
from copy import deepcopy
import time
import contextlib
import itertools
import queue
import ahk as autohotkey
import logging

class DataObject:
    def __init__(self, data_dict):
        self.__dict__ = data_dict

def millis_now():
    return int(time.time() * 1000)

'''
h: 0-179
s: 0-255
v: 0-255
'''
def hsv2rgb(hsv):
    img = np.zeros((1, 1, 3), dtype=np.uint8)
    img[0][0] = (np.array(hsv) * np.array([179, 255, 255])).astype(np.uint8)
    rgb = cv.cvtColor(img, cv.COLOR_HSV2BGR)
    return tuple(map(int, (rgb[0][0]*255).astype(np.uint8)))

def get_palette(size: int):
    assert 0 < size <= 42
    f = 1/size
    for c in range(size):
        print(c*f)
    return [hsv2rgb((c*f, 0.99, 0.99)) for c in range(size)]

def bits(n: int):
    l = [n >> i & 1 for i in range(n.bit_length())]
    ll = list(enumerate(l))
    lll = set(map(lambda x: x[1] << x[0], ll))
    return lll

class UiLocation(Enum):
    LEFT = 1
    RIGHT = 2
    TOP = 3
    BOTTOM = 4
    HCENTER = 5
    VCENTER = 6

    def __str__(self):
        return self.name.split('.')[-1].lower()


@dataclass
class Rect:
    x0: int
    y0: int
    w: int
    h: int


    def top(self) -> int:
        return self.y0
    
    def bottom(self) -> int:
        return self.y0 + self.h
    
    def left(self) -> int:
        return self.x0
    
    def right(self) -> int:
        return self.x0 + self.w
    
    def top_left(self):
        return (self.x0, self.y0)

    def top_right(self):
        return (self.x0 + self.w, self.y0)
    
    def bottom_left(self):
        return (self.x0, self.y0 + self.h)

    def bottom_right(self):
        return (self.x0 + self.w, self.y0 + self.h)
    
    def left_segment(self):
        return Segment(self.y0, self.y0 + self.h)

    def top_segment(self):
        return Segment(self.x0, self.x0 + self.w)
    
    def xywh(self):
        return (self.x0, self.y0, self.w, self.h)
    
    def xyxy(self):
        return (self.x0, self.y0, self.x0 + self.w, self.y0 + self.h)
    
    def width(self):
        return self.w
    
    def height(self):
        return self.h
    
    def wh(self):
        return (self.w, self.h)
    
    def xy(self):
        return (self.x0, self.y0)
    
    def sub_rect(self, sub: 'Rect'):
        return Rect(self.x0 + sub.x0, self.y0 + sub.y0, *sub.wh())
    
    def moved(self, dx: int, dy: int):
        return Rect(self.x0 + dx, self.y0 + dy, *self.wh())
    
    def __add__(self, other: np.array):
        return Rect(self.x0 + other[0], self.y0 + other[1], self.w, self.h)
    
    @classmethod
    def from_xyxy(cls, x0: int, y0: int, x1: int, y1: int) -> 'Rect':
        p0, p1 = sorted((x0, x1))
        q0, q1 = sorted((y0, y1))
        return Rect(p0, q0, p1 - p0, q1 - q0)
    
    @classmethod
    def from_top_left(cls, x: int, y: int, w: int, h: int) -> 'Rect':
        return Rect(x, y, w, h)

    @classmethod
    def from_bottom_left(cls, x: int, y: int, w: int, h: int) -> 'Rect':
        return Rect(x, y - h, w, h)

    @classmethod
    def from_bottom_right(cls, x: int, y: int, w: int, h: int) -> 'Rect':
        return Rect(x - w, y - h, w, h)

    @classmethod
    def from_top_right(cls, x: int, y: int, w: int, h: int) -> 'Rect':
        return Rect(x - w, y, w, h)
    
@dataclass
class Segment:
    left: int
    right: int

class BoundingRect:

    def __init__():
        pass

def is_inside(p: Segment, q: Segment, threshold: int = 1):
    '''
    check if `q` segment is inside another segment `p` with threshold `threshold`
    '''
    return q.left - p.left > threshold and p.right - q.right > threshold
    
def label_brect(rect: Rect, window: Rect, threshold: int = 1):
    lbls = set()
    if rect.left() - window.left() < threshold:
        lbls.add(UiLocation.LEFT)
    if window.right() - rect.right() < threshold:
        lbls.add(UiLocation.RIGHT)
    if rect.top() - window.top() < threshold:
        lbls.add(UiLocation.TOP)
    if window.bottom() - rect.bottom() < threshold:
        lbls.add(UiLocation.BOTTOM)
    if is_inside(window.left_segment(), rect.left_segment(), threshold):
        lbls.add(UiLocation.VCENTER)
    if is_inside(window.top_segment(), rect.top_segment(), threshold):
        lbls.add(UiLocation.HCENTER)
    return lbls

# def crop_image(img: np.ndarray, r: Rect) -> np.ndarray:
#     b = r.xyxy()
#     return img[b[1]:b[3], b[0]:b[2]].copy()

def crop_image(img: np.ndarray, r: Rect, debug = False) -> np.ndarray:
    """
    Crops a part of an image using a rectangle defined by the top-left corner, width, and height.
    If the rectangle goes beyond the image boundaries, it will be truncated.
    """

    r = deepcopy(r)

    x0 = max(0, r.x0)
    y0 = max(0, r.y0)
    x1 = min(img.shape[1], r.x0 + r.w)
    y1 = min(img.shape[0], r.y0 + r.h)
    if debug:
        return img[y0:y1, x0:x1].copy(), (x0, y0), (x1, y1)
    else:
        return img[y0:y1, x0:x1].copy()

def erode(img: np.ndarray, sz: int, shape):
    el = cv.getStructuringElement(shape, (2 * sz + 1, 2 * sz + 1), (sz, sz))
    return cv.erode(img, el)

def dilate(img: np.ndarray, sz: int, shape):
    el = cv.getStructuringElement(shape, (2 * sz + 1, 2 * sz + 1), (sz, sz))
    return cv.dilate(img, el)

class MoveDirectionSimple(Enum):
    UP     = 0b0001
    DOWN   = 0b0010
    LEFT   = 0b0100
    RIGHT  = 0b1000

    @classmethod
    def values(cls):
        return set([e.value for e in cls])

class MoveDirectionComposite(Enum):
    UP_LEFT = MoveDirectionSimple.UP.value | MoveDirectionSimple.LEFT.value
    UP_RIGTH = MoveDirectionSimple.UP.value | MoveDirectionSimple.RIGHT.value
    DOWN_LEFT = MoveDirectionSimple.DOWN.value | MoveDirectionSimple.LEFT.value
    DOWN_RIGHT = MoveDirectionSimple.DOWN.value | MoveDirectionSimple.RIGHT.value

    @classmethod
    def values(cls):
        return set([e.value for e in cls])

class MoveDirection(Enum):
    UP     = MoveDirectionSimple.UP.value
    DOWN   = MoveDirectionSimple.DOWN.value
    LEFT   = MoveDirectionSimple.LEFT.value
    RIGHT  = MoveDirectionSimple.RIGHT.value
    UP_LEFT = MoveDirectionComposite.UP_LEFT.value
    UP_RIGTH = MoveDirectionComposite.UP_RIGTH.value
    DOWN_LEFT = MoveDirectionComposite.DOWN_LEFT.value
    DOWN_RIGHT = MoveDirectionComposite.DOWN_RIGHT.value

    def simplify(self) -> List[MoveDirectionSimple]:
        bb = bits(self.value).difference({0})
        return [MoveDirectionSimple(m) for m in bb]

    @classmethod
    def values(cls):
        return set([e.value for e in cls])

class KeyState(Enum):
    PRESS = 0
    RELEASE = 1

def wrap(s: str, c: str) -> str:
    d = {
        '{': ('{', '}'),
        '}': ('{', '}'),
        '[': ('[', ']'),
        ']': ('[', ']'),
        '(': ('(', ')'),
        ')': ('(', ')'),
    }
    if c in d:
        return d[c][0] + s + d[c][1]
    else:
        raise RuntimeError('unreachable')

def get_ahk_sequence(dir: MoveDirection, key_state: KeyState) -> str:
    d2k = {
        MoveDirectionSimple.UP: 'w',
        MoveDirectionSimple.DOWN: 's',
        MoveDirectionSimple.LEFT: 'a',
        MoveDirectionSimple.RIGHT: 'd',
    }
    ks2s = {
        KeyState.RELEASE: 'up',
        KeyState.PRESS: 'down',
    }
    s = ks2s[key_state]
    k = [d2k[_k] for _k in dir.simplify()]
    ss = [wrap(f'{_k} {s}', '{') for _k in k]
    return ''.join(ss)


def time_range(dur: float):
    t0 = time.time()
    i = 0
    grid = [t0]
    while True:
        if len(grid) > 20:
            grid.pop(0)
        t = time.time()
        grid.append(t)
        if len(grid) > 1:
            diffs = [b-a for b, a in list(zip(grid[1:], grid[:-1]))]
            avg_time = sum(diffs) / len(diffs)
        i += 1
        if time.time() - t0 > dur:
            break
        fps = 0.0 if len(grid) < 2 or avg_time == 0 else 1 / avg_time
        yield t, fps, i

class timer_unit(Enum):
    SECOND = 1
    MILLISECOND = 2

@contextlib.contextmanager
def timer(unit: timer_unit = timer_unit.SECOND):
    if unit is timer_unit.SECOND:
        t0 = time.time()
        yield lambda : time.time() - t0
    elif unit is timer_unit.MILLISECOND:
        t0 = int(1000*time.time())
        yield lambda : int(1000*time.time()) - t0

@contextlib.contextmanager
def timer_sec():
    t0 = time.time()
    yield lambda : time.time() - t0

@contextlib.contextmanager
def timer_ms():
    t0 = int(1000*time.time())
    yield lambda : int(1000*time.time()) - t0

def cart_prod(x, y):
    return list(itertools.product(x, y))

def grid(vl: np.ndarray, hl: np.ndarray) -> np.ndarray:
    return np.array([[(v, h) for h in hl] for v in vl])

def hstack(imgs):
    maxh = max([i.shape[0] for i in imgs])
    out_imgs = []
    for img in imgs:
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        img = np.vstack((img, np.zeros((maxh - img.shape[0], img.shape[1], img.shape[2]), dtype=img.dtype)))
        out_imgs.append(img)
    return np.hstack(out_imgs)


@contextlib.contextmanager
def exit_hotkey(key = '^q', ahk = None):
    q = queue.Queue()
    if ahk is None:
        ahk = autohotkey.AHK()
    ahk.add_hotkey(key, lambda: q.put('exit'), logging.info('exit hotkey handler'))
    ahk.start_hotkeys()
    def get_command():
        if not q.empty():
            return q.get()
        return None
    yield get_command
    ahk.stop_hotkeys() 

@contextlib.contextmanager
def hotkey_handler(key, cmd):
    q = queue.Queue()
    ahk = autohotkey.AHK()
    logging.info(f'adding new hotkey {key} {cmd}')
    ahk.add_hotkey(key, lambda: q.put(cmd), logging.info(f"{cmd} command triggered"))
    ahk.start_hotkeys()
    def get_command():
        if not q.empty():
            cmd = q.get()
            logging.info(f'hotkey triggered {cmd}')
            return cmd
        return None
    yield get_command
    ahk.stop_hotkeys()

@dataclass
class point2d:
    xy: np.ndarray
    def __call__(self, inv: bool = False):
        if inv:
            return np.array((self.xy[1], self.xy[0]))
        return self.xy
    @classmethod
    def fromndarray(cls, arr: np.ndarray):
        assert arr.shape == (2, )
        return cls(arr) 
    @classmethod
    def fromxy(cls, x: int, y: int):
        return cls(np.array((x, y)))

@dataclass
class cell_loc:
    xy: np.ndarray
    def __call__(self, inv: bool = False):
        if inv:
            return np.array((self.xy[1], self.xy[0]))
        return self.xy
    def from_char_loc(loc: point2d, grid_width: int):
        return cell_loc(loc() // grid_width)

@contextlib.contextmanager
def timeout(tsec: float):
    t0 = time.time()
    def is_not_timeout():
        return time.time() - t0 < tsec
    yield is_not_timeout
def mixin(dst: np.ndarray, src: np.ndarray, alpha: float) -> np.ndarray:
    assert 0 < alpha <= 1.0
    dst = cv.addWeighted(dst, alpha, src, 1.0 - alpha, 0, dst)
    return dst

def get_midpoint(im: np.ndarray) -> point2d:
    return point2d.fromxy(im.shape[1] // 2, im.shape[0] // 2)

def strip_zeros_2d(image):
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy ndarray")
    
    if image.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")
    
    # Find rows and columns that are completely zero
    non_zero_rows = np.where(image.sum(axis=1) != 0)[0]
    non_zero_cols = np.where(image.sum(axis=0) != 0)[0]
    
    # If all rows or all columns are zero
    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
        return np.array([[]])  # Return an empty 2D array
    
    # Determine the first and last non-zero row and column
    first_non_zero_row = non_zero_rows[0]
    last_non_zero_row = non_zero_rows[-1]
    first_non_zero_col = non_zero_cols[0]
    last_non_zero_col = non_zero_cols[-1]
    
    # Slice the array to remove zero rows and columns
    stripped_image = image[first_non_zero_row:last_non_zero_row + 1, first_non_zero_col:last_non_zero_col + 1]
    
    return stripped_image


def is_entity_tile(tile):
    entity_color = (0, 255, 0)
    out = cv.inRange(tile, entity_color, entity_color)
    ent_color_num = cv.countNonZero(out)
    return ent_color_num > 5

def get_closest(mvl, v):
    if len(mvl) < 1:
        return None
    dist = map(lambda x: tuple([x[0], abs(x[1] - v)]), enumerate(mvl))
    closest = min(dist, key=lambda x: x[1])
    return closest[0]