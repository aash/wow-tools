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
import asyncio
# import logging
from typing import Callable, Any, Union
import sys
from PIL import Image as PILImage
from IPython.display import display

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

''' Creates a set of non-zero 2-ary polynome components for a number n
for example 13 = 0b1101 it returns {1, 4, 8} because 3 bits are non-zero
first, third and fourth
'''
def bits(n: int):
    non_zero_bits = [n >> i & 1 for i in range(0, n.bit_length())]
    non_zero_bits_enumerated = list(enumerate(non_zero_bits))
    lll = set(map(lambda x: x[1] << x[0], non_zero_bits_enumerated))
    lll.discard(0)
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
    
    def top_left(self) -> tuple:
        return (self.x0, self.y0)

    def top_right(self) -> tuple:
        return (self.x0 + self.w, self.y0)
    
    def bottom_left(self) -> tuple:
        return (self.x0, self.y0 + self.h)

    def bottom_right(self) -> tuple:
        return (self.x0 + self.w, self.y0 + self.h)
    
    def left_segment(self) -> 'Segment':
        return Segment(self.y0, self.y0 + self.h)

    def top_segment(self) -> 'Segment':
        return Segment(self.x0, self.x0 + self.w)
    
    def xywh(self) -> tuple:
        return (self.x0, self.y0, self.w, self.h)
    
    def xyxy(self) -> tuple:
        return (self.x0, self.y0, self.x0 + self.w, self.y0 + self.h)
    
    def width(self) -> int:
        return self.w
    
    def height(self) -> int:
        return self.h
    
    def wh(self) -> tuple:
        return (self.w, self.h)
    
    def xy(self) -> tuple:
        return (self.x0, self.y0)
    
    def sub_rect(self, sub: 'Rect') -> 'Rect':
        return Rect(self.x0 + sub.x0, self.y0 + sub.y0, *sub.wh())
    
    def moved(self, dx: int, dy: int) -> 'Rect':
        return Rect(self.x0 + dx, self.y0 + dy, *self.wh())
    
    def __add__(self, other: np.array) -> 'Rect':
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
    
    @classmethod
    def from_midpoint(cls, midpt:'point2d', r:int) -> 'Rect':
        return Rect.from_xyxy(*(midpt.xy - r), *(midpt.xy + r))
        return None
    
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
def exit_hotkey(key = '^q', ahk:autohotkey.AHK = None, handler: Callable[[], Any] = None):
    handler_name = 'exit'
    q = queue.Queue()
    # if ahk is None:
    #     ahk = autohotkey.AHK()
    # lambda: q.put('exit'), logging.info('exit hotkey handler')
    # ahk.start_hotkeys()
    def _handler_stub():
        if handler is not None:
            handler()
        q.put(handler_name)
    def get_command():
        if not q.empty():
            trigger_cmd = q.get()
            if trigger_cmd == handler_name:
                logging.info(f'triggered ${handler_name} handler')
                return True
        return False
    ahk.add_hotkey(key, _handler_stub)
    logging.info(f'start ${handler_name} handler')
    sys.stdout.flush()
    yield get_command
    logging.info(f'end ${handler_name} handler')
    sys.stdout.flush()
    ahk.remove_hotkey(key)

@contextlib.contextmanager
def hotkey_handler(key: str, cmd: str, ahk:autohotkey.AHK = None, handler: Callable[[], Any] = None):
    handler_name = cmd
    q = queue.Queue()
    def _handler_stub():
        if handler is not None:
            handler()
        q.put(handler_name)
    def get_command():
        if not q.empty():
            trigger_cmd = q.get()
            if trigger_cmd == cmd:
                logging.info(f'triggered ${handler_name} handler')
                return True
        return False
    ahk.add_hotkey(key, _handler_stub)
    logging.info(f'start ${handler_name} handler')
    sys.stdout.flush()
    yield get_command
    logging.info(f'end ${handler_name} handler')
    sys.stdout.flush()
    ahk.remove_hotkey(key)


@dataclass
class point2d:
    xy: np.ndarray
    def __init__(self, x: Union[int, float], y: Union[int, float]) -> None:
        """Initialize a 2D point with given x and y coordinates."""
        self.xy = np.array([x, y])

    def __repr__(self) -> str:
        """Return a string representation of the point."""
        return f"point2d({self.xy[0]}, {self.xy[1]})"

    def __eq__(self, other: object) -> bool:
        """Check if two points are equal."""
        if isinstance(other, point2d):
            return np.array_equal(self.xy, other.xy)
        return False
    
    def __neg__(self) -> 'point2d':
        return point2d(*(-self.xy))
    
    def __pos__(self) -> 'point2d':
        return point2d(*self.xy)

    def __add__(self, other: Union['point2d', int, float]) -> 'point2d':
        """Add two points."""
        if isinstance(other, point2d):
            return point2d(*(self.xy + other.xy))
        elif isinstance(other, (int, float)):
            return point2d(*(self.xy + other))
        return NotImplemented

    def __sub__(self, other: Union['point2d', int, float]) -> 'point2d':
        """Subtract one point from another."""
        if isinstance(other, point2d):
            return point2d(*(self.xy - other.xy))
        elif isinstance(other, (int, float)):
            return point2d(*(self.xy - other))
        return NotImplemented

    def __mul__(self, scalar: Union[int, float]) -> 'point2d':
        """Multiply the point by a scalar."""
        if isinstance(scalar, (int, float)):
            return point2d(*(self.xy * scalar))
        return NotImplemented

    def __truediv__(self, scalar: Union[int, float]) -> 'point2d':
        """Divide the point by a scalar."""
        if isinstance(scalar, (int, float)):
            return point2d(*(self.xy / scalar))
        return NotImplemented

    def __mod__(self, scalar: int) -> 'point2d':
        """Divide the point by a scalar."""
        if isinstance(scalar, int):
            return point2d(*(self.xy % scalar))
        return NotImplemented

    def __imod__(self, scalar: int) -> 'point2d':
        """Divide the point by a scalar."""
        if isinstance(scalar, int):
            self.xy %= scalar
            return self
        return NotImplemented

    def __floordiv__(self, scalar: Union[int, float]) -> 'point2d':
        """Floor divide the point by a scalar."""
        if isinstance(scalar, (int, float)):
            return point2d(*(self.xy // scalar))
        return NotImplemented

    def __iadd__(self, other: Union['point2d', int, float]) -> 'point2d':
        """In-place addition of two points."""
        if isinstance(other, point2d):
            self.xy += other.xy
            return self
        elif np.issubdtype(self.xy.dtype, np.integer) and isinstance(other, int):
            self.xy += other
            return self
        elif np.issubdtype(self.xy.dtype, np.floating) and isinstance(other, (int, float)):
            self.xy += other
            return self
        return NotImplemented

    def __isub__(self, other: Union['point2d', int, float]) -> 'point2d':
        """In-place subtraction of a point."""
        if isinstance(other, point2d):
            self.xy -= other.xy
            return self
        elif np.issubdtype(self.xy.dtype, np.integer) and isinstance(other, int):
            self.xy -= other
            return self
        elif np.issubdtype(self.xy.dtype, np.floating) and isinstance(other, (int, float)):
            self.xy -= other
            return self
        return NotImplemented

    # TODO: finish implementation of in-place operations
    def __imul__(self, scalar: Union[int, float]) -> 'point2d':
        """In-place multiplication of the point by a scalar."""
        if isinstance(scalar, (int, float)):
            self.xy *= scalar
            return self
        return NotImplemented

    def __itruediv__(self, scalar: Union[int, float]) -> 'point2d':
        """In-place division of the point by a scalar."""
        if isinstance(scalar, (int, float)):
            self.xy /= scalar
            return self
        return NotImplemented

    def __ifloordiv__(self, scalar: Union[int, float]) -> 'point2d':
        """In-place floor division of the point by a scalar."""
        if isinstance(scalar, (int, float)):
            self.xy //= scalar
            return self
        return NotImplemented
    
    def swapped(self):
        """Return underlying ndarray with optionally swapping coordinates"""
        return point2d(self.xy[1], self.xy[0])
    
    def int(self):
        return point2d(int(self.xy[0]), int(self.xy[1]))
    
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


@contextlib.asynccontextmanager
async def timeout_context_manager(timeout):
    exit_event = asyncio.Event()

    def premature_exit():
        """Function to trigger a premature exit."""
        logging.info("Premature exit triggered.")
        exit_event.set()

    print(f"Entering context, will wait for {timeout} seconds unless exited prematurely.")
    try:
        # Yield the function that can trigger a premature exit
        yield premature_exit

        logging.info(f"Waiting for {timeout} seconds or until a premature exit is triggered.")
        try:
            await asyncio.wait_for(exit_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logging.info("Exiting context after timeout.")
        else:
            logging.info("Exiting context due to premature exit.")
    finally:
        # Reset event or handle any necessary cleanup here
        exit_event.clear()    


@contextlib.asynccontextmanager
async def timeout_context_manager1(timeout):
    exit_event = asyncio.Event()

    async def premature_exit():
        logging.info("Premature exit triggered.")
        exit_event.set()

    try:
        logging.info(f"Entering context, will wait for {timeout} seconds unless exited prematurely.")
        yield premature_exit  # Yield the method to trigger early exit

    finally:
        if not exit_event.is_set():
            try:
                logging.info("Waiting for the context block to complete or for a timeout.")
                await asyncio.wait_for(exit_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                logging.info("Timeout reached: exiting context forcefully.")
            else:
                logging.info("Exiting context after successful completion.")
        exit_event.clear()



def alpha_blend(dst: np.ndarray, src: np.ndarray, alpha: float) -> np.ndarray:
    """ Alpha blend two images with alpha coefficient
    """
    assert 0 < alpha <= 1.0
    dst = cv.addWeighted(dst, alpha, src, 1.0 - alpha, 0)
    return dst

def draw_highlight(img, mask, ratio = 0.8, cl = (255, 0, 0), border: int = 1):
    """Draws a highlight on an image using a given mask.

    This function applies a highlight to the specified image by utilizing a mask. 
    The highlight is blended with the image at a certain ratio and appears in the 
    specified color.

    Args:
        img: The image on which the highlight is to be drawn. It should be a NumPy array or similar image representation.
        mask: The mask that determines the region of the image to be highlighted. It should be of the same size as `img`.
        ratio: A float value between 0 and 1 indicating the blending ratio of the highlight. Defaults to 0.8.
        cl: A tuple representing the color of the highlight in RGB format. Defaults to (255, 0, 0) for red.

    Returns:
        The image with the highlight applied. The return value will be of the same type as `img`, with the highlight modification.
    """
    assert len(img.shape) == 3 and len(mask.shape) == 2, 'img should be RGB image, mask should be binary image'
    assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1], 'image and the mask should be same size'
    gr = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
    contours, _ = cv.findContours(gr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mix = alpha_blend(img, mask, ratio)
    mix = mix & mask
    notmask = np.bitwise_not(mask)
    cutout = img & notmask
    final = mix | cutout
    cv.drawContours(final, contours, -1, cl, border)
    return final

def get_midpoint(im: np.ndarray) -> point2d:
    """ Get center point of an image
    """
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

'''Display image or list of images in Jupiter notebook
'''
def dis(*imgs: np.ndarray) -> None:
    pil_image_list = list(map(PILImage.fromarray, imgs))
    display(*pil_image_list)


def resize(im: np.ndarray, f: float) -> np.ndarray:
    return cv.resize(im, None, fx = f, fy = f)


SERVER_IP = '127.0.0.1'
SERVER_PORT = 12345
SHMEM_NAME = 'shmem_mask_analyzer'
SHMEM_DATA_SIZE = 1024 * 1024 * 256

OPCODE_RUN_SAM = 0xAB

