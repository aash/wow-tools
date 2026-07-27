from math import inf
from dataclasses import dataclass
import numpy as np
import cv2
from copy import deepcopy
from typing import Tuple

@dataclass
class Segment:
    left: int
    right: int

@dataclass
class Rect:
    x0: int
    y0: int
    w: int
    h: int

    def __hash__(self):
        return hash((self.x0, self.y0, self.w, self.h))
    
    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.xywh() == other
        elif isinstance(other, list):
            return list(self.xywh()) == other
        elif isinstance(other, Rect):
            return (self.x0, self.y0, self.w, self.h) == (other.x0, other.y0, other.w, other.h)
        else:
            return False

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return ','.join(map(str, self.xywh()))

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
    
    def xywh(self) -> Tuple[int, int, int, int]:
        return (self.x0, self.y0, self.w, self.h)
    
    def xyxy(self):
        return (self.x0, self.y0, self.x0 + self.w, self.y0 + self.h)
    
    def width(self):
        return self.w
    
    def height(self):
        return self.h
    
    def wh(self) -> np.ndarray:
        return np.array((self.w, self.h))
    
    def xy(self):
        return np.array((self.x0, self.y0))
    
    def sub_rect(self, sub: 'Rect'):
        return Rect(self.x0 + sub.x0, self.y0 + sub.y0, *sub.wh())
    
    def moved(self, dx: int, dy: int):
        return Rect(self.x0 + dx, self.y0 + dy, self.w, self.h)
    
    def __add__(self, other: np.ndarray):
        return Rect(self.x0 + other[0], self.y0 + other[1], self.w, self.h)
    
    def center(self) -> np.ndarray:
        return np.array((self.x0 + self.w // 2, self.y0 + self.h // 2))
    
    def union(self, other: 'Rect'):
        nx = min(self.x0, other.x0)
        ny = min(self.y0, other.y0)
        nw = max(self.x0 + self.w, other.x0 + other.w) - nx
        nh = max(self.y0 + self.h, other.y0 + other.h) - ny
        return Rect(nx, ny, nw, nh)

    def clip(self, clipping_rect: 'Rect'):
        nx = max(self.x0, clipping_rect.x0)
        ny = max(self.y0, clipping_rect.y0)
        nw = min(self.x0 + self.w, clipping_rect.x0 + clipping_rect.w) - nx
        nh = min(self.y0 + self.h, clipping_rect.y0 + clipping_rect.h) - ny
        return Rect(nx, ny, nw, nh)


    @classmethod
    def from_centdims(cls, cx, cy, w, h) -> 'Rect':
        """ Constructs Rect from center point (cx, cy) and dimensions (w, h)
        """
        return Rect(cx - w // 2, cy - h // 2, w, h)

    @classmethod
    def from_str(cls, s:str) -> 'Rect':
        """ Constructs Rect from string representation 'x, y, w, h' or 'Rect(x, y, w, h)'
        """
        assert isinstance(s, str), 'parameter should be str(ing)'
        if s.startswith('Rect'):
            # TODO: implement parsing of Rect(a, b, c ,d) strings
            return Rect(0, 0, 0, 0)
        return Rect(*map(int, s.split(',')))

    @classmethod
    def from_ptdm(cls, pt: Tuple, dim: Tuple) -> 'Rect':
        """ Constructs Rect from top left point pt and dimensions dim
        """
        return Rect(*pt, *dim)
    
    @classmethod
    def from_xyxy(cls, x0: int, y0: int, x1: int, y1: int) -> 'Rect':
        """ Constructs Rect from unordered diagonal points (x0, y0) and (x1, y1)
        """
        p0, p1 = sorted((x0, x1))
        q0, q1 = sorted((y0, y1))
        return Rect(p0, q0, p1 - p0, q1 - q0)
    
    @classmethod
    def from_top_left(cls, x: int, y: int, w: int, h: int) -> 'Rect':
        """ Constructs Rect from top left point (x, y) and dimensions (w, h)
        """
        return Rect(x, y, w, h)

    @classmethod
    def from_bottom_left(cls, x: int, y: int, w: int, h: int) -> 'Rect':
        """ Constructs Rect from bottom left point (x, y) and dimensions (w, h)
        """
        return Rect(x, y - h, w, h)

    @classmethod
    def from_bottom_right(cls, x: int, y: int, w: int, h: int) -> 'Rect':
        """ Constructs Rect from bottom right point (x, y) and dimensions (w, h)
        """
        return Rect(x - w, y - h, w, h)

    @classmethod
    def from_top_right(cls, x: int, y: int, w: int, h: int) -> 'Rect':
        """ Constructs Rect from top right point (x, y) and dimensions (w, h)
        """
        return Rect(x - w, y, w, h)


def crop_image(img: np.ndarray, r: Rect) -> np.ndarray:
    """
    Crops a part of an image using a rectangle defined by the top-left corner, width, and height.
    If the rectangle goes beyond the image boundaries, it will be truncated.
    """

    r = deepcopy(r)

    x0 = max(0, r.x0)
    y0 = max(0, r.y0)
    x1 = min(img.shape[1], r.x0 + r.w)
    y1 = min(img.shape[0], r.y0 + r.h)
    return img[y0:y1, x0:x1].copy()


def translate_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate image by integer (dx, dy) using array slice copy.

    Creates a same-size canvas. Pixels outside the original bounds are filled
    with zeros. Positive dx shifts content right, positive dy shifts content
    down.

    Args:
        img: Input image (any dtype, any number of channels)
        dx: Horizontal shift in pixels
        dy: Vertical shift in pixels

    Returns:
        Translated image of same shape and dtype as input
    """
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    sx, sy = max(0, dx), max(0, dy)
    ox, oy = max(0, -dx), max(0, -dy)
    copy_h = min(h, h + dy) - max(0, dy)
    copy_w = min(w, w + dx) - max(0, dx)
    if copy_h <= 0 or copy_w <= 0:
        return out
    out[sy:sy + copy_h, sx:sx + copy_w] = img[oy:oy + copy_h, ox:ox + copy_w]
    return out


def blend_translated(images: list, offsets: list) -> np.ndarray:
    """Blend multiple images at given integer offsets into a single canvas.

    Each image is placed at its (dx, dy) offset relative to the first image
    (offset[0] is typically (0, 0)). Overlapping pixels are averaged weighted
    by the number of contributing images.

    Args:
        images: List of images (same dtype, same height/width)
        offsets: List of (dx, dy) integer offsets, one per image.
                 offset[i] is the position of image i's top-left corner
                 relative to the composite canvas origin.

    Returns:
        Blended composite image (uint8)
    """
    assert len(images) == len(offsets)
    img_h, img_w = images[0].shape[:2]

    min_x = min(dx for dx, dy in offsets)
    min_y = min(dy for dx, dy in offsets)
    max_x = max(dx + img_w for dx, dy in offsets)
    max_y = max(dy + img_h for dx, dy in offsets)

    canvas_h = max_y - min_y
    canvas_w = max_x - min_x

    acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    count = np.zeros((canvas_h, canvas_w, 1), dtype=np.float64)

    for img, (dx, dy) in zip(images, offsets):
        x = dx - min_x
        y = dy - min_y
        rgb = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        acc[y:y + img_h, x:x + img_w] += rgb.astype(np.float64)
        count[y:y + img_h, x:x + img_w] += 1

    return (acc / np.maximum(count, 1)).astype(np.uint8)
