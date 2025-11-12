
from common import point2d
import numpy as np


def test_point2d_binops_unaryops_int():
    """ test binary operations and unary ops on int parameters
    """
    x = point2d(1, 2)
    y = point2d(2, 1)
    z = point2d(5, 9)
    assert np.array_equal((-x).xy, -(x.xy))
    assert +x == x
    assert x + y == point2d(3, 3)
    assert -x - y == point2d(-3, -3)
    assert x * 2 == point2d(2, 4)
    assert x / 2 == point2d(0.5, 1)
    assert x // 2 == point2d(0, 1)
    assert z // 2 == point2d(2, 4)
    assert x + (-y) == point2d(-1, 1)
    assert x % 2 == point2d(1, 0)

def test_point2d_binops_unaryops_mix_int_float():
    """ test binary operations and unary ops on mixed parameters
    """

def test_point2d_inplace_ops():
    """ test in-place operations
    """
    x = point2d(1, 2)
    y = point2d(2, 1)
    # z = point2d(5, 9)
    x += y
    assert x == point2d(3,3)
    x += 1
    assert x == point2d(4,4)
    x += 1.5
    assert x == point2d(5.5,5.5)
    
    
