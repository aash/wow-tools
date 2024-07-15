
import win32gui
import ahk
from common import Rect

def get_window_rect(ahk: ahk.AHK, window_name: str) -> Rect:
    window = ahk.find_window(title=window_name)
    window_id = int(window.id, 16)
    window.activate()
    client_area_zero = win32gui.ClientToScreen(window_id, (0,0))
    cr = win32gui.GetClientRect(window_id)
    return Rect(client_area_zero[0], client_area_zero[1], cr[2], cr[3])
