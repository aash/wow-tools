
import ahk as autohotkey
import d3dshot_stub as d3dshot
from win_tools import get_window_rect
from common import *
from overlay_client import overlay_client
from snail import Snail
from wow_client_utils import *


WOW_WINDOW_NAME = 'World of Warcraft'


def test_mapparser_getrate_capture():
    window_name = WOW_WINDOW_NAME
    ahk = autohotkey.AHK()
    window = ahk.find_window(title=window_name)
    window_id = int(window.id, 16)
    window.activate()
    r = get_window_rect(ahk, window_name)
    d3d = d3dshot.D3DShot(capture_output=d3dshot.CaptureOutputs.NUMPY, fps=60, roi=r)
    d3d.capture()
    n = 300
    t0 = millis_now()
    for i in range(n):
        img, t = d3d.wait_next_frame()
        logging.info(f'{i} new frame no: {t}')
        assert img.shape
        #cv2.imwrite(f'frame{i:06d}.bmp', img)
    dt = millis_now() - t0
    logging.info(f'time per frame: {dt / n} ms')
    logging.info(f'fps: {1000 * (n / dt)}')
    d3d.stop()


import cv2
import numpy as np
from matplotlib import pyplot as plt


from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np

def partition_objects(objects, metric, threshold):
    """
    Partition objects into groups based on a metric and a threshold.

    :param objects: List of objects.
    :param metric: Function that takes two objects and returns their "distance".
    :param threshold: The threshold for grouping based on the metric.
    :return: List of groups (each group is a list of objects).
    """
    n = len(objects)
    # Step 1: Compute the distance matrix
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = metric(objects[i], objects[j])

    # Step 2: Perform hierarchical clustering using single linkage
    condensed_distance_matrix = distance_matrix[np.triu_indices(n, 1)]
    Z = linkage(condensed_distance_matrix, method='single')

    # Step 3: Form clusters based on the threshold
    labels = fcluster(Z, t=threshold, criterion='distance')
    # Step 4: Group objects based on the clusters
    groups = [[] for _ in range(max(labels))]
    for i, label in enumerate(labels):
        groups[label - 1].append(objects[i])

    return groups

# Example usage: Objects are points in a 2D space for simplicity
# objects = [(1, 2), (2, 3), (3, 4), (8, 8), (9, 9), (10, 10)]
# def euclidean_metric(obj1, obj2):
    # return np.sqrt((obj1[0] - obj2[0])**2 + (obj1[1] - obj2[1])**2)

# threshold = 5  # Distance threshold for grouping
# partitioned_groups = partition_objects(objects, euclidean_metric, threshold)
# print("Partitioned Groups:", partitioned_groups)


def filter_lines_by_angle(lines, angle_threshold):
    angles = []
    filtered_lines = []
    measured_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate the angle of the line in degrees
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        measured_lines.append((line, angle))
        # angles.append(angle)


    # print(angles)
    # print(len(angles))
    def metric(x, y):
        return abs(x[1] - y[1])
    vec = partition_objects(measured_lines, metric, angle_threshold)
    # print(vec)
    largest_group = max(vec, key=len)
    s = sorted(vec, key=len)
    # print(largest_group)
    print(s[-1])
    print(s[-2])
    for line, a in s[-2]:
        filtered_lines.append(line)

    return filtered_lines

def highlight_segments(image, min_length, angle_threshold):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 80, apertureSize=3)
    
    # Detect line segments using the Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, 
                            rho=1, 
                            theta=np.pi/180, 
                            threshold=50, 
                            minLineLength=min_length, 
                            maxLineGap=20)
    

    if lines is not None:

        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(image, (x1, y1), (x2, y2), (0,255,0), 5)
        # Filter lines based on angle
        filtered_lines = filter_lines_by_angle(lines, angle_threshold)
        
        # Highlight the filtered line segments on the original image
        # for line in filtered_lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    
    return image

def monitor_bobber(im, fishing_line_color):
    fl = cv.inRange(im, fishing_line_color, fishing_line_color)
    non_black_coords = np.argwhere(fl > 0)
    if len(non_black_coords) > 5:
        center = np.array(fl.shape) // 2
        distances = np.sqrt((non_black_coords[:, 0] - center[0]) ** 2 + (non_black_coords[:, 1] - center[1]) ** 2)
        farthest_index = np.argmax(distances)
        farthest_pixel_coords = non_black_coords[farthest_index]
        yx = farthest_pixel_coords
        xy = np.array((yx[1], yx[0]))
        ofs = np.array((10,10))
        fr = im.copy()
        cv.circle(fr, xy, 3, (255, 0,0), 1)
        cv.rectangle(fr, xy - ofs, xy + ofs, (0, 255,0), 1 )
        return fr, True, xy
    return im, False, None

def test_get_tooltip1():
    i = 0
    with overlay_client() as ovl_show_img, Snail() as s, hotkey_handler('^q', 'exit') as cmd_exit, \
         hotkey_handler('^1', 'calibrate') as cmd_calib, \
         hotkey_handler('^2', 'fish') as start_fishing, \
         timeout(3000) as is_not_timeout:
        fishing_line_color = np.array((122,  90, 58))
        # fishing_line_color1 = np.array((126,  88,  48))
        state = 'idle'
        ii = 0
        while is_not_timeout():
            im = s.wait_next_frame()
            out_img = im.copy()
            if cmd_exit() == 'exit':
                logging.info('exiting')
                break
            if cmd_calib() == 'calibrate':
                logging.info('calibrating')
                # s.ahk.send('^{F1}')
                h, w, _ = im.shape
                h = h // 10
                w = w // 3
                img = crop_image(im, Rect(w, 0, w, h))
                mask = np.zeros((h, w), dtype=np.uint8)
                cv.rectangle(mask, Rect(w//3, 0, w//3, h).xywh(), (255), -1) 
                seg, m = run_grabcut(img, mask)
                pixels = seg.reshape(-1, 3)
                unique_colors_set = set([tuple(pixel.astype(int)) for pixel in pixels])
                unique_colors = sorted(unique_colors_set)
                unique_colors.remove((0,0,0))
                unique_colors = list(unique_colors)
                # ovl_show_img(seg)
                # time.sleep(1)
                if len(unique_colors) != 1:
                    logging.info('calibration unsuccessful: could not properly segment fishing line')
                else:
                    fishing_line_color = np.array(unique_colors[0])
                    ovl_show_img(hstack([img, seg]))
                    time.sleep(1)
                    # cv.imwrite(f'tmp/fishline{i}.png', seg)
                    # cv.imwrite(f'tmp/frame{i}.png', im)
                logging.info(fishing_line_color)
                # logging.info(fishing_line_color1)
            
            if start_fishing() == 'fish':
                if state == 'fishing':
                    state = 'idle'
                else:
                    state = 'fishing'
            
            if state == 'fishing':
                s.ahk.send('c')
                ll = 0
                time.sleep(3)
                logging.info(f'state is {state}')
                state = 'monitor-bobber'
                with timeout(2) as is_bobber_not_timeout:
                    while is_bobber_not_timeout():
                        im = s.wait_next_frame()
                        out_img, bobber_found, bxy = monitor_bobber(im, fishing_line_color)
                        if bobber_found:
                            break
                        time.sleep(0.01)
                    if not bobber_found:
                        state = 'idle'
                    

            if state == 'monitor-bobber':
                out_img, bobber_found, bxy = monitor_bobber(im, fishing_line_color)
                if bobber_found:
                    bobber_wh = np.array([80, 80])
                    ofs = bobber_wh // 2
                    bobber_img = crop_image(im, Rect(*(bxy - ofs), *bobber_wh))
                    bobber_msk = np.zeros(bobber_wh, dtype=np.uint8)
                    cv.rectangle(bobber_msk, bobber_wh // 4, 3 * bobber_wh // 4, (255), -1)
                    bobber_segmented, qwe = run_grabcut(bobber_img, bobber_msk)
                    if is_bobber_drown(bobber_segmented):
                        time.sleep(0.3)
                        s.ahk.mouse_move(x=bxy[0], y=bxy[1])
                        time.sleep(0.5)
                        s.ahk.right_click()
                        cv.rectangle(out_img, Rect(*(bxy - ofs), *bobber_wh).xywh(), (0, 255, 0), 2)
                        ovl_show_img(out_img)
                        time.sleep(1)
                        state = 'fishing'
                    # cv.imwrite(f'tmp/bobber/{ii:04d}.bmp', hstack((bobber_img, bobber_segmented)))
                    ii += 1

                if not bobber_found:
                    cv.imwrite(f'tmp/bobber_not_found{i:0d}.bmp', im)
                    cv.imwrite(f'tmp/bobber_not_found{i:0d}_.bmp', out_img)
                    logging.info(f'state is {state}')
                    state = 'idle'
            
            ovl_show_img(out_img)
            i += 1
            time.sleep(0.010)
