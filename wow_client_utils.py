import cv2 as cv
import cv2
import numpy as np
import itertools

def count_non_black_pixels(img):
    # Returns the count of non-black pixels in the image
    return cv2.countNonZero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

def calculate_angle(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

def calculate_line_length(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def find_best_canny_threshold_and_draw_lines(img, min_thresh_step=50, max_thresh_step=250, step_size=1, angle_tolerance=2):
    height, width, _ = img.shape
    
    best_thresholds = (0, 0)
    max_total_length = -float('inf')
    min_non_black_pixels = float('inf')
    best_lines_img = None

    for low_thresh in range(min_thresh_step, max_thresh_step, step_size):
        for high_thresh in range(low_thresh + step_size, max_thresh_step, step_size):
            edges = cv2.Canny(img, low_thresh, high_thresh)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
            
            if lines is None or len(lines) < 1:
                continue
            
            dominant_angle = calculate_angle(lines[0][0][0], lines[0][0][1], lines[0][0][2], lines[0][0][3])
            consistent_lines = []
            total_length = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = calculate_angle(x1, y1, x2, y2)
                if abs(angle - dominant_angle) < angle_tolerance:
                    consistent_lines.append(line)
                    line_length = calculate_line_length(x1, y1, x2, y2)
                    total_length += line_length

            if len(consistent_lines) < 1:
                continue
            
            lines_img = np.zeros_like(img)
            for line in consistent_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_img, (x1, y1), (x2, y2), (255, 255, 255), 2)

            combined_img = cv2.bitwise_or(edges, cv2.cvtColor(lines_img, cv2.COLOR_BGR2GRAY))
            non_black_pixels_count = count_non_black_pixels(cv2.cvtColor(combined_img, cv2.COLOR_GRAY2BGR))

            if total_length > max_total_length or (total_length == max_total_length and non_black_pixels_count < min_non_black_pixels):
                max_total_length = total_length
                min_non_black_pixels = non_black_pixels_count
                best_thresholds = (low_thresh, high_thresh)
                best_lines_img = lines_img
                
    return best_thresholds, best_lines_img


def run_grabcut(image, binary_mask):
    # Define the mask for the grabCut algorithm with the same size as the image
    grabcut_mask = np.zeros(image.shape[:2], np.uint8)

    # Mark the sure background (0) and probable background (2) regions in the grabcut mask
    grabcut_mask[binary_mask == 0] = cv2.GC_BGD  # Definitely background
    grabcut_mask[binary_mask == 255] = cv2.GC_PR_FGD  # Probably foreground
    
    # You can enhance the mask by labeling definite foreground regions (optional)
    # For example, if you have certain information about definite foreground, you can set:
    # grabcut_mask[definite_foreground_mask == 1] = cv2.GC_FGD

    # Create the models for GrabCut (they are just placeholders, required by the function)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply the grabCut algorithm
    cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    # Create the final mask where definite and probable foreground are set to 1, background to 0
    final_mask = np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')

    # Extract the segmented image using the final mask
    segmented_image = cv2.bitwise_and(image, image, mask=final_mask)

    return segmented_image, final_mask

def is_bobber_drown(segmented_bobber_img):
    seg = segmented_bobber_img
    h, w, _ = seg.shape
    bobber_wh = np.array((w, h))
    gr = cv.cvtColor(seg, cv.COLOR_RGB2GRAY)
    msk = (gr != 0).astype(np.uint8) * 255
    sp = itertools.product([-1, 0, 1], [-1, 0, 1])
    center_pixels_9 = [msk[*(bobber_wh // 2 + ofs)] for ofs in sp]
    return not np.any(center_pixels_9)
