import cv2
import numpy as np

class BobberTracker:
    def __init__(self, movement_threshold=15.0, smoothing_factor=0.2):
        """
        Initializes the robust vertical bobber tracker.
        
        :param movement_threshold: The sudden downward pixel distance required to trigger a catch.
        :param smoothing_factor: (0.0 to 1.0) Lower values ignore gentle wave bobbing better. 
        """
        self.movement_threshold = movement_threshold
        self.alpha = smoothing_factor
        
        # Tracks the baseline "resting" position of the bobber
        self.resting_cx = None
        self.resting_cy = None
        
        # State flag to prevent multiple triggers for the same catch
        self.catch_triggered = False

    def extract_top_circle(self, img):
        """
        Robustly locates the small circle at the top of a (potentially fragmented) bobber mask.
        """
        # Handle colored masks (like green on white/transparent)
        if len(img.shape) >= 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # Ensure standard binary mask (white lines on black background)
        if gray[0, 0] > 127:
            gray = cv2.bitwise_not(gray)
            
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Extract topology
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is None:
            return None
            
        valid_holes = []
        
        # Find ALL internal holes, regardless of what parent they belong to
        for i, h_data in enumerate(hierarchy[0]):
            parent_idx = h_data[3]
            
            # If parent_idx != -1, this contour is inside another contour (it's a hole)
            if parent_idx != -1:
                x, y, w, h = cv2.boundingRect(contours[i])
                
                # Filter out massive gaps or 1-pixel noise
                if 2 <= w <= 25 and 2 <= h <= 25:
                    valid_holes.append(i)
                    
        if not valid_holes:
            return None
            
        # The bobber ring is always the highest valid hole (Minimum Y)
        best_cx, best_cy = None, float('inf')
        
        for h_idx in valid_holes:
            M = cv2.moments(contours[h_idx])
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                if cy < best_cy:
                    best_cy = cy
                    best_cx = cx
                    
        if best_cx is not None:
            return (best_cx, best_cy)
            
        return None

    def update(self, frame_img):
        """
        Feeds a new frame into the tracker.
        Returns True if a sudden downward vertical movement is detected.
        """
        current_pos = self.extract_top_circle(frame_img)
        
        if current_pos is None:
            # Bobber lost or fragmented too much in this frame, do nothing
            return False
            
        curr_x, curr_y = current_pos
        
        # Initialize resting position on the first valid frame
        if self.resting_cx is None or self.resting_cy is None:
            self.resting_cx = curr_x
            self.resting_cy = curr_y
            return False
            
        # Calculate vertical distance (A bite pulls the bobber down, increasing Y)
        delta_y = curr_y - self.resting_cy
        
        # Check for sudden downward spike (ignoring X completely)
        if delta_y > self.movement_threshold:
            if not self.catch_triggered:
                self.catch_triggered = True
                print(f"🎣 CATCH DETECTED! Vertical Drop: {delta_y:.1f}px")
                return True
        else:
            # Reset trigger if the bobber returns to a stable position
            self.catch_triggered = False
            
        # Update the EWMA baseline to slowly track gentle wave movements in both axes
        self.resting_cx = (self.alpha * curr_x) + ((1.0 - self.alpha) * self.resting_cx)
        self.resting_cy = (self.alpha * curr_y) + ((1.0 - self.alpha) * self.resting_cy)
        
        return False


