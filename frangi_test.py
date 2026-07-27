import cv2
import numpy as np
from wow_client_utils import remove_small_connected_components, leave_large_ccs
from skimage.morphology import skeletonize, thin
from skimage.filters import frangi


def find_filtered_centroid(img, n_largest_to_remove=2):
    """
    Removes the N largest connected components from a binary image 
    and calculates the centroid of the remaining components.
    
    Args:
        image_path (str): Path to the binary mask image.
        n_largest_to_remove (int): Number of largest components to remove. 
                                   (e.g., 2 removes the background and the largest foreground object)
                                   
    Returns:
        tuple: (cX, cY) coordinates of the centroid, or None if calculation fails.
        numpy.ndarray: The filtered binary mask for visualization.
    """

    # Ensure the image is strictly binary (0 or 255)
    # _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    binary = img

    # 2. Run connected component analysis
    # connectivity=8 checks all surrounding pixels including diagonals
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Ensure we have enough components to remove and still calculate a centroid
    if num_labels <= n_largest_to_remove:
        print("Not enough components found in the image.")
        return None, None

    # 3. Extract the areas of all components (Area is the 5th column in stats)
    areas = stats[:, cv2.CC_STAT_AREA]

    # 4. Find the indices of the N largest components
    # argsort() sorts ascending, so [-N:] gets the last N elements (the largest)
    largest_indices = np.argsort(areas)[-n_largest_to_remove:]

    # 5. Create a new blank mask to hold our filtered components
    filtered_mask = np.zeros_like(binary)

    # Iterate through all labels. If a label is NOT in our largest list, draw it on the new mask
    for i in range(num_labels):
        if i not in largest_indices:
            filtered_mask[labels == i] = 255

    # 6. Calculate the global center of mass of the remaining pixels using moments
    M = cv2.moments(filtered_mask)
    
    # Avoid division by zero if the filtered mask ends up completely empty
    if M["m00"] == 0:
        print("Filtered mask is empty. Cannot calculate centroid.")
        return None, filtered_mask
        
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return (cX, cY), filtered_mask


def dilate(img: np.ndarray, sz: int, shape):
    el = cv2.getStructuringElement(shape, (2 * sz + 1, 2 * sz + 1), (sz, sz))
    return cv2.dilate(img, el)

def erode(img: np.ndarray, sz: int, shape):
    el = cv2.getStructuringElement(shape, (2 * sz + 1, 2 * sz + 1), (sz, sz))
    return cv2.erode(img, el)

def apply_frangi_filter(img, pre_dilate_size=1, post_dilate_size=1):
    # 2. Convert to Grayscale
    # The Frangi filter operates on a single-channel intensity map, not color.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = dilate(gray, pre_dilate_size, cv2.MORPH_DIAMOND)
    # gray = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    # 3. Apply the Frangi Filter
    # black_ridges=False: We are looking for bright lines (string/rod) on a dark background (water).
    # sigmas: Defines the thickness of the lines to detect. We use small values (1 to 3) 
    # because the fishing string is extremely thin (often 1-2 pixels wide).
    # print("Applying Frangi filter... this might take a moment depending on image size.")
    frangi_out = frangi(gray, sigmas=range(1, 4, 2), black_ridges=False, )

    # 4. Normalize for OpenCV
    # scikit-image outputs a float array (often very small numbers). 
    # We must normalize it to an 8-bit integer format (0-255) so OpenCV can display it.
    frangi_norm = cv2.normalize(frangi_out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    frangi_norm = dilate(frangi_norm, post_dilate_size, cv2.MORPH_ELLIPSE)


    # 5. Thresholding (Optional but recommended)
    # Convert the normalized Frangi output into a strict binary mask (black and white)
    # to isolate the strongest line detections. You may need to tune the '50' value.
    _, binary_mask = cv2.threshold(frangi_norm, 20, 255, cv2.THRESH_BINARY)


    binary_mask = remove_small_connected_components(binary_mask, min_size=100)

    # binary_mask = dilate(binary_mask, 5, cv2.MORPH_ELLIPSE)
    # binary_mask = erode(binary_mask, 7, cv2.MORPH_ELLIPSE)


    binary_mask = thin(binary_mask > 0)
    binary_mask = (binary_mask * 255).astype(np.uint8)
    # binary

    (cx, cy), msk = find_filtered_centroid(binary_mask)

    cv2.circle(img, (cx, cy), 10, (255,0,0), 3, -1)


    # binary_mask = dilate(binary_mask, 7, cv2.MORPH_ELLIPSE)
    # binary_mask = erode(binary_mask, 7, cv2.MORPH_ELLIPSE)

    binary_mask = thin(binary_mask > 0)
    binary_mask = (binary_mask * 255).astype(np.uint8)


 
    return binary_mask, frangi_norm, (cx, cy)

# Run the function
if __name__ == "__main__":
    img = cv2.imread("tmp/bobber.png")
    binary_mask, frangi_norm, cc = apply_frangi_filter(img)


    large_ccs = leave_large_ccs(binary_mask, cc_num=2)
 
    cv2.imshow("Original 'char_crop.png'", img)
    cv2.imshow("Frangi Filter Response", frangi_norm)
    cv2.imshow("Final Binary Mask", binary_mask)
    # cv2.imshow("thresh", binary_mask)
    cv2.imshow("large_ccs", large_ccs)
    cv2.imwrite('bobber_mask.png', binary_mask)

    # print("Press any key in the image windows to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 6. Display the results
