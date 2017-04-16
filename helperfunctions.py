import numpy as np
import cv2
from scipy import signal

# Use color transforms, gradients, etc., to create a thresholded binary image.Use color transforms, gradients, etc.,
# to create a thresholded binary image.


#Applying Sobel
def abs_sobel_thresh(img, orient='x',thresh_min=0, thresh_max=255):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  #image read with OpenCV so using COLOR_RGB2GRAY
    # Calculate the derivative in the x direction (the 1, 0 at the end denotes x direction)
    if orient == 'x':
        derivative = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y': #Calculate the derivative in the y direction (the 0, 1 at the end denotes y direction):
        derivative = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # Calculate the absolute value of the derivative
    abs_derivative = np.absolute(derivative)

    # Convert the absolute value image to 8-bit
    scaled_sobel = np.uint8(255 * abs_derivative / np.max(abs_derivative))

    # Create a binary threshold to select pixels based on gradient strength
    sobel_binary_output_mask = np.zeros_like(scaled_sobel)
    sobel_binary_output_mask[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return this mask as your binary_output image
    return sobel_binary_output_mask

#Magnitude of the Gradient
def mag_thresh(img, sobel_kernel=9, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Rescale to 8 bit
    scale_factor = np.uint8(255 * gradmag / np.max(gradmag))

    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary_output_mask = np.zeros_like(scale_factor)
    mag_binary_output_mask[(scale_factor >= mag_thresh[0]) & (scale_factor <= mag_thresh[1])] = 1

    # Return this mask as your binary_output image
    return mag_binary_output_mask

#Direction of the Gradient
def dir_threshold(image, sobel_kernel=15, thresh=(0, np.pi / 2)):
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the gradient direction
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    # direction of the gradient is arctangent of y gradient divided by the x gradien
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)

    # apply a threshold, and create a binary image mask
    dir_binary_output_mask = np.zeros_like(absgraddir)
    dir_binary_output_mask[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    #Return this mask as your binary_output
    return dir_binary_output_mask


#Combining Thresholds
def apply_thresholds(image, ksize=9):
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x',thresh_min=150,thresh_max=255)
    grady = abs_sobel_thresh(image, orient='y',thresh_min=70,thresh_max=200)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(70, 255))
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0.1, np.pi/2))

    # Combine thresholds
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined
#Combining Thresholds Revised as Grayscale had inferior performance
def apply_threshold_rev(image, xgrad_thresh=(20,100), s_thresh=(170,255)):
    # Grayscaling loses color information for the lane lines
    # Switch to other colors spaces
    # Applying Sobel
    sobel_binary_output_mask = abs_sobel_thresh(image,orient='x',thresh_min=xgrad_thresh[0],thresh_max=xgrad_thresh[1])

    # Convert to HLS colour space and separate the S channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    #Threshold colour channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine thresholds
    combined_binary = np.zeros_like(sobel_binary_output_mask)
    combined_binary[(s_binary == 1) | (sobel_binary_output_mask == 1)] = 1

    return combined_binary

#Apply a perspective transform to rectify binary image ("birds-eye view")

#Detect lane pixels and fit to find lane boundary.

def get_pixel_in_window(img, x_center, y_center, size):

    half_size = size // 2
    window = img[int(y_center - half_size):int(y_center + half_size), int(x_center - half_size):int(x_center + half_size)]

    x, y = (window.T == 1).nonzero()

    x = x + x_center - half_size
    y = y + y_center - half_size

    return x, y


def collapse_into_single_arrays(leftx, lefty, rightx, righty):
    leftx = [x
             for array in leftx
             for x in array]
    lefty = [x
             for array in lefty
             for x in array]
    rightx = [x
              for array in rightx
              for x in array]
    righty = [x
              for array in righty
              for x in array]

    leftx = np.array(leftx)
    lefty = np.array(lefty)
    rightx = np.array(rightx)
    righty = np.array(righty)

    return leftx, lefty, rightx, righty

def histogram_pixels(warped_thresholded_image, offset=50, steps=6,
                     window_radius=200, medianfilt_kernel_size=51,
                     horizontal_offset=50):
    # Initialise arrays
    left_x = []
    left_y = []
    right_x = []
    right_y = []

    # Parameters
    height = warped_thresholded_image.shape[0]
    offset_height = height - offset
    width = warped_thresholded_image.shape[1]
    half_frame = warped_thresholded_image.shape[1] // 2
    pixels_per_step = offset_height / steps

    for step in range(steps):
        left_x_window_centres = []
        right_x_window_centres = []
        y_window_centres = []

        # Define the window
        window_start_y = height - (step * pixels_per_step) + offset
        window_end_y = window_start_y - pixels_per_step + offset

        # Take a histogram of the botton of the image
        histogram = np.sum(warped_thresholded_image[int(window_end_y):int(window_start_y), int(horizontal_offset):int(width - horizontal_offset)], axis=0)

        # Smoothen the histogram
        histogram_smooth = signal.medfilt(histogram, medianfilt_kernel_size)

        # Identify the left and right peaks
        left_peaks = np.array(signal.find_peaks_cwt(histogram_smooth[:half_frame], np.arange(1, 10)))
        right_peaks = np.array(signal.find_peaks_cwt(histogram_smooth[half_frame:], np.arange(1, 10)))
        if len(left_peaks) > 0:
            left_peak = max(left_peaks)
            left_x_window_centres.append(left_peak)

        if len(right_peaks) > 0:
            right_peak = max(right_peaks) + half_frame
            right_x_window_centres.append(right_peak)

        # Add coordinates to window centres

        if len(left_peaks) > 0 or len(right_peaks) > 0:
            y_window_centres.append((window_start_y + window_end_y) // 2)

        # Get pixels in the left window
        for left_x_centre, y_centre in zip(left_x_window_centres, y_window_centres):
            left_x_additional, left_y_additional = get_pixel_in_window(warped_thresholded_image, left_x_centre,
                                                                       y_centre, window_radius)
            # plt.scatter(left_x_additional, left_y_additional)
            # Add pixels to list
            left_x.append(left_x_additional)
            left_y.append(left_y_additional)

        # Get pixels in the right window
        for right_x_centre, y_centre in zip(right_x_window_centres, y_window_centres):
            right_x_additional, right_y_additional = get_pixel_in_window(warped_thresholded_image, right_x_centre,
                                                                         y_centre, window_radius)
            # Add pixels to list
            right_x.append(right_x_additional)
            right_y.append(right_y_additional)

    return collapse_into_single_arrays(left_x, left_y, right_x, right_y)



def fit_second_order_poly(indep, dep, return_coeffs=False):
    fit = np.polyfit(indep, dep, 2)
    fitdep = fit[0]*indep**2 + fit[1]*indep + fit[2]
    if return_coeffs == True:
        return fitdep, fit
    else:
        return fitdep

def evaluate_poly(indep, poly_coeffs):
    return poly_coeffs[0]*indep**2 + poly_coeffs[1]*indep + poly_coeffs[2]

def highlight_lane_line_area(mask_template, left_poly, right_poly, start_y=0, end_y =720):
    area_mask = mask_template
    for y in range(start_y, end_y):
        left = evaluate_poly(y, left_poly)
        right = evaluate_poly(y, right_poly)
        area_mask[y][int(left):int(right)] = 1

    return area_mask


def center(y, left_poly, right_poly):
    center = (1.5 * evaluate_poly(y, left_poly)
              - evaluate_poly(y, right_poly)) / 2
    return center


def add_figures_to_image(img, curvature, vehicle_position, min_curvature, left_coeffs=(0,0,0), right_coeffs=(0,0,0)):

    # Convert from pixels to meters
    vehicle_position = vehicle_position / 12800 * 3.7
    curvature = curvature / 128 * 3.7
    min_curvature = min_curvature / 128 * 3.7

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Radius of Curvature = %d(m)' % curvature, (50, 50), font, 1, (255, 255, 255), 2)
    left_or_right = "left" if vehicle_position < 0 else "right"
    cv2.putText(img, 'Vehicle is %.2fm %s of center' % (np.abs(vehicle_position), left_or_right), (50, 100), font, 1,
                (255, 255, 255), 2)
    cv2.putText(img, 'Min Radius of Curvature = %d(m)' % min_curvature, (50, 150), font, 1, (255, 255, 255), 2)
    cv2.putText(img, 'Left poly coefficients = %.3f %.3f %.3f' % (left_coeffs[0], left_coeffs[1], left_coeffs[2]), (50, 200), font, 1, (255, 255, 255), 2)
    cv2.putText(img, 'Right poly coefficients = %.3f %.3f %.3f' % (right_coeffs[0], right_coeffs[1], right_coeffs[2]), (50, 250), font, 1, (255, 255, 255), 2)

#weed out the outliers
def plausible_curvature(left_curverad, right_curverad):
    if right_curverad < 500 or left_curverad < 500:
        return False
    else:
        return True
#weed out the outliers
def plausible_continuation_of_traces(left_coeffs, right_coeffs, prev_left_coeffs, prev_right_coeffs):
    if prev_left_coeffs == None or prev_right_coeffs == None:
        return True
    b_left = np.absolute(prev_left_coeffs[1] - left_coeffs[1])
    b_right = np.absolute(prev_right_coeffs[1] - right_coeffs[1])
    if b_left > 0.3 or b_right > 0.3:
        return False
    else:
        return True


#Warp the detected lane boundaries back onto the original image

def lane_poly(yval, poly_coeffs):
    """Returns x value for poly given a y-value.
    Note here x = Ay^2 + By + C."""
    return poly_coeffs[0]*yval**2 + poly_coeffs[1]*yval + poly_coeffs[2]


def draw_poly(img, poly, poly_coeffs, steps, color=[255, 0, 0], thickness=10, dashed=False):
    img_height = img.shape[0]
    pixels_per_step = img_height // steps

    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step

        start_point = (int(poly(start, poly_coeffs=poly_coeffs)), start)
        end_point = (int(poly(end, poly_coeffs=poly_coeffs)), end)

        if dashed == False or i % 2 == 1:
            img = cv2.line(img, end_point, start_point, color, thickness)

    return img
