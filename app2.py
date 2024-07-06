import numpy as np
import cv2
from skimage.transform import resize 
from moviepy.editor import VideoFileClip
import tensorflow as tf

# Load the model using the new .keras format
model = tf.keras.models.load_model('model.keras')  

class Runways():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

runways = Runways()

def runway_lines(image):
    original_shape = image.shape
    new_shape = (160, 320)  # Increased size to capture more of the runway
    small_img = resize(image, new_shape)
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]

    prediction = model.predict(small_img)[0] * 255
    runways.recent_fit.append(prediction)

    if len(runways.recent_fit) > 5:
        runways.recent_fit = runways.recent_fit[1:]

    runways.avg_fit = np.mean(runways.recent_fit, axis=0)

    if len(runways.avg_fit.shape) == 3:
        runways.avg_fit = np.mean(runways.avg_fit, axis=2)

    runway_drawn = np.zeros((runways.avg_fit.shape[0], runways.avg_fit.shape[1], 3), dtype=np.uint8)
    runway_drawn[:, :, 1] = runways.avg_fit.astype(np.uint8)  # Put the runway lines in the green channel

    runway_image = cv2.resize(runway_drawn, (original_shape[1], original_shape[0]))

    result = cv2.addWeighted(image, 1, runway_image, 1, 0)

    return result

def process_image(image):
    result = runway_lines(image)
    return result

def detect_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def region_of_interest(image):
    height, width = image.shape
    # Modified polygon to focus on the center of the image where the runway is likely to be
    polygon = np.array([[(width//4, height), (3*width//4, height), (3*width//4, height//3), (width//4, height//3)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def hough_lines(image):
    edges = detect_edges(image)
    roi = region_of_interest(edges)
    # Modified parameters to detect longer, straighter lines
    lines = cv2.HoughLinesP(roi, 1, np.pi/180, 50, np.array([]), minLineLength=100, maxLineGap=50)
    line_image = draw_lines(image, lines)
    return line_image

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def pipeline(image):
    try:
        runway_image = process_image(image)
        hough_image = hough_lines(image)
        
        if len(hough_image.shape) == 2:
            hough_image = cv2.cvtColor(hough_image, cv2.COLOR_GRAY2BGR)
        
        result = weighted_img(hough_image, runway_image)
        return result
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        return image 

# Process video
input_video = "lanes_clip.mp4"
output_video = "output_video.mp4"

clip = VideoFileClip(input_video)
processed_clip = clip.fl_image(pipeline)
processed_clip.write_videofile(output_video, audio=False)