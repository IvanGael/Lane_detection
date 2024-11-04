import numpy as np
import cv2
from skimage.transform import resize 
from moviepy.editor import VideoFileClip
import tensorflow as tf

class LaneLines():
    def __init__(self):
        self.model = tf.keras.models.load_model('model.keras')
        self.recent_fit = []
        self.avg_fit = []
    
    def road_lines(self, image):
        original_shape = image.shape
        new_shape = (80, 160)
        small_img = resize(image, new_shape)
        small_img = np.array(small_img)
        small_img = small_img[None, :, :, :]

        prediction = self.model.predict(small_img)[0] * 255
        self.recent_fit.append(prediction)

        if len(self.recent_fit) > 5:
            self.recent_fit = self.recent_fit[1:]

        self.avg_fit = np.mean(self.recent_fit, axis=0)

        # Ensure avg_fit is 2D
        if len(self.avg_fit.shape) == 3:
            self.avg_fit = np.mean(self.avg_fit, axis=2)

        # Create a 3-channel image
        lane_drawn = np.zeros((self.avg_fit.shape[0], self.avg_fit.shape[1], 3), dtype=np.uint8)
        lane_drawn[:, :, 1] = self.avg_fit.astype(np.uint8)  # Put the lane lines in the green channel

        # Resize lane_drawn to match the original image size
        lane_image = cv2.resize(lane_drawn, (original_shape[1], original_shape[0]))

        print(f"road_lines - Image shape: {image.shape}, Lane image shape: {lane_image.shape}")
        result = cv2.addWeighted(image, 1, lane_image, 1, 0)

        return result

    def process_image(self, image):
        result = self.road_lines(image)
        return result

    def detect_edges(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        return edges

    def region_of_interest(self, image):
        height, width = image.shape
        polygon = np.array([[(0, height), (width, height), (width//2, height//2)]])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygon, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def draw_lines(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (10, 252, 232), 10)
        return line_image

    def hough_lines(self, image):
        edges = self.detect_edges(image)
        roi = self.region_of_interest(edges)
        lines = cv2.HoughLinesP(roi, 1, np.pi/180, 15, np.array([]), minLineLength=40, maxLineGap=20)
        line_image = self.draw_lines(image, lines)
        return line_image

    def weighted_img(self, img, initial_img, α=0.8, β=1., γ=0.):
        print(f"weighted_img - Img shape: {img.shape}, Initial img shape: {initial_img.shape}")
        return cv2.addWeighted(initial_img, α, img, β, γ)

    def pipeline(self, image):
        try:
            print(f"pipeline - Input image shape: {image.shape}")
            lane_image = self.process_image(image)
            print(f"pipeline - Lane image shape: {lane_image.shape}")
            hough_image = self.hough_lines(image)
            print(f"pipeline - Hough image shape: {hough_image.shape}")
            
            # Ensure hough_image has 3 channels
            if len(hough_image.shape) == 2:
                hough_image = cv2.cvtColor(hough_image, cv2.COLOR_GRAY2BGR)
            
            result = self.weighted_img(hough_image, lane_image)
            return result
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            return image 

laneLines = LaneLines()
 
input_video = "harder_challenge_video.mp4"
output_video = "harder_challenge_video_output.mp4"

clip = VideoFileClip(input_video)
processed_clip = clip.fl_image(laneLines.pipeline)
processed_clip.write_videofile(output_video, audio=False)
