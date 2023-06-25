import cv2 as cv
import numpy as np

# creating video capture object
video = cv.VideoCapture('vtest.avi')

# getting video details (height, width, fps)
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
frame_rate = int(video.get(cv.CAP_PROP_FPS))

# getting first frame of video and converting to binary
ret, first_frame = video.read()
cv.imshow('out', first_frame)
grey_first_frame = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

# creating heatmap window with dimensions same as of our video
heatmap = np.zeros_like(grey_first_frame, dtype=np.float64)

while True:
    # getting next frame
    ret, second_frame = video.read()

    # if there is a frame captured (helps to find the end of video)
    if ret:
        # converting to binary and storing differences with the previous frame in 'diff'
        grey_second_frame = cv.cvtColor(second_frame, cv.COLOR_BGR2GRAY)
        diff = cv.absdiff(grey_second_frame, grey_first_frame)

        # retrieving foreground(moving object) and removing some noises
        threshold = 50
        ret, foreground = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY)
        foreground_noise_removed = cv.morphologyEx(foreground, cv.MORPH_OPEN, (3, 3), iterations=3)

        # marking foreground on heatmap
        heatmap = cv.add(heatmap, foreground_noise_removed.astype(np.float64))

        # normalising values to ensure all pixel values ranges from 0 to 255 on every iteration(for every frame)
        normalized = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
        
        # applying colormap for heatmap
        colored = cv.applyColorMap(normalized, cv.COLORMAP_HOT)
        
        # adding colormap to video
        result = cv.addWeighted(second_frame, 0.8, colored, 0.5, 0)

        # showing outputs
        cv.imshow('out', result)
        cv.imshow("heat", colored)

        # changing first frame as our second frame
        grey_first_frame = grey_second_frame

    # press key 'q' to stop the video
    if cv.waitKey(frame_rate) == ord('q'):
        break

video.release()
cv.destroyAllWindows()