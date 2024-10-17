import cv2

class VideoHandler:
    def __init__(self, source):
        # source : int or str
        # str - video path
        # int - webcam index
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise ValueError(f"could not open video source {source}")
    
    def read_frame(self):
        #returns ret:bool, frame:np.ndarray
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("End of video or cannot get the frame")
        return ret , frame
    
    def release(self):
        # release the video capture
        self.cap.release()

    def get_fps(self):
        #Returns the frames per second of the video source.
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame_dimensions(self):
        #Returns the width and height of the video frames.
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height
