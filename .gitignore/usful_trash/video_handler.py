import cv2
from imutils import face_utils 

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
    
    def visualisation(self, frame, detector, predictor):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Detect faces in the grayscale frame
        rects = detector(gray, 0)
        
        for (i, rect) in enumerate(rects):
            # Determine the facial landmarks for the face region
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Convert dlib's rectangle to an OpenCV-style bounding box (x, y, w, h)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show the face number
            cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Loop over the (x, y)-coordinates for the facial landmarks and draw them
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            
        # Display the resulting frame
        cv2.imshow("Facial Landmarks", frame)
