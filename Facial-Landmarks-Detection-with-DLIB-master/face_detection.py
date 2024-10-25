from imutils import face_utils
import dlib
import cv2

from eye import Eye

class Face:
    def __init__(self, shape_predictor_path):
        if not cv2.os.path.isfile(shape_predictor_path):
            raise FileNotFoundError(f"Shape predictor file not found at {shape_predictor_path}.")

        self.shape_predictor_path = shape_predictor_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.shape_predictor_path)
        self.frame = None

        self.right_eye = None
        self.left_eye = None
    
    def _analyze(self):
        """
            Method detects face using dlib, and saves to self.face_landmarks
        """

        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray,0)

        if len(faces) == 0:
            raise ValueError("No face detected")

        face = faces[0]
        landmarks = self.predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye = landmarks[36:42] 
        right_eye = landmarks[42:48]

        self.left_eye = Eye(original_frame=self.frame, landmarks=left_eye)
        self.right_eye = Eye( original_frame=self.frame, landmarks=right_eye)

    def refresh(self, frame):
        self.frame = frame
        self._analyze()

    def closed_eyes(self, threshold = 0.19):
        """
            Returns  left_eye_closed : bool, right_eye_closed : bool
        """
        left_eye_closed = self.left_eye.ear <= threshold
        right_eye_closed = self.right_eye.ear <= threshold

        return left_eye_closed, right_eye_closed
    
    def gaze_detection(self):
        if not self.left_eye.pupils_detected() or not self.right_eye.pupils_detected():
            return None, None
        left_pupil_pos = [self.left_eye.pupil.x, self.left_eye.pupil.y]
        left_eye_hight, left_eye_width = self.left_eye.frame.shape[:2] # (y, x)
        left_center = [left_eye_width/2, left_eye_hight/2]
        
        left_eye_vector = self._eye_offset(left_pupil_pos, left_center)

        right_pupil_pos = [self.right_eye.pupil.x, self.right_eye.pupil.y]
        right_eye_hight, right_eye_width = self.right_eye.frame.shape[:2] # (y, x)
        right_center = [right_eye_width/2, right_eye_hight/2]

        right_eye_vector = self._eye_offset(right_pupil_pos, right_center)

        return left_eye_vector, right_eye_vector
            

    def highlight_landmarks(self):
        if(self.right_eye is None or self.left_eye is None) or (not self.left_eye.pupils_detected() or not self.right_eye.pupils_detected()):
            return self.frame
        
        frame_with_landmarks = self.frame.copy()

        for (x, y) in self.left_eye.landmark_points:
            cv2.circle(frame_with_landmarks, (x, y), 1, (0, 255, 0), -1)

        for (x, y) in self.right_eye.landmark_points:
            cv2.circle(frame_with_landmarks, (x, y), 1, (0, 255, 0), -1)

        left_pupil_pos = (self.left_eye.origin[0] + self.left_eye.pupil.x, self.left_eye.origin[1] + self.left_eye.pupil.y)
        right_pupil_pos = (self.right_eye.origin[0] + self.right_eye.pupil.x, self.right_eye.origin[1] + self.right_eye.pupil.y)

        cv2.circle(frame_with_landmarks, left_pupil_pos, 2, (0, 0, 255), -1)
        cv2.circle(frame_with_landmarks, right_pupil_pos, 2, (0, 0, 255), -1)

        return frame_with_landmarks
    
    @staticmethod
    def _eye_offset(pupil_pos, eye_center):
        """
            Returns the position
        """
        #print(pupil_pos, eye_center)
        pupil_pos[0] -= eye_center[0]
        pupil_pos[1] -= eye_center[1]

        return pupil_pos

import imutils
if __name__ == "__main__":
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    image = cv2.imread("images\left.jpg")

    detected_face = Face(predictor_path)
    iamge = imutils.resize(image, width=600)

    detected_face.refresh(image)
    cv2.imshow("face", detected_face.frame)
    cv2.imshow("left eye", detected_face.left_eye.frame)
    cv2.imshow("left iris", detected_face.left_eye.pupil.iris_frame)
    cv2.imshow("landmarks", detected_face.highlight_landmarks())
    #print(detected_face.right_eye.pupil.x, detected_face.right_eye.pupil.y)

    cv2.waitKey(0)