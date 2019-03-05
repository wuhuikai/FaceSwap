import cv2
PREDICTOR_PATH = "models/shape_predictor_68_face_landmarks.dat"
CASCADE_PATH="models/haarcascade_frontalface_default.xml"
OPENCV_OBJECT_TRACKERS = {
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
    
}