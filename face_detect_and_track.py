import numpy as np
import dlib
import cv2
import logging
import time
import math
import argparse
from config import *
logging.basicConfig(level=logging.DEBUG,
                    format="%(levelname)s:%(lineno)d:%(message)s")


class Detector(object):
    def __init__(self):
        self.detector = cv2.CascadeClassifier(CASCADE_PATH)

    def face_detection(self, frame)->np.ndarray:
        face_rects = self.__get_face_rects(frame)
        if not isinstance(face_rects, np.ndarray):
            logging.debug("No face")
            return 0

        if not len(face_rects) == 1:
            logging.error("Too many face")
            return 0

        return face_rects[0]

    def __get_face_rects(self, img):
        # 1.3 5
        face_rects = self.detector.detectMultiScale(
            img, scaleFactor=1.1, minNeighbors=2, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(120, 120))
        # face_rects = self.detector.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5,flags=cv2.CASCADE_SCALE_IMAGE,minSize=(120,120))
        # face_rect:ndarray
        if face_rects == ():
            return None
        return face_rects


class DlibDetector(object):
    def __init__(self)->dlib.rectangle:
        self.detector = dlib.get_frontal_face_detector()

    def __get_face_rects(self, img, upsample_num_times=0):
        # For faster speed, I recommend the param upsample_num_times == 0
        face_rects = self.detector(img, upsample_num_times)
        if not face_rects:
            return None  # face_rects=None
        return face_rects

    def rect_to_bb(self, rect: dlib.rectangle)->tuple:
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return (x, y, w, h)

    def dlib_method_test(self):
        cap = cv2.VideoCapture(0)
        while (cv2.waitKey(1) != 27):
            start_t = cv2.getTickCount()
            grabbed, frame = cap.read()
            face_rects = self.__get_face_rects(frame)
            if not face_rects:
                logging.info("No face")
                cv2.imshow("output", frame)
                continue
            if not len(face_rects) == 1:
                logging.error("Too many faces")
                break
            (x, y, w, h) = self.rect_to_bb(face_rects[0])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("output", frame)
            end_t = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (end_t - start_t)
            logging.info("fps: {0}".format(fps))

    def face_detection(self, frame, upsample_times=0):
        face_rects = self.__get_face_rects(frame, upsample_times)
        if not face_rects:
            logging.info("No face")
            return 0

        if not len(face_rects) == 1:
            logging.error("Too many faces")
            return 0

        return self.rect_to_bb(face_rects[0])


class Tracker(object):
    """docstring for Tracker"""
    # I really recommend use mosse, besides, KCF may be a good choice, but it has slow speed in windows. I guess KCF could work in ubuntu

    def __init__(self, tracker_type="mosse"):
        self.tracker = OPENCV_OBJECT_TRACKERS[tracker_type]()

    def start_track(self, frame, x, y, w, h):
        self.tracker.init(frame, (x, y, w, h))

    def update_track(self, frame):
        return self.tracker.update(frame)

    def clear(self):
        return self.tracker.clear()


def expand_bbox(x, y, w, h):
    x = np.max([0, x - int(w * 0.10)])
    y = np.max([0, y - int(h * 0.10)])
    w = int(w * 1.2)
    h = int(h * 1.3)
    return (x, y, w, h)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Human face detect and Track')
    parser.add_argument('--video_path', default=0, help='Path for video')
    args = parser.parse_args()
    video_path = args.parse_args()
    d = Detector()
    t = Tracker()

    cap = cv2.VideoCapture(video_path)
    face_flag = 0
    lose_threshold = 10
    target_lose_cnt = 0
    while (cv2.waitKey(1) != 27):
        start_tc = cv2.getTickCount()
        grabbed, frame = cap.read()
        if not grabbed:
            break

        if not face_flag:
            face_bbox = d.face_detection(frame)
            # No face has been detected
            if isinstance(face_bbox, int):
                cv2.imshow("output", frame)
                continue
            # detect successfully
            else:
                logging.info("tracker init")
                face_flag = 1
                face_bbox = expand_bbox(*face_bbox)
                t.start_track(frame, face_bbox)
                target_lose_cnt = 0
                (x, y, w, h) = face_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("output", frame)

        else:
            # track
            if target_lose_cnt < lose_threshold:
                (success, box_predict) = t.update_track(frame)
                if not success:
                    logging.info("update failed")
                    target_lose_cnt += 1
                    cv2.imshow("output", frame)
                    continue
                else:
                    (old_x, old_y, old_w, old_h) = (int(v)
                                                    for v in box_predict)

                    # logging.debug("box_predict shape ({},{})".format(box_predict[2],box_predict[3]))

                    # maxY=np.min([old_y+old_h,frame.shape[0]])
                    # maxX=np.min([old_x+old_w,frame.shape[1]])
                    # bbox = d.cascade_method(frame[old_y:maxY,old_x:maxX])

                    # draw predict rect
                    cv2.rectangle(frame, (old_x, old_y),
                                  (old_x + old_w, old_y + old_h), (0, 0, 255), 2)
                    face_bbox = d.face_detection(
                        frame[old_y:old_y + old_h, old_x:old_x + old_w])
                    if isinstance(face_bbox, int):
                        target_lose_cnt += 1
                        cv2.imshow("output", frame)
                        continue
                    target_lose_cnt = 0
                    (x, y, w, h) = expand_bbox(*face_bbox)
                    x = int(old_x + x)
                    y = int(old_y + y)
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 255, 0), 2)
                    cv2.imshow("output", frame)
            # lose target
            else:
                logging.info("losing target")
                # reset tracker
                t = Tracker()
                face_flag = 0
                continue
        end_tc = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (end_tc - start_tc)
        logging.info(fps)
    cv2.destroyAllWindows()
    cap.release()
