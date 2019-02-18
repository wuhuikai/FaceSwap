import numpy as np
import dlib
import cv2
import logging
import time
import math
from config import *
from face_detect_and_track import *
from face_points_detection import *
from face_swap import *
logging.basicConfig(level=logging.DEBUG,format="%(levelname)s:%(lineno)d:%(message)s")


class VideoHandler(object):
    def __init__(self,video_path=0,detector_type="cascade",tracker_type="mosse",lose_threshold=10):
        self.cap=cv2.VideoCapture(video_path)
        self.detector = Detector()
        self.tracker = Tracker(tracker_type)

        self.lose_threshold = lose_threshold
        self.src_img=None

    def expand_bbox(self,x, y, w, h):
        x = np.max([0, x - int(w * 0.10)])
        y = np.max([0, y - int(h * 0.10)])
        w = int(w * 1.2)
        h = int(h * 1.3)
        return (x, y, w, h)

    def set_src_img(self,img_path):
        self.src_img=cv2.imread(img_path)

    def process_src_img(self):
        src_face = self.detector.face_detection(self.src_img)
        if isinstance(src_face, int):
            logging.error("No face in src img")
            exit(0)
        src_face_rect=self.box_to_dlib_rect(src_face)
        self.src_points=face_points_detection(self.src_img,src_face_rect)
        src_mask = mask_from_points(self.src_img.shape[:2],self. src_points)
        self.src_only_face=apply_mask(self.src_img,src_mask)

    def run_face_swap(self,dst_img,dst_face_rect:dlib.rectangle):
        dst_points=face_points_detection(dst_img,dst_face_rect) #4ms
        w,h = dst_img.shape[:2]
        t0=time.time()
        warped_dst_img = warp_image_3d(dst_img, dst_points[:48], self.src_points[:48], self.src_only_face.shape[:2]) #140ms
        t1=time.time()
        logging.info(t1-t0)
        self.src_only_face = correct_colours(warped_dst_img, self.src_only_face, self.src_points)
        warped_src_img = warp_image_3d(self.src_only_face, self.src_points[:48], dst_points[:48], (w, h))
        dst_mask = mask_from_points((w, h), dst_points)
        r = cv2.boundingRect(dst_mask)
        center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
        output = cv2.seamlessClone(warped_src_img, dst_img, dst_mask, center, cv2.NORMAL_CLONE)
        return  output

    # For TEST
    # def run_face_swap(self,dst_img,dst_face_bbox):
    #     dst_points=face_points_detection(dst_img,dst_face_bbox)
    #     for (point_index,point) in enumerate(dst_points):
    #         # pt_pos=(point.x,point.y)
    #
    #         cv2.circle(dst_img, (point[0],point[1]), 2, (0,0,255), -1)
    #         cv2.putText(dst_img,str(point_index),(point[0],point[1]),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,255),1)
    #     return dst_img

    def box_to_dlib_rect(self,bbox):
        (x,y,w,h)=[int(v) for v in bbox]
        dlib_rect=dlib.rectangle(x,y,x+w,y+h)
        return dlib_rect

    def cascade_vh(self):
        face_flag = 0
        target_lose_cnt = 0
        while (cv2.waitKey(1) != 27):
            start_tc = cv2.getTickCount()
            grabbed, frame = self.cap.read()
            if not face_flag:
                face_bbox = self.detector.face_detection(frame)
                # No face has been detected
                if isinstance(face_bbox, int):
                    cv2.imshow("frame", frame)
                    continue
                # detect successfully
                else:
                    logging.info("tracker init")
                    face_flag = 1
                    face_bbox = self.expand_bbox(*face_bbox)
                    self.tracker.start_track(frame, face_bbox)
                    target_lose_cnt = 0
                    (x, y, w, h) = face_bbox

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)



            else:
                # track
                if target_lose_cnt < self.lose_threshold:
                    (success, box_predict) = self.tracker.update_track(frame)
                    if not success:
                        logging.info("update failed")
                        target_lose_cnt += 1
                        cv2.imshow("frame", frame)
                        continue

                    (old_x, old_y, old_w, old_h) = (int(v) for v in box_predict)

                    # draw predict rect
                    cv2.rectangle(frame, (old_x, old_y), (old_x + old_w, old_y + old_h), (0, 0, 255), 2)
                    # face_bbox is relative coordinates!!!!
                    face_bbox = self.detector.face_detection(frame[old_y:old_y + old_h, old_x:old_x + old_w])
                    if isinstance(face_bbox, int):
                        target_lose_cnt += 1
                        cv2.imshow("frame", frame)
                        continue
                    target_lose_cnt = 0
                    (x, y, w, h) = self.expand_bbox(*face_bbox)
                    x = int(old_x + x)
                    y = int(old_y + y)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    face_bbox[0]=x
                    face_bbox[1]=y

                # lose target
                else:
                    logging.info("losing target")
                    # reset tracker
                    self.tracker = Tracker()
                    face_flag = 0
                    cv2.imshow("frame", frame)
                    continue

            face_rect = self.box_to_dlib_rect(face_bbox)
            frame = self.run_face_swap(frame, face_rect)
            # frame_roi=frame[y:y+h,x:x+w]
            # face_rect = self.box_to_dlib_rect(face_bbox)
            # frame_roi = self.run_face_swap(frame_roi, face_rect)

            end_tc = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (end_tc - start_tc)
            logging.info("fps {}".format(fps))

            cv2.imshow("frame", frame)

        cv2.destroyAllWindows()
        self.cap.release()


    def dlib_vh(self):
        face_flag = 0
        target_lose_cnt = 0
        while (cv2.waitKey(1) != 27):
            start_tc = cv2.getTickCount()
            grabbed, frame = self.cap.read()
            if not face_flag:
                face_bbox = self.detector.face_detection(frame)
                # No face has been detected
                if isinstance(face_bbox, int):
                    cv2.imshow("frame", frame)
                    continue
                # detect successfully
                else:
                    logging.info("tracker init")
                    face_flag = 1
                    face_bbox = self.expand_bbox(*face_bbox)
                    self.tracker.start_track(frame, face_bbox)
                    target_lose_cnt = 0
                    (x, y, w, h) = face_bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            else:
                # track
                if target_lose_cnt < self.lose_threshold:
                    (success, box_predict) = self.tracker.update_track(frame)
                    if not success:
                        logging.info("update failed")
                        target_lose_cnt += 1
                        cv2.imshow("frame", frame)
                        continue
                    else:
                        (old_x, old_y, old_w, old_h) = (int(v) for v in box_predict)

                        # logging.debug("box_predict shape ({},{})".format(box_predict[2],box_predict[3]))

                        # maxY=np.min([old_y+old_h,frame.shape[0]])
                        # maxX=np.min([old_x+old_w,frame.shape[1]])
                        # bbox = d.cascade_method(frame[old_y:maxY,old_x:maxX])

                        # draw predict rect
                        cv2.rectangle(frame, (old_x, old_y), (old_x + old_w, old_y + old_h), (0, 0, 255), 2)
                        face_bbox = self.detector.face_detection(frame[old_y:old_y + old_h, old_x:old_x + old_w])
                        if isinstance(face_bbox, int):
                            target_lose_cnt += 1
                            cv2.imshow("frame", frame)
                            continue
                        target_lose_cnt = 0
                        (x, y, w, h) = self.expand_bbox(*face_bbox)
                        x = int(old_x + x)
                        y = int(old_y + y)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        face_bbox[0] = x
                        face_bbox[1] = y
                        face_rect = self.box_to_dlib_rect(face_bbox)
                # lose target
                else:
                    logging.info("losing target")
                    # reset tracker
                    self.tracker = Tracker()
                    face_flag = 0
                    continue
            face_rect = self.box_to_dlib_rect(face_bbox)
            frame = self.run_face_swap(frame, face_rect)
            cv2.imshow("frame", frame)

            end_tc = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (end_tc - start_tc)
            logging.info("fps {}".format(fps))
        cv2.destroyAllWindows()
        self.cap.release()
if __name__ == '__main__':
    video_path = "D:/1120/Git/Other/PracticalPythonAndOpenCV_CaseStudies-master/Chapter03/video/adrian_face.mov"
    test=VideoHandler()
    test.set_src_img("D:/1120/Git/Mywork/FaceSwap/imgs/test7.jpg")
    test.process_src_img()
    test.cascade_vh()