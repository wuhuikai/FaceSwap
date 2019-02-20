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
            raise Exception("No face detected in src image!")
        src_face_rect=self.box_to_dlib_rect(src_face)
        self.src_points=face_points_detection(self.src_img,src_face_rect)
        src_mask = mask_from_points(self.src_img.shape[:2],self. src_points)
        self.src_only_face=apply_mask(self.src_img,src_mask)

    def run_face_swap(self,dst_img,dst_face_rect:dlib.rectangle):
        if True:
            return self.fast_face_swap(dst_img,dst_face_rect)
        else:
            return self.slow_face_swap(dst_img, dst_face_rect)


    def fast_face_swap(self,dst_img,dst_face_rect:dlib.rectangle):
        h,w=dst_img.shape[:2]
        dst_points = face_points_detection(dst_img, dst_face_rect)  # 4ms
        dst_mask = mask_from_points((h,w), dst_points)
        dst_only_face = apply_mask(dst_img, dst_mask)

        warped_src_img = warp_image_3d(self.src_img, self.src_points[:48], dst_points[:48],(h,w) )
        src_only_face = correct_colours(dst_only_face, warped_src_img, dst_points)

        center=tuple(dst_points[33])
        
        output = cv2.seamlessClone(warped_src_img, dst_img, dst_mask, center, cv2.NORMAL_CLONE)
        return output


    def slow_face_swap(self,dst_img,dst_face_rect:dlib.rectangle):
        dst_points=face_points_detection(dst_img,dst_face_rect) #4ms
        w,h = dst_img.shape[:2]
        warped_dst_img = warp_image_3d(dst_img, dst_points[:48], self.src_points[:48], self.src_only_face.shape[:2]) #140ms
        self.src_only_face = correct_colours(warped_dst_img, self.src_only_face, self.src_points)
        warped_src_img = warp_image_3d(self.src_only_face, self.src_points[:48], dst_points[:48], (w, h))
        dst_mask = mask_from_points((w, h), dst_points)
        r = cv2.boundingRect(dst_mask)
        center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
        output = cv2.seamlessClone(warped_src_img, dst_img, dst_mask, center, cv2.NORMAL_CLONE)
        return  output

    # For DEBUG
    def draw_landmarks(self,dst_img,dst_face_bbox):
        dst_points=face_points_detection(dst_img,dst_face_bbox)
        for (point_index,point) in enumerate(dst_points):
            # pt_pos=(point.x,point.y)
    
            cv2.circle(dst_img, (point[0],point[1]), 2, (0,0,255), -1)
            cv2.putText(dst_img,str(point_index),(point[0],point[1]),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,255),1)
        

    def box_to_dlib_rect(self,bbox):
        (x,y,w,h)=[int(v) for v in bbox]
        dlib_rect=dlib.rectangle(x,y,x+w,y+h)
        return dlib_rect

    def _check_face_rect(self,face_rect,h,w):
        if face_rect.left() > 0 and face_rect.top()>0 and face_rect.right()<w and face_rect.height()<h:
            return True
        else :
            return False

    def cascade_vh(self):
        face_flag = 0
        target_lose_cnt = 0
        while (cv2.waitKey(1) != 27):
            start_tc = cv2.getTickCount()
            grabbed, frame = self.cap.read()
            if not face_flag:
                face_bbox:np.ndarray = self.detector.face_detection(frame)
                # No face has been detected
                if isinstance(face_bbox, int):
                    cv2.imshow("output", frame)
                    continue
                # detect successfully
                else:
                    logging.info("tracker init")
                    face_flag = 1
                    face_bbox = self.expand_bbox(*face_bbox)
                    self.tracker.start_track(frame, face_bbox)
                    target_lose_cnt = 0


            else:
                # track
                if target_lose_cnt < self.lose_threshold:
                    (success, box_predict) = self.tracker.update_track(frame)
                    if not success:
                        logging.debug("update failed")
                        target_lose_cnt += 1
                        cv2.imshow("output", frame)
                        continue

                    (old_x, old_y, old_w, old_h) = (int(v) for v in box_predict)

                    # draw predict rect
                    # cv2.rectangle(frame, (old_x, old_y), (old_x + old_w, old_y + old_h), (0, 0, 255), 2)

                    # face_bbox is relative coordinates!!!!
                    face_bbox = self.detector.face_detection(frame[old_y:old_y + old_h, old_x:old_x + old_w])
                    if isinstance(face_bbox, int):
                        target_lose_cnt += 1
                        cv2.imshow("output", frame)
                        continue
                    target_lose_cnt = 0
                    (x, y, w, h) = self.expand_bbox(*face_bbox)

                    # Convert it to absolute coordinates
                    x = int(old_x + x)
                    y = int(old_y + y)
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    face_bbox[0]=x
                    face_bbox[1]=y

                # lose target
                else:
                    logging.info("target lost")
                    # reset tracker
                    self.tracker = Tracker()
                    face_flag = 0
                    cv2.imshow("output", frame)
                    continue

            face_rect:dlib.rectangle = self.box_to_dlib_rect(face_bbox)
            # Notice that dlib could extract the face's landmarks even we did not provide full face.
            # It's a pretty rough solution. Maybe there is a method with more accurate
            # if not self._check_face_rect(face_rect,*frame.shape[:2]):
            #     logging.info("face out frame")
            #     continue
            # TODO: I still can't solve the problem when part of the face is out of picture, even             # I add function "_check_face_rect(self,face_rect,h,w)".
            try:
                frame=self.run_face_swap(frame, face_rect)
            except :
                continue
            else:
                end_tc = cv2.getTickCount()
                fps = cv2.getTickFrequency() / (end_tc - start_tc)
                logging.info("fps {}".format(fps))
                cv2.imshow("frame", frame)

        cv2.destroyAllWindows()
        self.cap.release()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format="%(levelname)s:%(lineno)d:%(message)s")

    video_path = "D:/1120/Git/Other/PracticalPythonAndOpenCV_CaseStudies-master/Chapter03/video/adrian_face.mov"
    test=VideoHandler()
    test.set_src_img("D:/1120/Git/Mywork/FaceSwap/imgs/test7.jpg")
    test.process_src_img()
    test.cascade_vh()