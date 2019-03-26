import numpy as np
import cv2
import dlib
import logging
import time
import argparse
from config import *
from face_detect_and_track import *
from face_points_detection import *
from face_swap import *


class VideoHandler(object):
    def __init__(self, video_path=0, detector_type="cascade", tracker_type="mosse", lose_threshold=10):
        self.cap = cv2.VideoCapture(video_path)
        self.detector = Detector()
        self.tracker = Tracker(tracker_type)

        self.lose_threshold = lose_threshold
        self.src_img = None

    def expand_bbox(self, x, y, w, h, w_resize_ratio=1.1, h_resize_ratio=1.3):
        x = np.max([0, x - int(w * 0.05)])
        y = np.max([0, y - int(h * 0.05)])
        w = int((x + w) * w_resize_ratio - x)
        h = int((y + h) * h_resize_ratio - y)
        return np.array([x, y, w, h])

    def set_src_img(self, img_path):
        self.src_img = cv2.imread(img_path)

    def process_src_img(self):
        src_face = self.detector.face_detection(self.src_img)
        if isinstance(src_face, int):
            raise Exception("No face detected in src image!")
        src_face_rect = self.bbox_to_rect(src_face)
        self.src_points = face_points_detection(self.src_img, src_face_rect)

        # shrink the size of src image, to speed up. Although it is not obvious.
        src_points_face = self.src_img.copy()
        for (point_index, point) in enumerate(self.src_points):
            cv2.circle(src_points_face,
                       (point[0], point[1]), 2, (0, 0, 255), -1)
        logging.info('''Select the Face and then press SPACE or ENTER button!
Cancel the selection process by pressing c button!''')
        while True:
            initBB = cv2.selectROI("src_roi", src_points_face,
                                   fromCenter=False, showCrosshair=False)
            if initBB != (0, 0, 0, 0):
                break
        cv2.destroyWindow("src_roi")
        (x, y, w, h) = initBB
        self.src_points -= (x, y)
        self.src_img = self.src_img[y:y + h, x:x + w]

        src_mask = mask_from_points(self.src_img.shape[:2], self. src_points)
        self.src_only_face = apply_mask(self.src_img, src_mask)

    def run_face_swap(self, dst_img, dst_face_rect: dlib.rectangle):
        if True:
            return self.fast_face_swap(dst_img, dst_face_rect)
        else:
            return self.slow_face_swap(dst_img, dst_face_rect)

    def fast_face_swap(self, dst_img, dst_face_rect: dlib.rectangle):
        success = 1
        failed = 0
        dst_points = face_points_detection(dst_img, dst_face_rect)
        if not check_points(dst_img, dst_points):
            logging.error("part of Face")
            return (failed, dst_img)

        dst_mask = mask_from_points(
            dst_img.shape[:2], dst_points, erode_flag=1)

        r = cv2.boundingRect(dst_points)
        (x, y, w, h) = r

        if y + h > dst_img.shape[0] or x + w > dst_img.shape[1]:
            return (failed, dst_img)

        dst_roi = dst_img[y:y + h, x:x + w]
        dst_mask = dst_mask[y:y + h, x:x + w]
        dst_points -= (x, y)

        dst_only_face = apply_mask(dst_roi, dst_mask)

        warped_src_face = warp_image_3d(
            self.src_only_face, self.src_points[:48], dst_points[:48], dst_roi.shape[:2])
        new_src_face = correct_colours(
            dst_only_face, warped_src_face, dst_points)

        # center=tuple(dst_points[33]+(x,y))
        center = (int(x + w / 2), int(y + h / 2))

        output = cv2.seamlessClone(
            new_src_face, dst_img, dst_mask, center, cv2.NORMAL_CLONE)
        return (success, output)

    def slow_face_swap(self, dst_img, dst_face_rect: dlib.rectangle):
        dst_points = face_points_detection(dst_img, dst_face_rect)  # 4ms
        w, h = dst_img.shape[:2]
        warped_dst_img = warp_image_3d(
            dst_img, dst_points[:48], self.src_points[:48], self.src_only_face.shape[:2])  # 140ms
        self.src_only_face = correct_colours(
            warped_dst_img, self.src_only_face, self.src_points)
        warped_src_img = warp_image_3d(
            self.src_only_face, self.src_points[:48], dst_points[:48], (w, h))
        dst_mask = mask_from_points((w, h), dst_points)
        r = cv2.boundingRect(dst_mask)
        center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
        output = cv2.seamlessClone(
            warped_src_img, dst_img, dst_mask, center, cv2.NORMAL_CLONE)
        return output

    def face_swap_2d(self, dst_img, dst_face_rect: dlib.rectangle):
        success = 1
        failed = 0
        dst_points = face_points_detection(dst_img, dst_face_rect)
        if not check_points(dst_img, dst_points):
            logging.error("part of Face")
            return (failed, dst_img)

        dst_mask = mask_from_points(
            dst_img.shape[:2], dst_points, erode_flag=1)

        r = cv2.boundingRect(dst_points)
        (x, y, w, h) = r

        if y + h > dst_img.shape[0] or x + w > dst_img.shape[1]:
            logging.error("part of landmarks out of face")
            return (failed, dst_img)

        dst_roi = dst_img[y:y + h, x:x + w]
        dst_mask = dst_mask[y:y + h, x:x + w]
        dst_points -= (x, y)

        dst_only_face = apply_mask(dst_roi, dst_mask)

        warped_src_face = warp_image_2d(self.src_only_face, transformation_from_points(
            dst_points, self.src_points), (h, w, 3))
        new_src_face = correct_colours(
            dst_only_face, warped_src_face, dst_points)
        center = (int(x + w / 2), int(y + h / 2))

        output = cv2.seamlessClone(
            new_src_face, dst_img, dst_mask, center, cv2.NORMAL_CLONE)
        return (success, output)

    # For DEBUG
    def draw_landmarks(self, dst_img, dst_face_bbox):
        dst_points = face_points_detection(dst_img, dst_face_bbox)
        for (point_index, point) in enumerate(dst_points):
            cv2.circle(dst_img, (point[0], point[1]), 2, (0, 0, 255), -1)
            cv2.putText(dst_img, str(
                point_index), (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    def bbox_to_rect(self, bbox)->dlib.rectangle:
        (x, y, w, h) = [int(v) for v in bbox]
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)
        return dlib_rect

    def rect_to_bbox(self, rect: dlib.rectangle):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return np.array([x, y, w, h])

    def _check_face_rect(self, face_rect, h, w):
        if face_rect.left() > 0 and face_rect.top() > 0 and face_rect.right() < w and face_rect.height() < h:
            return True
        else:
            return False

    def cascade_vh(self):
        face_flag = 0
        target_lose_cnt = 0
        while (cv2.waitKey(1) != 27):
            start_tc = cv2.getTickCount()
            grabbed, frame = self.cap.read()
            if not grabbed:
                logging.info("READ OVER")
                break

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
                    self.tracker.start_track(frame, *face_bbox)
                    target_lose_cnt = 0

            else:
                # track
                if target_lose_cnt < self.lose_threshold:
                    (success, box_predict) = self.tracker.update_track(frame)
                    if not success:
                        logging.debug("update failed")
                        target_lose_cnt += 1
                        cv2.imshow("frame", frame)
                        continue

                    (old_x, old_y, old_w, old_h) = (int(v)
                                                    for v in box_predict)

                    # draw predict rect
                    # self.draw_rect(frame,(old_x, old_y, old_w, old_h),(255,0,0))
                    # face_bbox is relative coordinates!!!!
                    face_bbox = self.detector.face_detection(
                        frame[old_y:old_y + old_h, old_x:old_x + old_w])
                    if isinstance(face_bbox, int):
                        target_lose_cnt += 1
                        cv2.imshow("frame", frame)
                        continue
                    target_lose_cnt = 0
                    face_bbox = self.expand_bbox(*face_bbox)
                    face_bbox[0] += int(old_x)
                    face_bbox[1] += int(old_y)

                # lose target
                else:
                    logging.info("losing target")
                    # reset tracker
                    self.tracker = Tracker()
                    face_flag = 0
                    cv2.imshow("frame", frame)
                    continue

            # self.draw_rect(frame,face_bbox,(0,255,0))
            face_rect = self.bbox_to_rect(face_bbox)
            success, frame = self.run_face_swap(frame, face_rect)
            if not success:
                pass

            cv2.imshow("output", frame)

            end_tc = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (end_tc - start_tc)
            logging.info("fps {}".format(fps))
        cv2.destroyAllWindows()
        self.cap.release()

    # For DEBUG
    def single_dst(self, dst_img):
        face_bbox = self.detector.face_detection(dst_img)
        self.expand_bbox(*face_bbox)
        face_rect = self.bbox_to_rect(face_bbox)
        output = self.fast_face_swap(dst_img, face_rect)
        return output

    def draw_rect(self, img, bbox, color=(255, 255, 255)):
        (x, y, w, h) = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), color)
        return img


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(lineno)d:%(message)s")

    parser = argparse.ArgumentParser(description='FaceSwap Video')
    parser.add_argument('--src_img', required=True,
                        help='Path for source image')
    parser.add_argument('--video_path', default=0,
                        help='Path for video')
    args = parser.parse_args()

    video_path = args.video_path
    test = VideoHandler(video_path)
    test.set_src_img(args.src_img)
    test.process_src_img()
    test.cascade_vh()
