import os
import cv2
import logging
import argparse

from face_detection import select_face
from face_swap import face_swap


class VideoHandler(object):
    def __init__(self, video_path=0, img_path=None, args=None):
        self.src_points, self.src_shape, self.src_face = select_face(cv2.imread(img_path))
        if self.src_points is None:
            print('No face detected in the source image !!!')
            exit(-1)
        self.args = args
        self.video = cv2.VideoCapture(video_path)
        self.writer = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc(*'MJPG'), self.video.get(cv2.CAP_PROP_FPS),
                                      (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    def start(self):
        while self.video.isOpened():
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            _, dst_img = self.video.read()
            dst_points, dst_shape, dst_face = select_face(dst_img, choose=False)
            if dst_points is not None:
                dst_img = face_swap(self.src_face, dst_face, self.src_points, dst_points, dst_shape, dst_img, self.args, 68)
            self.writer.write(dst_img)
            if self.args.show:
                cv2.imshow("Video", dst_img)

        self.video.release()
        self.writer.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(lineno)d:%(message)s")

    parser = argparse.ArgumentParser(description='FaceSwap Video')
    parser.add_argument('--src_img', required=True,
                        help='Path for source image')
    parser.add_argument('--video_path', default=0,
                        help='Path for video')
    parser.add_argument('--warp_2d', default=False, action='store_true', help='2d or 3d warp')
    parser.add_argument('--correct_color', default=False, action='store_true', help='Correct color')
    parser.add_argument('--show', default=False, action='store_true', help='Show')
    parser.add_argument('--save_path', required=True, help='Path for storing output video')
    args = parser.parse_args()

    dir_path = os.path.dirname(args.save_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    VideoHandler(args.video_path, args.src_img, args).start()
