import cv2
import dlib
import json
import argparse

## Face detection
def face_detection(img,upsample_times=1):
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, upsample_times)

    return faces

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Detection')
    parser.add_argument('--img_path', required=True, help='Path for input image')
    parser.add_argument('--out', required=True, help='Path for storing bboxes of faces')
    args = parser.parse_args()

    # ## Debug
    # args.img_path = 'imgs/multi_faces.jpg'
    # args.out = 'results/multi_faces.faces.json'

    # Read images
    img = cv2.imread(args.img_path)
    faces = face_detection(img)
    bboxs = []

    for face in faces:
        bboxs.append((face.left(), face.top(), face.right(), face.bottom()))

    with open(args.out, 'w') as f:
        json.dump(bboxs, f)