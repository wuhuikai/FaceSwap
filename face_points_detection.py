#! /usr/bin/env python
import json
import cv2
import dlib
import argparse
import numpy as np

PREDICTOR_PATH = 'models/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
## Face and points detection
def face_points_detection(img, bbox):
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, bbox)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    coords = np.asarray(list([p.x, p.y] for p in shape.parts()), dtype=np.int)

    # return the array of (x, y)-coordinates
    return coords

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceSwap Demo')
    parser.add_argument('--img_path', required=True, help='Path for input image')
    parser.add_argument('--bbox_path', required=True, help='Path for bboxes')
    parser.add_argument('--index', type=int, default=0, help='Which bbox to use')
    parser.add_argument('--out', required=True, help='Path for storing face points')
    args = parser.parse_args()

    # ## Debug
    # args.img_path = 'imgs/multi_faces.jpg'
    # args.bbox_path = 'results/multi_faces.faces.json'
    # args.index = 0
    # args.out = 'results/multi_faces.points.json'

    # Read images
    img = cv2.imread(args.img_path)
    with open(args.bbox_path) as f:
        bbox = dlib.rectangle(*json.load(f)[args.index])

    # Array of corresponding points
    points = face_points_detection(img, bbox)

    with open(args.out, 'w') as f:
        json.dump(points, f)