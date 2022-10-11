#! /usr/bin/env python
import os
import random
import cv2
import argparse
from time import sleep
import dlib
import numpy as np

"""
References: 
https://github.com/codeniko/shape_predictor_81_face_landmarks
https://github.com/davisking/dlib-models
https://github.com/davisking/dlib
https://github.com/ageitgey/face_recognition
https://github.com/spmallick/learnopencv



datasets:
https://github.com/yinguobing/facial-landmark-dataset
"""

from face_detection import select_face, select_all_faces
from face_swap import face_swap

def detect_face_using_mmod_face_detector(img,upsample_times=1):
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    detector = dlib.cnn_face_detection_model_v1('detectors/mmod_human_face_detector.dat')
    print('using detectors/mmod_human_face_detector.dat')
    faces = detector(img, upsample_times)
    print('Number of faces detected: {}'.format(len(faces)))

    return [trim_css_to_bounds(rect_to_css(face.rect), img.shape) for face in faces]

def detect_face_using_frontal_face_detector(img,upsample_times=1):
    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    detector = dlib.get_frontal_face_detector()
    print('using dlib.get_frontal_face_detector')
    faces = detector(img, upsample_times)
    print('Number of faces detected: {}'.format(len(faces)))

    return [trim_css_to_bounds(rect_to_css(face), img.shape) for face in faces]

def predict_landmarks(predictor, img, face):
    # Get the landmarks/parts for the face.
    shape = predictor(img, css_to_rect(face))
    print("found {} shapes".format(shape.num_parts))
    coords = np.asarray([(p.x, p.y) for p in shape.parts()], dtype=int)
    return coords

def predict_5_face_landmarks(img, face):
    predictor = dlib.shape_predictor('predictors/shape_predictor_5_face_landmarks.dat')
    print('using predictors/shape_predictor_5_face_landmarks.dat')
    return predict_landmarks(predictor, img, face)

def predict_68_face_landmarks(img, face):
    predictor = dlib.shape_predictor('predictors/shape_predictor_68_face_landmarks.dat')
    print('using predictors/shape_predictor_68_face_landmarks.dat')
    return predict_landmarks(predictor, img, face)

def predict_68_GTX_face_landmarks(img, face):
    predictor = dlib.shape_predictor('predictors/shape_predictor_68_face_landmarks_GTX.dat')
    print('using predictors/shape_predictor_68_GTX_face_landmarks.dat')
    return predict_landmarks(predictor, img, face)

def predict_81_face_landmarks(img, face):
    predictor = dlib.shape_predictor('predictors/shape_predictor_81_face_landmarks.dat')
    print('using predictors/shape_predictor_81_face_landmarks.dat')
    return predict_landmarks(predictor, img, face)

def rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order
    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()

def css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])    

def trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.
    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
thickness=2

def save_tagged_image(src_img, faces, name):
    image = src_img.copy()
    for idx, face in enumerate(faces):
        color=colors[idx % 3]
        cv2.rectangle(image, (face[3], face[0]), (face[1], face[2]), color, thickness)
    cv2.imwrite(name, image)


def save_landmarked_image(src_img, faces, name):
    image = src_img.copy()
    for idx, face in enumerate(faces):
        color=colors[idx % 3]
        for idx, point in enumerate(face):
            cv2.circle(image, point, 2, color, thickness)
    cv2.imwrite(name, image)

def save_delauney_image(src_img, subdiv, name):
    image = src_img.copy()
    triangles = subdiv.getTriangleList()

    for t in triangles :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        cv2.line(image, pt1, pt2, (255, 0, 0), 1)
        cv2.line(image, pt2, pt3, (255, 0, 0), 1)
        cv2.line(image, pt3, pt1, (255, 0, 0), 1)

    cv2.imwrite(name, image)

def save_voronai_image(src_img, subdiv, name):
    image = src_img.copy()
    (facets, centers) = subdiv.getVoronoiFacetList([])

    for (facet, center) in zip(facets, centers):
        ifacet = np.array(facet, int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.fillConvexPoly(image, ifacet, color)
        cv2.polylines(image, [ifacet], True, (0, 0, 0), 1)
        cv2.circle(image, (int(center[0]), int(center[1])), 3, (0, 0, 0))

    cv2.imwrite(name, image)


def subdiv2D(img, landmarks): 
    size = img.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    for p in landmarks :
        subdiv.insert((int(p[0]), int(p[1])))

    return subdiv


# Check if a point is inside a rectangle
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[0] + rect[2] :
        return False
    elif point[1] > rect[1] + rect[3] :
        return False
    return True

def calculateDelaunayTriangles(rect, points):
    #create subdiv
    subdiv = cv2.Subdiv2D(rect)
    
    # Insert points into subdiv
    for p in points:
        subdiv.insert((int(p[0]), int(p[1])))
    
    triangleList = subdiv.getTriangleList()
    
    delaunayTri = []
    
    pt = []    
        
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])        
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            #Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)    
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph 
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        
        pt = []        
            
    
    return delaunayTri

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

def swap(src_img, src_face_landmarks, dest_img, dest_face_landmarks):
    # src_hull = cv2.convexHull(src_face_landmarks, returnPoints = True)
    # src_hull = np.asarray([(p[0][0], p[0][1]) for p in src_hull], dtype=int)
    # save_landmarked_image(src_img, [src_hull], "output/src-hull.jpg")

    # src_subdiv = subdiv2D(src_img, src_hull)
    # save_delauney_image(src_img, src_subdiv, "output/src-delauney.jpg")
    # save_voronai_image(src_img, src_subdiv, "output/src-voronai.jpg")
    # landmarks -> convex hull
    # Delaunay Triangulation
    # affinity warp triangles
    # clone
    # Find convex hull
    warp_img = np.copy(dest_img)

    src_hull = []
    dest_hull = []

    hullIndex = cv2.convexHull(dest_face_landmarks, returnPoints = False)
          
    for i in range(0, len(hullIndex)):
        src_hull.append(src_face_landmarks[int(hullIndex[i])])
        dest_hull.append(dest_face_landmarks[int(hullIndex[i])])
    
    
    # Find delanauy traingulation for convex hull points
    size_dest = dest_img.shape    
    rect = (0, 0, size_dest[1], size_dest[0])
     
    dt = calculateDelaunayTriangles(rect, dest_hull)
    
    if len(dt) == 0:
        return

    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(dt)):
        t1 = []
        t2 = []
        
        #get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(src_hull[dt[i][j]])
            t2.append(dest_hull[dt[i][j]])
        
        warpTriangle(src_img, warp_img, t1, t2)

    cv2.imwrite("output/swap-warp.jpg", warp_img)
      
    # Calculate Mask
    hull8U = []
    for i in range(0, len(dest_hull)):
        hull8U.append((dest_hull[i][0], dest_hull[i][1]))
    
    mask = np.zeros(dest_img.shape, dtype = dest_img.dtype)  
    
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
    
    r = cv2.boundingRect(np.float32([dest_hull]))    
    
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
        
    # # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(warp_img), dest_img, mask, center, cv2.NORMAL_CLONE)
    cv2.imwrite("output/swap-output.jpg", output)

# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []


    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

def morph(src_img, src_face_landmarks, dest_img, dest_face_landmarks):
    alpha = 0.5

    src_img = np.float32(src_img)
    dest_img = np.float32(dest_img)

    points = []

    # Compute weighted average point coordinates
    for i in range(0, len(src_face_landmarks)):
        x = ( 1 - alpha ) * src_face_landmarks[i][0] + alpha * dest_face_landmarks[i][0]
        y = ( 1 - alpha ) * src_face_landmarks[i][1] + alpha * dest_face_landmarks[i][1]
        points.append((x,y))

    # Allocate space for final output
    img_morph = np.copy(dest_img)

    # Find delanauy traingulation 
    size_dest = dest_img.shape    
    rect = (0, 0, size_dest[1], size_dest[0])
     
    dt = calculateDelaunayTriangles(rect, dest_face_landmarks)

    for (x, y, z) in dt:
        
        x = int(x)
        y = int(y)
        z = int(z)
        
        t1 = [src_face_landmarks[x], src_face_landmarks[y], src_face_landmarks[z]]
        t2 = [dest_face_landmarks[x], dest_face_landmarks[y], dest_face_landmarks[z]]
        t = [ points[x], points[y], points[z] ]

        # Morph one triangle at a time.
        morphTriangle(src_img, dest_img, img_morph, t1, t2, t, alpha)
    
    cv2.imwrite("output/morph-transform.jpg", img_morph)

    # dest_hull = []

    dest_hull = cv2.convexHull(dest_face_landmarks, returnPoints = True)
          
    # for i in range(0, len(hullIndex)):
        # dest_hull.append(dest_face_landmarks[int(hullIndex[i])])

    # Calculate Mask
    hull8U = []
    for i in range(0, len(dest_hull)):
        hull8U.append((dest_hull[i][0], dest_hull[i][1]))

    mask = np.zeros(dest_img.shape, dtype = dest_img.dtype)  
    
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))    
    r = cv2.boundingRect(np.float32([dest_hull]))    
    
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
        
    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img_morph), dest_img, mask, center, cv2.NORMAL_CLONE)
    cv2.imwrite("output/morph-output.jpg", output)


    # landmarks -> Delaunay Triangulation
    # affinity warp triangles
    # alpha blend

if __name__ == '__main__':

    # Read images
    src_img = cv2.imread('imgs/test7.jpg')
    dest_img = cv2.imread('imgs/test6.jpg')

    # dlib.get_frontal_face_detector
    src_faces = detect_face_using_frontal_face_detector(src_img)
    dest_faces = detect_face_using_frontal_face_detector(dest_img)
    save_tagged_image(src_img, src_faces, "output/dlib-face-tagged-src.jpg")
    save_tagged_image(dest_img, dest_faces, "output/dlib-face-tagged-dest.jpg")

    # src_face_landmarks = [predict_5_face_landmarks(src_img, face) for face in src_faces]
    # save_landmarked_image(src_img, src_face_landmarks, "output/dlib_5_src.jpg")
    # src_face_landmarks = [predict_68_face_landmarks(src_img, face) for face in src_faces]
    # save_landmarked_image(src_img, src_face_landmarks, "output/dlib_68_src.jpg")
    # src_face_landmarks = [predict_68_GTX_face_landmarks(src_img, face) for face in src_faces]
    # save_landmarked_image(src_img, src_face_landmarks, "output/dlib_68_GTX_src.jpg")
    src_face_landmarks = [predict_81_face_landmarks(src_img, face) for face in src_faces]
    save_landmarked_image(src_img, src_face_landmarks, "output/dlib_81_src.jpg")

    # dest_face_landmarks = [predict_5_face_landmarks(dest_img, face) for face in dest_faces]
    # save_landmarked_image(dest_img, dest_face_landmarks, "output/dlib_5_dest.jpg")
    # dest_face_landmarks = [predict_68_face_landmarks(dest_img, face) for face in dest_faces]
    # save_landmarked_image(dest_img, dest_face_landmarks, "output/dlib_68_dest.jpg")
    # dest_face_landmarks = [predict_68_GTX_face_landmarks(dest_img, face) for face in dest_faces]
    # save_landmarked_image(dest_img, dest_face_landmarks, "output/dlib_68_GTX_dest.jpg")
    dest_face_landmarks = [predict_81_face_landmarks(dest_img, face) for face in dest_faces]
    save_landmarked_image(dest_img, dest_face_landmarks, "output/dlib_81_dest.jpg")

    swap(src_img, src_face_landmarks[0], dest_img, dest_face_landmarks[0])
    morph(src_img, src_face_landmarks[0], dest_img, dest_face_landmarks[0])


    # # mmod_human_face_detector
    # src_faces = detect_face_using_mmod_face_detector(src_img)
    # dest_faces = detect_face_using_mmod_face_detector(dest_img)
    # save_tagged_image(src_img, src_faces, "output/mmod-face-tagged-src.jpg")
    # save_tagged_image(dest_img, dest_faces, "output/mmod-face-tagged-dest.jpg")

    # src_face_landmarks = [predict_5_face_landmarks(src_img, face) for face in src_faces]
    # save_landmarked_image(src_img, src_face_landmarks, "output/mmod_5_src.jpg")
    # src_face_landmarks = [predict_68_face_landmarks(src_img, face) for face in src_faces]
    # save_landmarked_image(src_img, src_face_landmarks, "output/mmod_68_src.jpg")
    # src_face_landmarks = [predict_68_GTX_face_landmarks(src_img, face) for face in src_faces]
    # save_landmarked_image(src_img, src_face_landmarks, "output/mmod_68_GTX_src.jpg")
    # src_face_landmarks = [predict_81_face_landmarks(src_img, face) for face in src_faces]
    # save_landmarked_image(src_img, src_face_landmarks, "output/mmod_81_src.jpg")

    # dest_face_landmarks = [predict_5_face_landmarks(dest_img, face) for face in dest_faces]
    # save_landmarked_image(dest_img, dest_face_landmarks, "output/mmod_5_dest.jpg")
    # dest_face_landmarks = [predict_68_face_landmarks(dest_img, face) for face in dest_faces]
    # save_landmarked_image(dest_img, dest_face_landmarks, "output/mmod_68_dest.jpg")
    # dest_face_landmarks = [predict_68_GTX_face_landmarks(dest_img, face) for face in dest_faces]
    # save_landmarked_image(dest_img, dest_face_landmarks, "output/mmod_68_GTX_dest.jpg")
    # dest_face_landmarks = [predict_81_face_landmarks(dest_img, face) for face in dest_faces]
    # save_landmarked_image(dest_img, dest_face_landmarks, "output/mmod_81_dest.jpg")

    # # Select src face
    # src_points, src_shape, src_face = select_face(src_img)
    # # Select dst face
    # dst_faceBoxes = select_all_faces(dst_img)

    # if dst_faceBoxes is None:
    #     print('Detect 0 Face !!!')
    #     exit(-1)

    # output = dst_img
    # for k, dst_face in dst_faceBoxes.items():
    #     output = face_swap(src_face, dst_face["face"], src_points,
    #                        dst_face["points"], dst_face["shape"],
    #                        output, args)

    # dir_path = os.path.dirname(args.out)
    # if not os.path.isdir(dir_path):
    #     os.makedirs(dir_path)

    # cv2.imwrite("output/81_6_7_1.jpg", output)

    # ##For debug
    # if not args.no_debug_window:
    #     cv2.imshow("From", dst_img)
    #     cv2.imshow("To", output)
    #     cv2.waitKey(0)
        
    #     cv2.destroyAllWindows()
