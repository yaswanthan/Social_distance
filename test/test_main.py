import itertools
import math

import numpy as np
import cv2
import yaml
import onnxruntime as rt
import time
from src.config import *
from src.postprocessing import *
from src.preprocessing import *
import imutils

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
BIG_CIRCLE = 60
SMALL_CIRCLE = 3


def get_human_box_detection(bbox):
    """
    For each object detected, check if it is a human and if the confidence >> our threshold.
    Return 2 coordonates necessary to build the box.
    @ boxes : all our boxes coordinates
    @ scores : confidence score on how good the prediction is -> between 0 & 1
    @ classes : the class of the detected object ( 1 for human )
    @ height : of the image -> to get the real pixel value
    @ width : of the image -> to get the real pixel value
    """
    array_boxes = []  # Create an empty list
    for i, bbox in enumerate(bbox):
        # If the class of the detected object is 1 and the confidence of the prediction is > 0.6
        if bbox[5] == 0:
            box = np.array(bbox[:4], dtype=np.int32)
            array_boxes.append[box[0],box[1]]
    return array_boxes


def get_centroids_and_groundpoints(array_boxes_detected):
    """
    For every bounding box, compute the centroid and the point located on the bottom center of the box
    @ array_boxes_detected : list containing all our bounding boxes
    """
    array_centroids, array_groundpoints = [], []  # Initialize empty centroid and ground point lists
    for index, box in enumerate(array_boxes_detected):
        # Draw the bounding box
        # c
        # Get the both important points
        centroid, ground_point = get_points_from_box(box)
        array_centroids.append(centroid)
        array_groundpoints.append(centroid)
    return array_centroids, array_groundpoints


def get_points_from_box(box):
    """
    Get the center of the bounding and the point "on the ground"
    @ param = box : 2 points representing the bounding box
    @ return = centroid (x1,y1) and ground point (x2,y2)
    """
    # Center of the box x = (x1+x2)/2 et y = (y1+y2)/2
    center_x = int(((box[1] + box[3]) / 2))
    center_y = int(((box[0] + box[2]) / 2))
    # Coordiniate on the point at the bottom center of the box
    center_y_ground = center_y + ((box[2] - box[0]) / 2)
    return (center_x, center_y), (center_x, int(center_y_ground))


def change_color_on_topview(pair):
    """
    Draw red circles for the designated pair of points
    """
    cv2.circle(bird_view_img, (pair[0][0], pair[0][1]), BIG_CIRCLE, COLOR_RED, 2)
    cv2.circle(bird_view_img, (pair[0][0], pair[0][1]), SMALL_CIRCLE, COLOR_RED, -1)
    cv2.circle(bird_view_img, (pair[1][0], pair[1][1]), BIG_CIRCLE, COLOR_RED, 2)
    cv2.circle(bird_view_img, (pair[1][0], pair[1][1]), SMALL_CIRCLE, COLOR_RED, -1)


def draw_rectangle(corner_points):
    # Draw rectangle box over the delimitation area
    cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[1][0], corner_points[1][1]), COLOR_BLUE,
             thickness=1)
    cv2.line(frame, (corner_points[1][0], corner_points[1][1]), (corner_points[3][0], corner_points[3][1]), COLOR_BLUE,
             thickness=1)
    cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE,
             thickness=1)
    cv2.line(frame, (corner_points[3][0], corner_points[3][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE,
             thickness=1)


def compute_perspective_transform(corner_points, width, height, image):
    """ Compute the transformation matrix
    @ corner_points : 4 corner points selected from the image
    @ height, width : size of the image
    """
    # Create an array out of the 4 corner points
    corner_points_array = np.float32(corner_points)
    # Create an array with the parameters (the dimensions) required to build the matrix
    img_params = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Compute and return the transformation matrix
    matrix = cv2.getPerspectiveTransform(corner_points_array, img_params)
    img_transformed = cv2.warpPerspective(image, matrix, (width, height))
    return matrix, img_transformed


def compute_point_perspective_transformation(matrix, list_downoids):
    """ Apply the perspective transformation to every ground point which have been detected on the main frame.
    @ matrix : the 3x3 matrix
    @ list_downoids : list that contains the points to transform
    return : list containing all the new points
    """
    # Compute the new coordinates of our points
    list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
    # Loop over the points and add them to the list that will be returned
    transformed_points_list = list()
    for i in range(0, transformed_points.shape[0]):
        transformed_points_list.append([transformed_points[i][0][0], transformed_points[i][0][1]])
    return transformed_points_list


#########################################
# Load the config for the top-down view #
#########################################
print("[ Loading config file for the bird view transformation ] ")
with open("/home/sri/education/social_distancing/data/config_birdview.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)
width_og, height_og = 0, 0
corner_points = []
for section in cfg:
    corner_points.append(cfg["image_parameters"]["p1"])
    corner_points.append(cfg["image_parameters"]["p2"])
    corner_points.append(cfg["image_parameters"]["p3"])
    corner_points.append(cfg["image_parameters"]["p4"])
    width_og = int(cfg["image_parameters"]["width_og"])
    height_og = int(cfg["image_parameters"]["height_og"])
    img_path = cfg["image_parameters"]["img_path"]
    size_frame = cfg["image_parameters"]["size_frame"]
print(" Done : [ Config file loaded ] ...")

#########################################
#		    Minimal distance			#
#########################################
distance_minimum = input("Prompt the size of the minimal distance between 2 pedestrians : ")
if distance_minimum == "":
    distance_minimum = "110"

#########################################
#     Compute transformation matrix		#
#########################################
# Compute  transformation matrix from the original frame
matrix, imgOutput = compute_perspective_transform(corner_points, width_og, height_og, cv2.imread(img_path))
height, width, _ = imgOutput.shape
blank_image = np.zeros((height, width, 3), np.uint8)
height = blank_image.shape[0]
width = blank_image.shape[1]
dim = (width, height)

print('Device:', rt.get_device())
print('All Available Device:', rt.get_available_providers())
if 'CUDAExecutionProvider' in rt.get_available_providers():
    sess = rt.InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider'])
else:
    sess = rt.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

capture = cv2.VideoCapture(VIDEO_PATH)
while True:
    start_time = time.time()
    # Load the image of the ground and resize it to the correct size
    img = cv2.imread("/home/sri/education/social_distancing/test/static_frame_from_video.jpg")
    bird_view_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    ret, frame = capture.read()
    # Test if it has reached the end of the video
    input_size = 416
    original_image = imutils.resize(frame, width=1000)
    original_image_size = original_image.shape[:2]
    print('Original Image Shape:', original_image.shape)
    image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    print("Preprocessed image shape:", image_data.shape)  # shape of the preprocessed input

    outputs = sess.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))
    input_name = sess.get_inputs()[0].name

    detections = sess.run(output_names, {input_name: image_data})
    print("Output shape:", list(map(lambda detection: detection.shape, detections)))

    ANCHORS = ANCHOR_PATH
    STRIDES = [8, 16, 32]
    XYSCALE = [1.2, 1.1, 1.05]

    ANCHORS = get_anchors(ANCHORS)
    STRIDES = np.array(STRIDES)

    pred_bbox = postprocess_bbbox(detections, ANCHORS, STRIDES, XYSCALE)
    bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
    bboxes = nms(bboxes, 0.213, method='nms')  # 0.213
    print('No.of.Detections with all classes:', len(bboxes))
    # print('----bboxes----')
    # for i in bboxes:
    #     print(i)
    # image = draw_bbox(original_image, bboxes)
    array_boxes_detected = get_human_box_detection(bboxes)
    print('No.of.Detections with all classes:', len(array_boxes_detected))

    # Both of our lists that will contain the centroïds coordonates and the ground points
    array_centroids, array_groundpoints = get_centroids_and_groundpoints(array_boxes_detected)
    transformed_downoids = compute_point_perspective_transformation(matrix, array_groundpoints)
    # Show every point on the top view image
    for point in transformed_downoids:
        x, y = point
        cv2.circle(bird_view_img, (x, y), BIG_CIRCLE, COLOR_GREEN, 2)
        cv2.circle(bird_view_img, (x, y), SMALL_CIRCLE, COLOR_GREEN, -1)

        # Check if 2 or more people have been detected (otherwise no need to detect)
        if len(transformed_downoids) >= 2:
            for index, downoid in enumerate(transformed_downoids):
                if not (downoid[0] > width or downoid[0] < 0 or downoid[1] > height + 200 or downoid[1] < 0):
                    cv2.rectangle(frame, (array_boxes_detected[index][1], array_boxes_detected[index][0]),
                                  (array_boxes_detected[index][3], array_boxes_detected[index][2]), COLOR_GREEN, 2)

            # Iterate over every possible 2 by 2 between the points combinations
            list_indexes = list(itertools.combinations(range(len(transformed_downoids)), 2))
            for i, pair in enumerate(itertools.combinations(transformed_downoids, r=2)):
                # Check if the distance between each combination of points is less than the minimum distance chosen
                if math.sqrt((pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2) < int(distance_minimum):
                    # Change the colors of the points that are too close from each other to red
                    if not (pair[0][0] > width or pair[0][0] < 0 or pair[0][1] > height + 200 or pair[0][1] < 0 or
                            pair[1][0] > width or pair[1][0] < 0 or pair[1][1] > height + 200 or pair[1][1] < 0):
                        change_color_on_topview(pair)
                        # Get the equivalent indexes of these points in the original frame and change the color to red
                        index_pt1 = list_indexes[i][0]
                        index_pt2 = list_indexes[i][1]
                        cv2.rectangle(frame, (array_boxes_detected[index_pt1][1], array_boxes_detected[index_pt1][0]),
                                      (array_boxes_detected[index_pt1][3], array_boxes_detected[index_pt1][2]),
                                      COLOR_RED, 2)
                        cv2.rectangle(frame, (array_boxes_detected[index_pt2][1], array_boxes_detected[index_pt2][0]),
                                      (array_boxes_detected[index_pt2][3], array_boxes_detected[index_pt2][2]),
                                      COLOR_RED, 2)
    draw_rectangle(corner_points)
    #cv2.imshow("Bird view", bird_view_img)
    cv2.imshow("Original picture", frame)
    if cv2.waitKey(30) & 0xff == ord('q'):
        break
    # cv2.imshow('img', image)
    print("FPS: ", 1.0 / (time.time() - start_time))
