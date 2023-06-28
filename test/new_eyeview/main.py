# imports
import imutils
import onnxruntime as rt

# own modules
import plot
import utills
from src.postprocessing import *
from src.preprocessing import *

mouse_pts = []


def get_human_box_detection(bbox):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    array_boxes = []  # Create an empty list
    for i, bbox in enumerate(bbox):
        # If the class of the detected object is 1 and the confidence of the prediction is > 0.6
        if bbox[5] == 0:
            box = np.array(bbox[:4], dtype=np.int32)
            width = int(box[2] - box[0])
            height = int(box[3] - box[1])
            array_boxes.append([box[0], box[1], width, height])
    return array_boxes


def get_mouse_points(event, x, y, flags, param):
    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)

        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts) - 1][0], mouse_pts[len(mouse_pts) - 1][1]), (70, 70, 70),
                     2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)

        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))


def calculate_social_distancing():
    count = 0
    print('Device:', rt.get_device())
    print('All Available Device:', rt.get_available_providers())
    if 'CUDAExecutionProvider' in rt.get_available_providers():
        sess = rt.InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider'])
    else:
        sess = rt.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

    capture = cv2.VideoCapture(VIDEO_PATH)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(capture.get(cv2.CAP_PROP_FPS))

    # Set scale for birds eye view
    # Bird's eye view will only show ROI
    scale_w, scale_h = utills.get_scale(width, height)
    points = []
    global image
    while True:
        ret, frame = capture.read()
        input_size = 416
        original_image = imutils.resize(frame, width=500)
        frame = original_image
        # original_image=frame
        (H, W) = original_image.shape[:2]

        # first frame will be used to draw ROI and horizontal and vertical 180 cm distance(unit length in both directions)
        # if count == 0:
        #     while True:
        #         image = original_image
        #         cv2.imshow("image", image)
        #         cv2.waitKey(1)
        #         if len(mouse_pts) == 8:
        #             cv2.destroyWindow("image")
        #             break

        points = POINTS

        # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are
        # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view.
        # This bird eye view then has the property property that points are distributed uniformally horizontally and
        # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are
        # equally distributed, which was not case for normal view.
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        prespective_transform = cv2.getPerspectiveTransform(src, dst)

        # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]

        # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
        # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
        # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
        # which we can use to calculate distance between two humans in transformed view or bird eye view
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)

        ####################################################################################
        ##model
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
        boxes1 = get_human_box_detection(bboxes)
        # postprocesing
        # Here we will be using bottom center point of bounding box for all boxes and will transform all those
        # bottom center points to bird eye view
        if len(boxes1) == 0:
            count = count + 1
            continue
        person_points = utills.get_transformed_points(boxes1, prespective_transform)

        # Here we will calculate distance between transformed points(humans)
        distances_mat, bxs_mat = utills.get_distances(boxes1, person_points, distance_w, distance_h)
        risk_count = utills.get_count(distances_mat)

        frame1 = np.copy(frame)

        # Draw bird eye view and frame with bouding boxes around humans according to risk factor
        bird_image = plot.bird_eye_view(frame, distances_mat, person_points, scale_w, scale_h, risk_count)
        img = plot.social_distancing_view(frame1, bxs_mat, boxes1, risk_count)
        print('Bird Image:', bird_image.shape)
        print('Frame Image:', img.shape)
        if cv2.waitKey(30) & 0xff == ord('q'):
            break
        cv2.imshow('bird', img)
    capture.release()
    cv2.destroyAllWindows()


# set mouse callback

#cv2.namedWindow("image")
#cv2.setMouseCallback("image", get_mouse_points)
np.random.seed(42)
count = 0
count = 0
print('Device:', rt.get_device())
print('All Available Device:', rt.get_available_providers())
if 'CUDAExecutionProvider' in rt.get_available_providers():
    sess = rt.InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider'])
else:
    sess = rt.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

capture = cv2.VideoCapture(VIDEO_PATH)
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(capture.get(cv2.CAP_PROP_FPS))

# Set scale for birds eye view
# Bird's eye view will only show ROI
scale_w, scale_h = utills.get_scale(width, height)
points = []
global image
while True:
    ret, frame = capture.read()
    input_size = 416
    original_image = imutils.resize(frame, width=500)
    frame = original_image
    # original_image=frame
    (H, W) = original_image.shape[:2]

    # first frame will be used to draw ROI and horizontal and vertical 180 cm distance(unit length in both directions)
    # if count == 0:
    #     while True:
    #         image = original_image
    #         cv2.imshow("image", image)
    #         cv2.waitKey(1)
    #         if len(mouse_pts) == 8:
    #             cv2.destroyWindow("image")
    #             break

    points = POINTS

    # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are
    # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view.
    # This bird eye view then has the property property that points are distributed uniformally horizontally and
    # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are
    # equally distributed, which was not case for normal view.
    src = np.float32(np.array(points[:4]))
    dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
    prespective_transform = cv2.getPerspectiveTransform(src, dst)

    # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
    pts = np.float32(np.array([points[4:7]]))
    warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]

    # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
    # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
    # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
    # which we can use to calculate distance between two humans in transformed view or bird eye view
    distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
    distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
    pnts = np.array(points[:4], np.int32)
    cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)

    ####################################################################################
    ##model
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
    boxes1 = get_human_box_detection(bboxes)
    # postprocesing
    # Here we will be using bottom center point of bounding box for all boxes and will transform all those
    # bottom center points to bird eye view
    if len(boxes1) == 0:
        count = count + 1
        continue
    person_points = utills.get_transformed_points(boxes1, prespective_transform)

    # Here we will calculate distance between transformed points(humans)
    distances_mat, bxs_mat = utills.get_distances(boxes1, person_points, distance_w, distance_h)
    risk_count = utills.get_count(distances_mat)

    frame1 = np.copy(frame)

    # Draw bird eye view and frame with bouding boxes around humans according to risk factor
    bird_image = plot.bird_eye_view(frame, distances_mat, person_points, scale_w, scale_h, risk_count)
    img = plot.social_distancing_view(frame1, bxs_mat, boxes1, risk_count)
    print('Bird Image:', bird_image.shape)
    print('Frame Image:', img.shape)
    if cv2.waitKey(30) & 0xff == ord('q'):
        break
    cv2.imshow('bird', img)

capture.release()
cv2.destroyAllWindows()
