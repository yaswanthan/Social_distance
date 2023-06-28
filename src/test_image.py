import sys

import onnxruntime as rt
import time
from config import *
from postprocessing import *
from preprocessing import *
print('Device:',rt.get_device())
print('All Available Device:',rt.get_available_providers())
if 'CUDAExecutionProvider' in rt.get_available_providers():
    sess = rt.InferenceSession(ONNX_PATH,providers=['CUDAExecutionProvider'])
else:
    sess = rt.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

    # Test if it has reached the end of the video
    input_size = 416
    original_image = cv2.imread(IMG_PATH)
    original_image_size = original_image.shape[:2]

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
    bboxes = nms(bboxes, 0.213, method='nms')
    print('No.of.BBOX:',len(bboxes))
    #  bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    print(bboxes)
    b=np.array(bboxes)


    image = draw_bbox(original_image, bboxes)
    #image=cv2.resize(image,(500,500))
    print(sys.version)
    cv2.imshow('img', image)
    cv2.waitKey(0)


print("[INFO] cleaning up...")

