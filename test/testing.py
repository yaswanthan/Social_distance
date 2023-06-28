import onnxruntime as rt
import time
from config import *
from postprocessing import *
from preprocessing import *
import imutils

print('Device:', rt.get_device())
print('All Available Device:', rt.get_available_providers())
if 'CUDAExecutionProvider' in rt.get_available_providers():
    sess = rt.InferenceSession(ONNX_PATH, providers=['CUDAExecutionProvider'])
else:
    sess = rt.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

capture = cv2.VideoCapture(VIDEO_PATH)

while True:
    start_time = time.time()
    ret, frame = capture.read()
    # Test if it has reached the end of the video
    input_size = 416
    original_image = imutils.resize(frame,width=1000)
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
    bboxes = nms(bboxes, 0.213, method='nms') #0.213
    print('No.of.Detections:',len(bboxes))
    print('----bboxes----')
    for i in bboxes:
        print(i)
    image = draw_bbox(original_image, bboxes)
    if cv2.waitKey(30) & 0xff == ord('q'):
        break
    cv2.imshow('img', image)
    print("FPS: ", 1.0 / (time.time() - start_time))

print("[INFO] cleaning up...")
capture.release()
cv2.destroyAllWindows()
