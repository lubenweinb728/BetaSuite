import cv2
import math
import numpy as np
import onnxruntime

import betaconfig
import betaconst

def get_session():
    providers = [
        ('CUDAExecutionProvider', {'device_id': betaconfig.cuda_device_id})
    ] if betaconfig.gpu_enabled else [
        ('CPUExecutionProvider', {})
    ]
    session = onnxruntime.InferenceSession('../model/640m.onnx', providers=providers)
    return session


def prep_img_for_nn(raw_img, size, _scale_not_used=None):
    h, w = raw_img.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(raw_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    img = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_w = (size - new_w) // 2
    pad_h = (size - new_h) // 2
    img[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img, scale, pad_w, pad_h


def scale_coords_back(boxes, scale, pad_w, pad_h):
    boxes[:, 0] = (boxes[:, 0] - pad_w) / scale
    boxes[:, 1] = (boxes[:, 1] - pad_h) / scale
    boxes[:, 2] = (boxes[:, 2] - pad_w) / scale
    boxes[:, 3] = (boxes[:, 3] - pad_h) / scale
    return boxes


def get_raw_model_output(img_array, session):
    outputs = []
    for img in img_array:
        input_tensor = np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
        try:
            output = session.run(None, {betaconst.model_input: input_tensor})
            if not output or len(output[0]) == 0 or output[0][0].shape[0] == 0:
                outputs.append(np.zeros((0, 6), dtype=np.float32))
            else:
                outputs.append(output[0][0])
        except Exception as e:
            print("ONNX Runtime Error:", e)
            outputs.append(np.zeros((0, 6), dtype=np.float32))
    return outputs


def raw_boxes_from_model_output(model_output, scale_info_array, t):
    all_raw_boxes = []
    for detections, (scale, pad_w, pad_h) in zip(model_output, scale_info_array):
        if detections.shape[0] > 0:
            detections = scale_coords_back(detections.copy(), scale, pad_w, pad_h)

        raw_boxes = []
        for det in detections:
            x1, y1, x2, y2, score, class_id = det.tolist()

            if score < betaconst.global_min_prob:
                continue

            class_id_int = int(class_id)
            label = betaconst.class_name_from_id(class_id_int).lower()

            if label not in [item.lower() for item in betaconfig.items_to_censor]:
                continue

            raw_boxes.append({
                'x': float(math.floor(x1)),
                'y': float(math.floor(y1)),
                'w': float(math.ceil(x2 - x1)),
                'h': float(math.ceil(y2 - y1)),
                'class_id': float(class_id),
                'score': float(score),
                't': t,
                'label': label  # 添加 label 字段，兼容 process_raw_box 使用
            })

        all_raw_boxes.append(raw_boxes)
    return all_raw_boxes


def detect_raw_boxes(img_array, session, scale_info_array, t):
    model_output = get_raw_model_output(img_array, session)
    return raw_boxes_from_model_output(model_output, scale_info_array, t)


def raw_boxes_for_img(img, size, session, t):
    adj_img, scale, pad_w, pad_h = prep_img_for_nn(img, size)
    raw_boxes = detect_raw_boxes([adj_img], session, [(scale, pad_w, pad_h)], t)
    return raw_boxes[0] if raw_boxes else []
