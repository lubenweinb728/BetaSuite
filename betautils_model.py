import cv2
import math
import numpy as np
import onnxruntime
import os

import betaconfig
import betaconst
import betautils_model as bu_model

_eye_boxes_cache = []

def set_eye_boxes(eye_boxes):
    global _eye_boxes_cache
    _eye_boxes_cache = eye_boxes

def get_eye_boxes():
    return _eye_boxes_cache

# --- 配置模型路径 ---
DETECTOR_MODEL_PATH = '../model/detector_v2_default_checkpoint.onnx'
PFLD_MODEL_PATH = '../model/PFLD_112_1_opt_sim.onnx'

# ====================
# Detector 部分
# ====================

def get_session():
    if betaconfig.gpu_enabled:
        print(f"使用 GPU 推理")
        providers = [('CUDAExecutionProvider', {'device_id': betaconfig.cuda_device_id})]
    else:
        print(f"使用 CPU 推理")
        providers = [('CPUExecutionProvider', {})]
    return onnxruntime.InferenceSession(DETECTOR_MODEL_PATH, providers=providers)

def get_resize_scale(img_h, img_w, max_length):
    return 1 if max_length == 0 else max_length / max(img_h, img_w)

def get_image_resize_scale(raw_img, max_length):
    (s1, s2, _) = raw_img.shape
    return get_resize_scale(s1, s2, max_length)

def prep_img_for_nn(raw_img, size, scale):
    adj_img = cv2.resize(raw_img, None, fx=scale, fy=scale)
    if size > 0:
        (h, w, _) = adj_img.shape
        adj_img = cv2.copyMakeBorder(adj_img, 0, size - h, 0, size - w, cv2.BORDER_CONSTANT, value=0)
    adj_img = adj_img.astype(np.float32)
    adj_img -= [103.939, 116.779, 123.68]
    return adj_img

def get_raw_model_output(img_array, session):
    output = [
        np.zeros((len(img_array), 300, 4), dtype=np.float32),
        np.zeros((len(img_array), 300), dtype=np.float32),
        np.zeros((len(img_array), 300), dtype=np.int32),
    ]
    for i, img in enumerate(img_array):
        print(f"开始主模型推理")
        output[0][i], output[1][i], output[2][i] = session.run(betaconst.model_outputs, {betaconst.model_input: [img_array[i]]})
    return output

def raw_boxes_from_model_output(model_output, scale_array, t):
    all_raw_boxes = []
    all_boxes, all_scores, all_classes = model_output
    for boxes, scores, classes, scale in zip(all_boxes, all_scores, all_classes, scale_array):
        raw_boxes = []
        for box, score, class_id in zip(boxes, scores, classes):
            if score > betaconst.global_min_prob:
                raw_boxes.append({
                    'x': float(math.floor(box[0] / scale)),
                    'y': float(math.floor(box[1] / scale)),
                    'w': float(math.ceil((box[2] - box[0]) / scale)),
                    'h': float(math.ceil((box[3] - box[1]) / scale)),
                    'class_id': float(class_id),
                    'score': float(score),
                    't': t,
                })
        all_raw_boxes.append(raw_boxes)
    return all_raw_boxes

def detect_raw_boxes(img_array, session, scale_array, t):
    model_output = get_raw_model_output(img_array, session)
    return raw_boxes_from_model_output(model_output, scale_array, t)

# ====================
# EyeOccluder 模块
# ====================

class EyeOccluder:
    def __init__(self):
        providers = [('CUDAExecutionProvider', {'device_id': betaconfig.cuda_device_id})] if betaconfig.gpu_enabled else [('CPUExecutionProvider', {})]
        if betaconfig.eyes_bar[0] == 'on':
            if not os.path.exists(PFLD_MODEL_PATH):
                raise FileNotFoundError(f"[ERROR] 模型文件不存在: {PFLD_MODEL_PATH}")
            self.session = onnxruntime.InferenceSession(PFLD_MODEL_PATH, providers=providers)
            self.input_name = self.session.get_inputs()[0].name

    def resize_with_padding(self, face_img, target_size=112):
        h, w = face_img.shape[:2]
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(face_img, (new_w, new_h))
        padded_img = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2
        padded_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        return padded_img, scale, pad_x, pad_y

    def predict_eyes(self, face_img):
        padded_img, scale, pad_x, pad_y = self.resize_with_padding(face_img)
        input_blob = padded_img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0
        print(f"开始 PFLD 推理")
        output = self.session.run(None, {self.input_name: input_blob})[0]
        if output.shape[-1] != 196:
            raise ValueError(f"[ERROR] 不支持的关键点输出 shape: {output.shape}")
        landmarks = output[0].reshape(98, 2)
        return landmarks, scale, pad_x, pad_y

    def extract_eyes_from_faces(self, image, face_boxes):
        h_img, w_img = image.shape[:2]
        eye_list = []
        for idx, box in enumerate(face_boxes):
            x, y, w, h = int(box['x']), int(box['y']), int(box['w']), int(box['h'])
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(x + w, w_img), min(y + h, h_img)
            if x2 <= x1 or y2 <= y1:
                continue
            face_crop = image[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            try:
                landmarks, scale, pad_x, pad_y = self.predict_eyes(face_crop)
                lx = int((landmarks[60][0] * 112 - pad_x) / scale) + x1
                ly = int((landmarks[60][1] * 112 - pad_y) / scale) + y1
                rx = int((landmarks[72][0] * 112 - pad_x) / scale) + x1
                ry = int((landmarks[72][1] * 112 - pad_y) / scale) + y1
                eye_list.append(((lx, ly), (rx, ry)))
                print(f"眼睛坐标 @box[{idx}]: L=({lx},{ly}) R=({rx},{ry})")
            except Exception as e:
                print(f"[ERROR] 眼睛推理失败 @box[{idx}]: {e}")
        return eye_list

# ========== 初始化一次 ==========
eye_occluder = EyeOccluder()

# ========== 原接口：主程序会调用这个 ==========
def raw_boxes_for_img(img, size, session, t):
    global _eye_boxes_cache

    scale = get_image_resize_scale(img, size)
    adj_img = prep_img_for_nn(img, size, scale)
    raw_boxes = detect_raw_boxes(np.expand_dims(adj_img, axis=0), session, [scale], t)[0]

    # 提取人脸框 → 眼睛位置
    if betaconfig.eyes_bar[0] == 'on':
        min_score = ( betaconfig.item_overrides.get('face_femme', {} ) ).get( 'min_prob',betaconfig.default_min_prob)
        print(f"人脸推理最小置信度: {min_score}")
        face_boxes = [box for box in raw_boxes if int(box.get("class_id", -1)) == 6 and box['score'] >= min_score]
        _eye_boxes_cache = eye_occluder.extract_eyes_from_faces(img, face_boxes)
        bu_model.set_eye_boxes(_eye_boxes_cache)
    else:
        print(f"眼睛推理关闭")
    return raw_boxes
