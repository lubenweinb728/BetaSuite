import cv2
import math
import numpy as np
import logging
from functools import lru_cache

import betaconfig
import betaconst
import betautils_hash as bu_hash

DETECTOR_MODEL_PATH = betaconst.PROJECT_PATH / 'model' / '320n.onnx'
PFLD_MODEL_PATH = betaconst.PROJECT_PATH / 'model' / 'PFLD_112_1_opt_sim.onnx'
LEGACY_MODEL_NAME = 'detector_v2_default_checkpoint.onnx'
LEGACY_SIZE = 1280
LEGACY_MIN_PROB = 0.30
# Original ONNX output order, translated to the current class table.
LEGACY_CLASSES = (
    'ANUS_EXPOSED', 'ARMPITS_EXPOSED', 'BELLY_COVERED', 'BELLY_EXPOSED',
    'BUTTOCKS_COVERED', 'BUTTOCKS_EXPOSED', 'FACE_FEMALE', 'FACE_MALE',
    'FEET_COVERED', 'FEET_EXPOSED', 'FEMALE_BREAST_COVERED', 'FEMALE_BREAST_EXPOSED',
    'FEMALE_GENITALIA_COVERED', 'FEMALE_GENITALIA_EXPOSED',
    'MALE_BREAST_EXPOSED', 'MALE_GENITALIA_EXPOSED',
)
LEGACY_OUTPUTS = [
    'filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0',
    'filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0',
    'filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0',
]
_fallback_backend = None
_eyes_failed = set()


@lru_cache(maxsize=1)
def get_core():
    import openvino as ov

    core = ov.Core()
    core.set_property({'CACHE_DIR': str(betaconst.PROJECT_PATH / 'model' / 'openvino_cache')})
    return core


def compile_model(path, input_shape, output_shape):
    core = get_core()
    graph = core.read_model(path)
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise ValueError(f'Expected one input and one output in {path.name}')
    if graph.input().partial_shape.is_dynamic or tuple(graph.input().shape) != input_shape:
        graph.reshape(input_shape)
    if tuple(graph.output().shape) != output_shape:
        raise ValueError(f'Unexpected output shape in {path.name}: {graph.output().shape}')
    session = core.compile_model(graph, 'CPU', {'PERFORMANCE_HINT': 'LATENCY'})
    logging.info('%s compiled on %s', path.name, session.get_property('EXECUTION_DEVICES'))
    return session


class DetectorBackendError(RuntimeError):
    pass


def detector_backend():
    return _fallback_backend or ('openvino' if betaconfig.gpu_enabled else 'onnx')


def legacy_model_path():
    model_dir = DETECTOR_MODEL_PATH.parent
    local_path = model_dir / LEGACY_MODEL_NAME
    if local_path.is_file():
        return local_path
    original_path = model_dir.parent.parent / 'betasuite' / 'model' / LEGACY_MODEL_NAME
    if original_path.is_file():
        return original_path
    raise FileNotFoundError(f'Original ONNX model required for CPU fallback: {local_path}')


def load_onnx_session(path):
    try:
        import onnxruntime
    except (ImportError, OSError) as err:
        raise RuntimeError('CPU fallback requires the onnxruntime package') from err
    session = onnxruntime.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    logging.info('%s loaded with ONNX Runtime CPU', path.name)
    return session


@lru_cache(maxsize=1)
def get_legacy_session():
    return load_onnx_session(legacy_model_path())


@lru_cache(maxsize=2)
def compile_onnx_model(path, input_shape, output_shape):
    session = load_onnx_session(path)
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f'Expected one input and one output in {path.name}')
    for port, expected in ((inputs[0], input_shape), (outputs[0], output_shape)):
        if (port.type != 'tensor(float)' or len(port.shape) != len(expected) or
                any(isinstance(dim, int) and dim != size for dim, size in zip(port.shape, expected))):
            raise ValueError(f'Unexpected tensor {port.name} in {path.name}: {port.type} {port.shape}')
    input_name = inputs[0].name
    output_name = outputs[0].name

    def infer(tensor):
        return session.run([output_name], {input_name: tensor})

    return infer


def detector_shapes():
    size = betaconst.DETECTOR_SIZE
    return (1, 3, size, size), (1, 4 + len(betaconst.classes), 2100)


@lru_cache(maxsize=1)
def get_openvino_session():
    return compile_model(DETECTOR_MODEL_PATH, *detector_shapes())


def get_session():
    backend = detector_backend()
    if backend == 'legacy':
        return get_legacy_session()
    try:
        if backend == 'openvino':
            return get_openvino_session()
        return compile_onnx_model(DETECTOR_MODEL_PATH, *detector_shapes())
    except Exception as err:
        raise DetectorBackendError(f'Could not initialize {backend} detector: {err}') from err


def run_with_detector(process, sample_rate=None):
    global _fallback_backend
    while detector_backend() != 'legacy':
        backend = detector_backend()
        try:
            if backend == 'openvino':
                get_core()
            key = bu_hash.detection_hash(DETECTOR_MODEL_PATH, sample_rate, backend=backend)
        except Exception as err:
            failure = err
        else:
            try:
                return process(key, eyes_model_key())
            except DetectorBackendError as err:
                failure = err
        if backend == 'openvino':
            _fallback_backend = 'onnx'
            next_name = 'updated ONNX Runtime CPU detector'
        else:
            _fallback_backend = 'legacy'
            next_name = 'original ONNX Runtime CPU detector'
        logging.warning('%s unavailable: %s. Switching to %s.', backend, failure, next_name)

    key = bu_hash.detection_hash(legacy_model_path(), sample_rate,
                                (LEGACY_SIZE, LEGACY_MIN_PROB, LEGACY_CLASSES))
    return process(key, None)


@lru_cache(maxsize=1)
def get_openvino_eyes_session():
    return compile_model(PFLD_MODEL_PATH, (1, 3, 112, 112), (1, 196))


def get_eyes_session():
    if detector_backend() == 'onnx':
        return compile_onnx_model(PFLD_MODEL_PATH, (1, 3, 112, 112), (1, 196))
    return get_openvino_eyes_session()


def eyes_enabled():
    return betaconfig.eyes_bar[0] == 'on' or bool(betaconfig.debug_mode & 1)


def eyes_model_key():
    if not eyes_enabled():
        return None
    try:
        return bu_hash.dictionary_hash({'model': bu_hash.model_hash(PFLD_MODEL_PATH),
                                        'backend': detector_backend(), 'eyes_version': 1}, 20)
    except OSError as err:
        logging.warning('Eye model unavailable: %s', err)
        return 'unavailable'


def prep_img_for_nn( raw_img ):
    target_size = betaconst.DETECTOR_SIZE
    img_h, img_w, _ = raw_img.shape

    scale = min(target_size / img_h, target_size / img_w)
    adj_img = cv2.resize(raw_img, (max(1, round(img_w * scale)),
                                  max(1, round(img_h * scale))), interpolation=cv2.INTER_LINEAR)

    new_h, new_w = adj_img.shape[:2]
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    bottom = target_size - new_h - top
    right = target_size - new_w - left
    adj_img = cv2.copyMakeBorder(adj_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114,114,114))
    
    tensor = np.empty((3, target_size, target_size), dtype=np.float32)
    np.divide(adj_img.transpose(2, 0, 1)[::-1], 255, out=tensor, dtype=np.float32)
    return tensor, (scale, top, left)


# 面部图片前处理 - 保持不变
def resize_face_img( face_img ):
    target_size = 112
    h, w = face_img.shape[:2]
    
    scale = min(target_size / w, target_size / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    tensor = np.full((1, 3, target_size, target_size), np.float32(128) / np.float32(255),
                     dtype=np.float32)
    roi = tensor[0, :, pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    np.divide(resized.transpose(2, 0, 1), 255, out=roi, dtype=np.float32)
    return tensor, scale, pad_x, pad_y


def get_model_output(tensor, session, scale_params, t):
    output = session(tensor[None])[0]
    if output.shape != detector_shapes()[1]:
        raise ValueError(f'Unexpected detector output shape: {output.shape}')
    pred = output[0]
    scores = pred[4:].max(axis=0)
    score_thresh = betaconst.global_min_prob
    selected = np.flatnonzero((scores >= score_thresh) & (pred[2] > 0) & (pred[3] > 0))
    candidates = pred[:, selected].T
    finite = np.isfinite(candidates).all(axis=1)
    candidates = candidates[finite]
    if not len(candidates):
        return []

    class_ids = candidates[:, 4:].argmax(axis=1)
    scores = scores[selected][finite]
    coords = candidates[:, :4].astype(np.float64)
    coords[:, :2] -= coords[:, 2:] / 2
    indices = cv2.dnn.NMSBoxes(coords.tolist(), scores.tolist(), score_thresh,
                               getattr(betaconst, 'nms_threshold', 0.45))
    scale, top, left = scale_params
    boxes = []
    for index in np.asarray(indices).reshape(-1):
        x, y, w, h = coords[index]
        boxes.append({
            'x': float(math.floor((x - left) / scale)),
            'y': float(math.floor((y - top) / scale)),
            'w': float(math.ceil(w / scale)),
            'h': float(math.ceil(h / scale)),
            'class_id': float(class_ids[index]),
            'score': float(scores[index]),
            't': t,
        })
    return boxes


def raw_boxes_for_img(img, session, t):
    backend = detector_backend()
    if backend == 'legacy':
        return legacy_boxes_for_img(img, session, t)
    adj_img, scale = prep_img_for_nn(img)
    try:
        return get_model_output(adj_img, session, scale, t)
    except Exception as err:
        raise DetectorBackendError(f'{backend} detector inference failed: {err}') from err


def legacy_boxes_for_img(img, session, t):
    scale = LEGACY_SIZE / max(img.shape[:2])
    h, w = (max(1, round(length * scale)) for length in img.shape[:2])
    adj_img = cv2.resize(img, (w, h))
    adj_img = cv2.copyMakeBorder(adj_img, 0, LEGACY_SIZE - h, 0, LEGACY_SIZE - w,
                                 cv2.BORDER_CONSTANT, value=0)
    adj_img = adj_img.astype(np.float32)
    adj_img -= [103.939, 116.779, 123.68]
    coords, scores, labels = session.run(LEGACY_OUTPUTS, {'input_1:0': adj_img[None]})

    class_ids = {label: index for index, (label, _) in enumerate(betaconst.classes)}
    boxes = []
    for box, score, label in zip(coords[0], scores[0], labels[0]):
        if not (score > LEGACY_MIN_PROB and 0 <= label < len(LEGACY_CLASSES)):
            continue
        if not np.isfinite(box).all():
            continue
        x1, y1, x2, y2 = box
        boxes.append({
            'x': math.floor(x1 / scale),
            'y': math.floor(y1 / scale),
            'w': math.ceil((x2 - x1) / scale),
            'h': math.ceil((y2 - y1) / scale),
            'class_id': class_ids[LEGACY_CLASSES[int(label)]],
            'score': float(score),
            't': t,
        })
    return boxes


def needs_eye_points(raw, parts_to_blur):
    label = betaconst.classes[int(raw['class_id'])][0]
    return (eyes_enabled() and label == 'FACE_FEMALE' and label in parts_to_blur
            and raw['score'] > parts_to_blur[label]['min_prob'])


def get_eye_points(img, box):
    backend = detector_backend()
    if backend in _eyes_failed:
        return None
    x, y, w, h = (int(box[key]) for key in ('x', 'y', 'w', 'h'))
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    if x2 <= x1 or y2 <= y1:
        return None

    try:
        session = get_eyes_session()
    except Exception as err:
        _eyes_failed.add(backend)
        logging.warning('Eye model unavailable; continuing without eye bars: %s', err)
        return None

    try:
        blob, scale, pad_x, pad_y = resize_face_img(img[y1:y2, x1:x2])
        output = session(blob)[0].reshape(98, 2)
        eyes = (output[[60, 72]] * 112 - [pad_x, pad_y]) / scale
        if not np.isfinite(eyes).all():
            return None
        eyes = np.trunc(eyes).astype(np.int64) + [x1, y1]
        if (eyes < [x1, y1]).any() or (eyes >= [x2, y2]).any():
            return None
        if np.array_equal(eyes[0], eyes[1]):
            return None
        return eyes.tolist()
    except Exception as err:
        logging.warning('Eye inference failed; continuing without eye bar: %s', err)
        return None

def eye_box_key(raw):
    return bu_hash.dictionary_hash([raw[k] for k in ('t', 'x', 'y', 'w', 'h')], 24)


def prepare_cached_eyes(cache, box_path, eye_key):
    if eye_key is None or cache['eye_key'] == eye_key:
        return False
    # Read old sidecars when available. All subsequent writes use the box cache.
    sidecar = box_path.with_name(f'{box_path.stem}-{eye_key}-eyes.gz')
    old_eyes = bu_hash.read_json(sidecar, dict) or {}
    for raw in cache['boxes']:
        eyes = old_eyes.get(eye_box_key(raw)) if old_eyes else None
        if bu_hash.valid_eye_points(eyes):
            raw['eyes'] = eyes
        else:
            raw.pop('eyes', None)
    cache['eye_key'] = eye_key
    return True


def add_eye_points(img, raw_boxes, parts_to_blur):
    changed = False
    for raw in raw_boxes:
        if not needs_eye_points(raw, parts_to_blur) or bu_hash.valid_eye_points(raw.get('eyes')):
            continue
        eyes = get_eye_points(img, raw)
        raw['eyes'] = eyes
        if eyes is not None:
            changed = True
    return changed
