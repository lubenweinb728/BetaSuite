global_min_prob = 0.30
picture_saved_box_version = 1

ptb_hash_len = 4

model_outputs = [
        'filtered_detections/map/TensorArrayStack/TensorArrayGatherV3:0',   # boxes
        'filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3:0', # scores
        'filtered_detections/map/TensorArrayStack_2/TensorArrayGatherV3:0', # classes
]

model_input = 'images'

picture_path_uncensored = '../uncensored_pics/'
picture_path_censored   = '../censored_pics/'

video_path_uncensored = '../uncensored_vids/'
video_path_censored   = '../censored_vids/'

# includes the RGB debug square color.  Order matches the 640m.onnx model.
classes = [
    ['covered_belly',    (0, 100,   0)],  # darkgreen
    ['covered_breast',   (0, 255, 255)],  # aqua
    ['covered_buttocks', (255,   0,   0)],  # red
    ['covered_feet',     (0, 255,   0)],  # lime
    ['covered_vulva',    (216, 191, 216)],  # thistle
    ['exposed_anus',     (47,  79,  79)],  # darkslategray
    ['exposed_armpits',  (139,  69,  19)],  # saddlebrown
    ['exposed_belly',    (0,   0, 139)],  # darkblue
    ['exposed_breast',   (0,   0, 255)],  # blue
    ['exposed_buttocks', (255, 165,   0)],  # orange
    ['exposed_chest',    (30, 144, 255)],  # dodgerblue
    ['exposed_feet',     (0, 250, 154)],  # mediumspringgreen
    ['exposed_penis',    (240, 230, 140)],  # khaki
    ['exposed_vulva',    (255,   0, 255)],  # fuchsia
    ['face_femme',       (255, 255,   0)],  # yellow
    ['face_masc',        (199,  21, 133)],  # mediumvioletred
]


def class_name_from_id(class_id: int) -> str:
    """Return class name for a given numeric id from the 640m model.

    Args:
        class_id: Numeric identifier from the model output.

    Returns:
        The corresponding class name, or an empty string if the id is out of
        range.
    """
    if 0 <= class_id < len(classes):
        return classes[class_id][0]
    return ''

bv_ss_timestamp1_name = 'bv_ss_timestamp1_name'
bv_ss_timestamp2_name = 'bv_ss_timestamp2_name'
bv_detect_timestamp1_name = 'bv_detect_timestamp1_name'
bv_detect_timestamp2_name = 'bv_detects_timestamp2_name'

bv_detect_shm_0_name = "bv_detect_shm_0_name"
bv_detect_shm_1_name = "bv_detect_shm_1_name"
bv_detect_shm_2_name = "bv_detect_shm_2_name"

imwrite_extensions = [
        'bmp',
        'dib',
        'jpeg',
        'jpg',
        'jpe',
        'jp2',
        'png',
        'webp',
        'pbm',
        'pgm',
        'pxm',
        'pnm',
        'sr',
        'ras',
        'tiff',
        'tif',
        'exr',
        'hdr',
        'pic',
]
