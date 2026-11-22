import gzip
import hashlib
import json
import logging
import math
import os
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import betautils_config as bu_config

import betaconst
import betaconfig

def dictionary_hash( to_hash, hash_len ):
    dict_hash = hashlib.md5(json.dumps( to_hash, sort_keys=True, ensure_ascii=True).encode('utf-8')).hexdigest()
    return( dict_hash[:hash_len] )

def get_censor_hash(extra=None):
    hash_dict = {
        'render_version': 2,
        'parts_to_blur': bu_config.get_parts_to_blur(),
        'extra': extra,
    }
    for key in ('reverse_mode', 'default_censor_style', 'censor_overlap_strategy',
                'censor_overlap_rate', 'censor_scale_strategy', 'eyes_bar',
                'enable_area_bar', 'enable_betasuite_watermark', 'watermark_text',
                'watermark_text_color', 'debug_mode'):
        hash_dict[key] = getattr(betaconfig, key)
    return dictionary_hash(hash_dict, 16)


@lru_cache(maxsize=None)
def model_hash(path):
    path = Path(path)
    return [md5_for_file(path, 32)]


def detection_hash(model_path, sample_rate=None, legacy=None, *, backend=None):
    settings = {
        'version': betaconst.picture_saved_box_version,
        'model': model_hash(model_path),
        'classes': betaconst.classes,
        'min_prob': betaconst.global_min_prob,
        'nms_threshold': getattr(betaconst, 'nms_threshold', 0.45),
        'sample_rate': sample_rate,
    }
    if backend is not None:
        settings['backend'] = backend
    if legacy is not None:
        settings['legacy'] = legacy
    return dictionary_hash(settings, 20)


@contextmanager
def temporary_file(filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=filename.parent, prefix='.betasuite-',
                                     suffix=filename.suffix, delete=False) as file:
        temporary_path = Path(file.name)
    try:
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json(variable, filename):
    with temporary_file(filename) as temp:
        with gzip.open(temp, 'wt', encoding='UTF-8') as fout:
            json.dump(variable, fout, allow_nan=False)
        os.replace(temp, filename)


def read_json(filename, expected_type):
    try:
        with gzip.open(filename, 'rt', encoding='UTF-8') as fin:
            result = json.load(fin)
        if not isinstance(result, expected_type):
            raise ValueError('Unexpected cache format')
        return result
    except FileNotFoundError:
        return None
    except (OSError, EOFError, ValueError) as err:
        logging.warning('Ignoring unreadable cache %s: %s', filename, err)
        return None


def valid_eye_points(eyes):
    return isinstance(eyes, list) and len(eyes) == 2 and all(
        isinstance(point, list) and len(point) == 2 and
        all(type(value) is int for value in point) for point in eyes
    )


def read_box_cache(filename, video=False):
    cache = read_json(filename, (list, dict))
    boxes = cache.get('boxes') if isinstance(cache, dict) else cache
    if not isinstance(boxes, list):
        return None
    for box in boxes:
        try:
            if not all(math.isfinite(box[key]) for key in ('x', 'y', 'w', 'h', 'score', 't')):
                return None
            if video and (type(box['frame']) is not int or box['frame'] < 0):
                return None
            class_id = box['class_id']
            if int(class_id) != class_id or not 0 <= class_id < len(betaconst.classes):
                return None
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if 'eyes' in box and not valid_eye_points(box['eyes']):
            box['eyes'] = None
    if isinstance(cache, list):
        return {'boxes': boxes, 'eye_key': None}
    return {'boxes': boxes, 'eye_key': cache.get('eye_key')}


def md5_for_file( filename, length ):
    assert length <= 32
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(1024 * 1024):
            file_hash.update(chunk)

    return(file_hash.hexdigest()[32-length:]) 
