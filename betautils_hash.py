import gzip
import hashlib
import json

import betautils_config as bu_config
import betaconst
import betaconfig

def dictionary_hash(to_hash, hash_len):
    dict_hash = hashlib.md5(json.dumps(to_hash, sort_keys=True, ensure_ascii=True).encode('utf-8')).hexdigest()
    return dict_hash[:hash_len]

def get_censor_hash():
    parts_to_blur = bu_config.get_parts_to_blur()
    hash_dict = {
        'parts_to_blur': parts_to_blur,
        'picture_sizes': betaconfig.picture_sizes,
        'censor_overlap_strategy': betaconfig.censor_overlap_strategy,
        'censor_scale_strategy': betaconfig.censor_scale_strategy,
        'enable_betasuite_watermark': betaconfig.enable_betasuite_watermark,
    }
    ptb_hash = dictionary_hash(hash_dict, betaconst.ptb_hash_len)
    return ptb_hash

def write_json(variable, filename):
    # 自动附加 eye_boxes 数据（如果存在）
    try:
        from betautils_model import get_eye_boxes
        eye_boxes = get_eye_boxes()
        if isinstance(variable, list):
            data = {
                "raw_boxes": variable,
                "eye_boxes": eye_boxes
            }
            print(f"存入缓存的eye_boxes数据: {data['eye_boxes']}")
        else:
            data = variable
    except Exception as e:
        print("Warning: write_json failed to get eye_boxes:", e)
        data = variable

    with gzip.open(filename, 'wt', encoding='UTF-8') as fout:
        json.dump(data, fout)

def read_json(filename):
    with gzip.open(filename, 'rt', encoding='UTF-8') as fin:
        data = json.load(fin)

        # 如果包含 eye_boxes 就恢复
        if isinstance(data, dict) and "raw_boxes" in data:
            try:
                from betautils_model import set_eye_boxes
                if "eye_boxes" in data:
                    print(f"恢复缓存中eye_boxes数据: {data['eye_boxes']}")
                    set_eye_boxes(data["eye_boxes"])
            except Exception as e:
                print("Warning: Failed to restore eye_boxes from cache:", e)
            return data["raw_boxes"]
            
        print("Read eye_boxes from cache:", data.get("eye_boxes"))

        return data


def md5_for_file(filename, length):
    assert length <= 32
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()[32 - length:]
