import cv2
import logging
import os
import time
import hashlib

from pathlib import Path

import numpy as np

import betaconst
import betautils_config as bu_config
import betautils_hash as bu_hash
import betautils_model as bu_model
import betautils_censor as bu_censor


def process_image(image, uncensored_path, censored_folder, parts_to_blur,
                  detector_key, eye_key, t0):
    fname = uncensored_path.name
    censor_hash = bu_hash.get_censor_hash([detector_key, eye_key])
    img_h, img_w = image.shape[:2]
    image_hash = hashlib.md5(image).hexdigest()
    image_key = f'{image_hash}-{img_w}x{img_h}'
    censored_path = censored_folder / (
        f'{uncensored_path.stem}-{image_key}-{censor_hash}{uncensored_path.suffix}')
    if censored_path.is_file():
        logging.info('Skipping existing output: %s', fname)
        return

    box_path = betaconst.PROJECT_PATH / 'pic_hashes' / f'{image_key}-{detector_key}.gz'
    cache = bu_hash.read_box_cache(box_path)
    changed = cache is None
    if cache is None:
        cache = {'boxes': bu_model.raw_boxes_for_img(image, bu_model.get_session(), 0),
                 'eye_key': eye_key}
    raw_boxes = cache['boxes']
    if eye_key is not None:
        changed |= bu_model.prepare_cached_eyes(cache, box_path, eye_key)
        changed |= bu_model.add_eye_points(image, raw_boxes, parts_to_blur)
    if changed:
        bu_hash.write_json(cache, box_path)

    boxes = []
    for raw in raw_boxes:
        box = bu_censor.process_raw_box(raw, img_w, img_h, parts_to_blur)
        if box is not None:
            boxes.append(box)
    image = bu_censor.censor_img_for_boxes(image, boxes)
    success, encoded = cv2.imencode(uncensored_path.suffix, image)
    if not success:
        raise OSError(f'Could not encode {fname}')
    with bu_hash.temporary_file(censored_path) as temp:
        encoded.tofile(temp)
        os.replace(temp, censored_path)
    delete_result = bu_config.delete_file_with_probability(uncensored_path, censored_path)
    logging.info('Processed %s in %.3fs%s', fname, time.perf_counter() - t0, delete_result)


def main():
    bu_config.setup_logging()
    parts_to_blur = bu_config.get_parts_to_blur()
    bu_config.validate_config(parts_to_blur)
    bu_config.verify_input_delete_probability()
    input_dir = Path(betaconst.picture_path_uncensored)
    output_dir = Path(betaconst.picture_path_censored)

    for root, _, f_names in os.walk(input_dir):
        root = Path(root)
        censored_folder = output_dir / root.relative_to(input_dir)
        for fname in f_names:
            t0 = time.perf_counter()
            uncensored_path = root / fname
            try:
                image = cv2.imdecode(np.fromfile(uncensored_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    logging.info('Skipping non-image: %s', fname)
                    continue
                bu_model.run_with_detector(
                    lambda detector_key, eye_key: process_image(
                        image, uncensored_path, censored_folder, parts_to_blur,
                        detector_key, eye_key, t0
                    )
                )
            except Exception as err:
                logging.error('Skipping %s: %s', uncensored_path, err)


if __name__ == "__main__":
    bu_config.run_interruptible(main)
