import argparse
import cv2
import logging
import math
import os
import subprocess as sp
import tempfile
import time
from contextlib import closing, suppress
from pathlib import Path

import betaconst
import betaconfig

import betautils_config as bu_config
import betautils_hash as bu_hash
import betautils_model as bu_model
import betautils_video as bu_video
import betautils_censor as bu_censor


QSV_ENCODE_PRESET = 'slow'
CPU_ENCODE_PRESET = 'medium'


def qsv_device_args():
    render_nodes = sorted(Path('/dev/dri').glob('renderD*'))
    if not render_nodes:
        return []
    return ['-init_hw_device', f'qsv=qs:hw,child_device={render_nodes[0]}']


def check_qsv_available():
    command = [
        'ffmpeg', '-v', 'error', *qsv_device_args(), '-f', 'lavfi', '-i', 'color=c=black:s=128x128:r=1',
        '-frames:v', '1', '-vf', 'format=nv12', '-c:v', 'h264_qsv',
        '-global_quality', '22', '-preset', QSV_ENCODE_PRESET,
        '-f', 'null', '-'
    ]
    try:
        result = sp.run(command, capture_output=True, timeout=15)
        if result.returncode:
            logging.info('QSV unavailable: %s', result.stderr.decode(errors='replace').strip()
                         or f'FFmpeg exited with {result.returncode}')
        return result.returncode == 0
    except (OSError, sp.TimeoutExpired) as err:
        logging.warning('QSV probe failed: %s', err)
        return False


def video_frames(filepath, fps, size, *, grab_only=False):
    cap = cv2.VideoCapture(str(filepath))
    try:
        if not cap.isOpened():
            raise OSError(f'Could not open {filepath}')
        frame_index = 0
        previous = -1.0
        base_time = None
        while cap.grab():
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if base_time is None:
                base_time = timestamp if math.isfinite(timestamp) else 0.0
            timestamp -= base_time
            if not math.isfinite(timestamp) or timestamp <= previous:
                timestamp = previous + 1 / fps if frame_index else 0.0
            previous = timestamp
            # A caller can pick the frames that need decoding to a BGR array.
            if grab_only:
                yield frame_index, timestamp, cap
            else:
                yield frame_index, timestamp, retrieve_frame(cap, size)
            frame_index += 1
        if not frame_index:
            raise OSError(f'No frames decoded from {filepath}')
    finally:
        cap.release()


def retrieve_frame(cap, size):
    ret, frame = cap.retrieve()
    if not ret or frame.shape[:2] != (size[1], size[0]):
        raise OSError('Could not decode a frame at the expected dimensions')
    return frame


def detect_video(filepath, fps, size, parts_to_blur, eye_key):
    sample_rate = min(betaconfig.video_censor_fps, fps)
    next_sample = 0.0
    boxes = []
    session = bu_model.get_session()
    with closing(video_frames(filepath, fps, size, grab_only=True)) as frames:
        for index, timestamp, cap in frames:
            if timestamp + 1e-7 < next_sample:
                continue
            frame = retrieve_frame(cap, size)
            raw_boxes = bu_model.raw_boxes_for_img(frame, session, timestamp)
            for raw in raw_boxes:
                raw['frame'] = index
            if eye_key is not None:
                bu_model.add_eye_points(frame, raw_boxes, parts_to_blur)
            boxes.extend(raw_boxes)
            next_sample = (math.floor((timestamp + 1e-7) * sample_rate) + 1) / sample_rate
    return boxes


def fill_missing_eyes(filepath, fps, size, raw_boxes, parts_to_blur):
    pending = {}
    for raw in raw_boxes:
        if bu_model.needs_eye_points(raw, parts_to_blur) and not bu_hash.valid_eye_points(raw.get('eyes')):
            pending.setdefault(raw['frame'], []).append(raw)
    if not pending:
        return False
    changed = False
    with closing(video_frames(filepath, fps, size, grab_only=True)) as frames:
        for index, _, cap in frames:
            group = pending.pop(index, None)
            if group:
                changed |= bu_model.add_eye_points(retrieve_frame(cap, size), group, parts_to_blur)
            if not pending:
                break
    return changed


def render_frames(filepath, fps, size, boxes):
    live_boxes = []
    cursor = 0
    previous = None
    output_index = 0
    with closing(video_frames(filepath, fps, size)) as frames:
        for _, timestamp, frame in frames:
            # FFmpeg consumes CFR raw frames. Use the source timestamps to retain
            # elapsed time when normalizing a variable-rate source.
            output_position = round(timestamp * fps)
            if output_position - output_index > 60 * fps:
                raise ValueError('Video contains a timestamp gap larger than 60 seconds')
            while previous is not None and output_index < output_position:
                yield previous
                output_index += 1
            live_boxes = [box for box in live_boxes if box['end'] >= timestamp]
            while cursor < len(boxes) and boxes[cursor]['start'] <= timestamp:
                if boxes[cursor]['end'] >= timestamp:
                    live_boxes.append(boxes[cursor])
                cursor += 1
            previous = bu_censor.censor_img_for_boxes(frame, live_boxes)
    if previous is not None:
        yield previous


def encode_video(filepath, output, fps, size, boxes, qsv, audio_codec, audio_offset):
    width, height = size
    command = [
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-video_size', f'{width}x{height}', '-framerate', f'{fps:.9f}', '-i', '-'
    ]
    if qsv:
        command[1:1] = ['-filter_threads', '1', *qsv_device_args()]
    if audio_codec:
        command += ['-itsoffset', str(audio_offset), '-i', str(filepath),
                    '-map', '0:v:0', '-map', '1:a', '-c:a', audio_codec]
    encoder = ['-c:v', 'h264_qsv', '-global_quality', '22'] if qsv else [
        '-c:v', 'libx264', '-crf', '22']
    pixel_format = 'nv12' if qsv else 'yuv420p'
    preset = QSV_ENCODE_PRESET if qsv else CPU_ENCODE_PRESET
    command += encoder + [
        '-preset', preset, '-vf', f'pad=ceil(iw/2)*2:ceil(ih/2)*2,format={pixel_format}',
        '-movflags', '+faststart', str(output)
    ]

    frames_written = 0
    with tempfile.TemporaryFile() as stderr:
        with sp.Popen(command, stdin=sp.PIPE, stderr=stderr) as proc:
            try:
                try:
                    with closing(render_frames(filepath, fps, size, boxes)) as frames:
                        for frame in frames:
                            proc.stdin.write(memoryview(frame))
                            frames_written += 1
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
                returncode = proc.wait()
            except BaseException:
                with suppress(ProcessLookupError):
                    proc.kill()
                with suppress(OSError):
                    proc.stdin.close()
                raise
        if returncode:
            stderr.seek(0)
            return stderr.read().decode(errors='replace').strip() or f'FFmpeg exited with {returncode}'
    try:
        streams = bu_video.probe_video(output)['streams']
    except (sp.SubprocessError, OSError, ValueError, KeyError) as err:
        return f'Could not validate encoded video: {err}'
    video = next((stream for stream in streams if stream['codec_type'] == 'video'), {})
    frame_count = str(video.get('nb_frames', ''))
    if not frames_written or not frame_count.isdigit() or int(frame_count) != frames_written:
        return f'Encoded frame count mismatch: wrote {frames_written}, got {frame_count or "unknown"}'
    if audio_codec and not any(stream['codec_type'] == 'audio' for stream in streams):
        return 'Encoded video has no audio'
    return None


def process_video(filepath, censored_folder, detector_key, eye_key, censor_hash,
                  parts_to_blur, has_qsv):
    file_hash = bu_hash.md5_for_file(filepath, 24)
    censored_path = censored_folder / f'{filepath.stem}-{file_hash}-{censor_hash}.mp4'
    if censored_path.is_file():
        logging.info('Skipping existing output: %s', filepath.name)
        return

    start = time.perf_counter()
    logging.info('Processing video: %s', filepath.name)
    probe = bu_video.probe_video(filepath)
    streams = probe['streams']
    video = next((stream for stream in streams if stream['codec_type'] == 'video'
                  and not stream.get('disposition', {}).get('attached_pic')), None)
    if video is None:
        logging.info('Skipping non-video: %s', filepath.name)
        return
    audio = [stream for stream in streams if stream['codec_type'] == 'audio']
    audio_offset = bu_video.stream_start(probe.get('format', {})) - bu_video.stream_start(video)
    cap = cv2.VideoCapture(str(filepath))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read()
        if not ret or not math.isfinite(fps) or fps <= 0:
            raise OSError('Could not read first video frame or determine frame rate')
        size = (frame.shape[1], frame.shape[0])
    finally:
        cap.release()

    box_path = betaconst.PROJECT_PATH / 'vid_hashes' / f'{file_hash}-{detector_key}.gz'
    cache = bu_hash.read_box_cache(box_path, video=True)
    changed = cache is None
    if cache is None:
        logging.info('Detecting frames: %s', filepath.name)
        cache = {'boxes': detect_video(filepath, fps, size, parts_to_blur, eye_key),
                 'eye_key': eye_key}
    elif eye_key is not None:
        changed |= bu_model.prepare_cached_eyes(cache, box_path, eye_key)
        changed |= fill_missing_eyes(filepath, fps, size, cache['boxes'], parts_to_blur)
    if changed:
        bu_hash.write_json(cache, box_path)
    raw_boxes = cache['boxes']

    boxes = []
    for raw in raw_boxes:
        box = bu_censor.process_raw_box(raw, *size, parts_to_blur)
        if box is not None:
            boxes.append(box)
    boxes.sort(key=lambda box: box['start'])

    audio_codec = 'copy' if audio else None
    attempts = [(True, audio_codec), (False, audio_codec)] if has_qsv else [(False, audio_codec)]
    if audio:
        attempts.append((False, 'aac'))
    while attempts:
        qsv, codec = attempts.pop(0)
        logging.info('Encoding %s with %s (audio=%s)', filepath.name,
                     'QSV' if qsv else 'libx264', codec or 'none')
        with bu_hash.temporary_file(censored_path) as temp:
            error = encode_video(filepath, temp, fps, size, boxes, qsv, codec, audio_offset)
            if error is None:
                os.replace(temp, censored_path)
                break
            logging.warning('Encoding failed (%s, audio=%s): %s', 'QSV' if qsv else 'CPU', codec, error)
            if codec == 'copy' and 'codec not currently supported in container' in error:
                attempts = [(True, 'aac'), (False, 'aac')] if qsv else [(False, 'aac')]
    else:
        raise OSError('All video encoding attempts failed')
    delete_result = bu_config.delete_file_with_probability(filepath, censored_path)
    logging.info('Processed %s in %.3fs%s', filepath.name, time.perf_counter() - start, delete_result)


def process_video_with_detector(filepath, censored_folder, parts_to_blur, has_qsv,
                                detector_key, eye_key):
    censor_hash = bu_hash.get_censor_hash([
        detector_key, eye_key, 'h264',
        QSV_ENCODE_PRESET if has_qsv else None,
        CPU_ENCODE_PRESET,
    ])
    process_video(filepath, censored_folder, detector_key, eye_key, censor_hash,
                  parts_to_blur, has_qsv)


def main(*, confirm_input_delete=True):
    bu_config.setup_logging()
    parts_to_blur = bu_config.get_parts_to_blur()
    bu_config.validate_config(parts_to_blur)
    if confirm_input_delete:
        bu_config.verify_input_delete_probability()
    has_qsv = check_qsv_available()
    logging.info('Video encoding: %s', 'QSV with CPU fallback' if has_qsv else 'CPU')
    input_dir = Path(betaconst.video_path_uncensored)
    output_dir = Path(betaconst.video_path_censored)
    for root, _, f_names in os.walk(input_dir):
        root = Path(root)
        censored_folder = output_dir / root.relative_to(input_dir)
        for fname in f_names:
            try:
                bu_model.run_with_detector(
                    lambda detector_key, eye_key: process_video_with_detector(
                        root / fname, censored_folder, parts_to_blur, has_qsv, detector_key, eye_key
                    ),
                    betaconfig.video_censor_fps
                )
            except Exception as err:
                logging.error('Skipping %s: %s', root / fname, err)


def cli():
    parser = argparse.ArgumentParser(description='Process BetaSuite media.')
    parser.add_argument('--all', action='store_true', help='Process images before videos.')
    if parser.parse_args().all:
        import betastare

        betastare.main()
        main(confirm_input_delete=False)
    else:
        main()


if __name__ == "__main__":
    bu_config.run_interruptible(cli)
