import logging
import math
import os
import random
import signal
import sys

import betaconfig
import betaconst

class LogFormatter(logging.Formatter):
    def format(self, record):
        line = super().format(record)
        # Keep multiline errors and terminal controls inside one log record.
        return line.translate({
            13: r'\r', 10: r'\n', 9: r'\t', 8: r'\b', 27: r'\x1b',
        })


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(LogFormatter('%(asctime)s %(levelname)s %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S'))
    logging.basicConfig(level=logging.DEBUG if betaconfig.debug_mode & 1 else logging.INFO,
                        handlers=[handler])


class StopProcessing(KeyboardInterrupt):
    pass


def run_interruptible(main):
    stopping = False

    def stop(signum, _frame):
        nonlocal stopping
        if not stopping:
            stopping = True
            raise StopProcessing(signal.Signals(signum).name)

    signals = (signal.SIGTERM, signal.SIGINT)
    previous = {signum: signal.signal(signum, stop) for signum in signals}
    try:
        main()
    except StopProcessing as err:
        logging.info('Stopped on %s. Exiting.', err)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def get_parts_to_blur():
    parts_to_blur={}
    for item in betaconfig.items_to_censor:
        parts_to_blur[item] = {
                'min_prob':           ( betaconfig.item_overrides.get( item, {} ) ).get( 'min_prob',           betaconfig.default_min_prob    ),
                'width_area_safety':  ( betaconfig.item_overrides.get( item, {} ) ).get( 'width_area_safety',  betaconfig.default_area_safety ),
                'height_area_safety': ( betaconfig.item_overrides.get( item, {} ) ).get( 'height_area_safety', betaconfig.default_area_safety ),
                'time_safety':        ( betaconfig.item_overrides.get( item, {} ) ).get( 'time_safety',        betaconfig.default_time_safety ),
                'censor_style':       ( betaconfig.item_overrides.get( item, {} ) ).get( 'censor_style',       betaconfig.default_censor_style ),
                'shape_style':       ( betaconfig.item_overrides.get( item, {} ) ).get( 'shape_style',       betaconfig.default_shape_style ),
        }

    if betaconfig.debug_mode&1:
        for item in betaconst.classes:
            parts_to_blur[item[0]] = {
                    'min_prob': betaconst.global_min_prob,
                    'width_area_safety': 0,
                    'height_area_safety': 0,
                    'time_safety': 0.4,
                    'censor_style': [ 'debug', item[1] ],
                    'shape_style': betaconfig.default_shape_style
            }

    return parts_to_blur

def verify_input_delete_probability():
    prob = betaconfig.input_delete_probability
    if prob == 0:
        return()

    logging.warning('Source files may be deleted after processing (%.2f%% chance). '
                    'Type DELETE MY FILES to confirm.', 100 * prob)
    try:
        response = input()
    except EOFError:
        response = ''
    if response != 'DELETE MY FILES':
        logging.error('Source deletion was not confirmed. Aborting.')
        raise SystemExit(1)
    logging.info('Source deletion confirmed (%.2f%% chance).', 100 * prob)


def delete_file_with_probability( delete_path, check_path ):
    prob = betaconfig.input_delete_probability
    if prob == 0:
        return('')

    if not os.path.exists( check_path ) or os.path.getsize( check_path ) < 1000:
        return( ' (Original NOT DELETED - censored version NOT FOUND)' )

    random_number = random.random()
    if random_number < prob:
        os.remove( delete_path )
        return( ' (Original DELETED!!!!)' )
    else:
        return( ' (Original not deleted)' )

def validate_config(parts_to_blur):
    if betaconfig.gpu_enabled not in (0, 1):
        raise ValueError('gpu_enabled must be 0 or 1')
    if not 0 <= betaconfig.input_delete_probability <= 1:
        raise ValueError('input_delete_probability must be between 0 and 1')
    if not math.isfinite(betaconfig.video_censor_fps) or betaconfig.video_censor_fps <= 0:
        raise ValueError('video_censor_fps must be positive')
    labels = {item[0] for item in betaconst.classes}
    if set(betaconfig.items_to_censor) - labels:
        raise ValueError('Unknown detection category in items_to_censor')
    if betaconfig.censor_scale_strategy not in ('none', 'feature', 'image'):
        raise ValueError('Invalid censor_scale_strategy')
    if not 0 < betaconfig.censor_overlap_rate <= 1:
        raise ValueError('censor_overlap_rate must be in (0, 1]')
    if betaconfig.default_censor_style[0] not in ('bar', 'pixel', 'blur'):
        raise ValueError('Invalid default censor style')
    for setting in ('eyes_bar', 'enable_area_bar'):
        config = getattr(betaconfig, setting)
        if config[0] not in ('on', 'off'):
            raise ValueError(f'{setting} must be on or off')
    styles = [betaconfig.default_censor_style]
    for part in parts_to_blur.values():
        styles.append(part['censor_style'])
        if part['shape_style'] not in ('rectangle', 'circle'):
            raise ValueError('Invalid shape_style')
        if not 0 <= part['min_prob'] <= 1:
            raise ValueError('min_prob must be between 0 and 1')
        for key in ('width_area_safety', 'height_area_safety', 'time_safety'):
            if not math.isfinite(part[key]) or part[key] < 0:
                raise ValueError(f'{key} must be nonnegative and finite')
    for style, value in styles:
        if style not in ('bar', 'pixel', 'blur', 'debug'):
            raise ValueError(f'Unknown censor style: {style}')
        if style in ('pixel', 'blur') and (not math.isfinite(value) or value <= 0):
            raise ValueError(f'{style} strength must be positive and finite')
        if style in ('bar', 'debug') and (len(value) != 3 or any(not 0 <= c <= 255 for c in value)):
            raise ValueError('Expected three color channels between 0 and 255')
        if betaconfig.censor_overlap_strategy.get(style) not in ('none', 'single-pass'):
            raise ValueError(f'Invalid overlap strategy for {style}')
