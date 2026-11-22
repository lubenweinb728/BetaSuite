from pathlib import Path


PROJECT_PATH = Path(__file__).resolve().parent.parent

global_min_prob = 0.2

picture_saved_box_version = 2
DETECTOR_SIZE = 320

picture_path_uncensored = str(PROJECT_PATH / 'uncensored_pics')
picture_path_censored   = str(PROJECT_PATH / 'censored_pics')

video_path_uncensored = str(PROJECT_PATH / 'uncensored_vids')
video_path_censored   = str(PROJECT_PATH / 'censored_vids')

# includes the RGB debug square color
classes = [
    [ 'FEMALE_GENITALIA_COVERED',     ( 47,  79,  79), ], #darkslategray
    [ 'FACE_FEMALE',  (139,  69,  19), ], #saddlebrown
    [ 'BUTTOCKS_EXPOSED',    (  0, 100,   0), ], #darkgreen
    [ 'FEMALE_BREAST_EXPOSED',    (  0,   0, 139), ], #darkblue
    [ 'FEMALE_GENITALIA_EXPOSED', (255,   0,   0), ], #red
    [ 'MALE_BREAST_EXPOSED', (255, 165,   0), ], #orange
    [ 'ANUS_EXPOSED',       (255, 255,   0), ], #yellow
    [ 'FEET_EXPOSED',        (199,  21, 133), ], #mediumvioletred
    [ 'BELLY_COVERED',     (  0, 255,   0), ], #lime
    [ 'FEET_COVERED',     (  0, 250, 154), ], #mediumspringgreen
    [ 'ARMPITS_COVERED',   (  0, 255, 255), ], #aqua
    [ 'ARMPITS_EXPOSED',   (  0,   0, 255), ], #blue
    [ 'FACE_MALE',    (216, 191, 216), ], #thistle
    [ 'BELLY_EXPOSED',    (255,   0, 255), ], #fuchsia
    [ 'MALE_GENITALIA_EXPOSED',    ( 30, 144, 255), ], #dodgerblue
    [ 'ANUS_COVERED',    (240, 230, 140), ], #khaki
    [ 'FEMALE_BREAST_COVERED',    ( 30, 144, 255), ], #dodgerblue
    [ 'BUTTOCKS_COVERED',    (240, 230, 140), ], #khaki
]
