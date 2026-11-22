# 1: prefer OpenVINO CPU; 0: skip OpenVINO and use ONNX Runtime CPU.
gpu_enabled = 1

##################################################################
#每秒抽样检测次数
video_censor_fps = 5

###################################################################
###### 屏蔽设置

#反转模式。0为正常模式，1为反转模式。反转模式选中部分不会打码，其余部分全图打码。
reverse_mode = 1

### 类型选择
### 删除前面的#为未选中；增加前面的#为选中
items_to_censor = [
    #'FEMALE_GENITALIA_COVERED',
    'FACE_FEMALE',
    #'BUTTOCKS_EXPOSED',
    #'FEMALE_BREAST_EXPOSED',
    #'FEMALE_GENITALIA_EXPOSED',
    #'MALE_BREAST_EXPOSED',
    #'ANUS_EXPOSED',
    'FEET_EXPOSED',
    #'BELLY_COVERED',
    'FEET_COVERED',
    #'ARMPITS_COVERED',
    #'ARMPITS_EXPOSED',
    #'FACE_MALE',
    #'BELLY_EXPOSED',
    #'MALE_GENITALIA_EXPOSED',
    #'ANUS_COVERED',
    #'FEMALE_BREAST_COVERED',
    #'BUTTOCKS_COVERED',
]

# censor style: whether the default censor should be black bars, pixelate, or blur.  Uncomment
# one of the below (by removing the #).
# you can override these per item below in Item Overrides
#default_censor_style = [ 'bar', (0,0,0) ] # second item is the color of the bar, in RGB code 
#default_censor_style = [ 'blur', 10 ] # second item is how aggressive a blur.  20 is a reasonable number.  Higher is more blurry.
default_censor_style = [ 'pixel', 2 ] # second item is the how much to pixelate.  Higher is more censored.  10 means that a 200x400 pixel region is pixelated to 20x40 pixels.

#默认打码区域形状选择：rectangle（矩形）or circle（圆形）
default_shape_style = 'rectangle'

# 默认最小置信度值。置信度低于此值的区域将不会被处理
default_min_prob = 0.4     #0.50 means 50% certainty

# area_safety: do we want to censor a safety region around the identified region
# note that in the overrides below, this can be set independently for width and height
default_area_safety = 0.1 # i.e., 0.2 means add 20% to width and 20% to height

# time_safety: how long before and after the identification do we want to censor?
default_time_safety = 0.4 #i.e., 0.3 means 0.2 seconds before and 0.2 seconds after

# Item Overrides: to override any of the above defaults for a specific item
item_overrides = {
        #example:
        #FEMALE_GENITALIA_EXPOSED : {'censor_style': [ 'pixel', 10 ], shape:rotundity, 'min_prob': 0.40, 'width_area_safety': 0.3, 'height_area_safety': 0.1, 'time_safety': 0.6, 'shape_style': 'circle' },

        'FEMALE_GENITALIA_EXPOSED':  {'width_area_safety': 0.4, 'height_area_safety': 0.4 },
        'FEMALE_BREAST_EXPOSED': {'width_area_safety': 0.4 },
        'FEMALE_BREAST_COVERED': {'width_area_safety': 0.4 },
        'FACE_FEMALE': {'min_prob': 0.5, 'shape_style': 'circle', 'width_area_safety': 0.1, 'height_area_safety': 0.2 },
        'FEET_EXPOSED': {'min_prob': 0.2, 'censor_style': [ 'pixel', 10 ] },
        'FEET_COVERED': {'min_prob': 0.25, 'censor_style': [ 'pixel', 10 ] },

}

#眼部打码（开关，颜色）
eyes_bar = [ 'on', ( 0,0,0 ) ]

#是否开启边缘框。第二项为框颜色rgb，第三项为宽度。
enable_area_bar = ['off', (255,0,0), 3]  # 状态、RGB颜色、线宽

################################################
# Input Delete Probability
#### If you set this to 1, all uncensored files
#### will be deleted after censoring.  If you set
#### this to a number between 0 and 1, there will 
#### be a random chance of the input being deleted.
#### For example, if you set it to 0.7, for each
#### uncensored file, there is a 70% chance it will
#### be deleted.
#### I ***HIGHLY*** recommend testing censoring
#### with this set to zero first.  Using this 
#### setting is dangerous.  BetaSuite will try 
#### to verify that the file was successfully 
#### censored before deleting, but if there is
#### some issue with the image or video writing
#### process, BetaSuite might not know.
#### SET THIS AT YOUR OWN RISK.
#### DON'T COME CRYING IF YOUR FILE WAS DELETED
input_delete_probability = 0

###################################################################
#### Miscellaneous

# this determines how various censor styles deal with overlap
# You probably don't need to change this
censor_overlap_strategy = {
        'blur': 'single-pass',
        'bar': 'none',
        'pixel': 'single-pass',
        'debug': 'none',
        }

censor_overlap_rate = 0.3

# this determines how pixel and blur censors 
# "scale" with image size.  'feature' is recommended
censor_scale_strategy = 'feature' # scales N->1 by min feature dimension, with a 100 base (so a 200x400 feature would be 2N->1)
#censor_scale_strategy = 'image' # scales N->1 by max image dimension, with 1000 base (so a 2000x1200 image would be 2N->1)
#censor_scale_strategy = 'none' # uses N -> 1 reduction

### Debug 模式
debug_mode = 0
#debug_mode = 1 # just show debug information
#debug_mode = 3 # also save debug info to BetaSuite folder

### 开启水印
#enable_betasuite_watermark = True
enable_betasuite_watermark = False
#水印内容。
watermark_text = "made by telegram" #暂不支持中文
#水印颜色（RGB）
watermark_text_color = (0,0,0)
