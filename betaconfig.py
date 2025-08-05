###################################################################
#### If you have completed the 'GPU Acceleration' Steps in the 
#### BetaVision install guide, set this to 1, and it will
#### very, very significantly speed up censoring.
#### If you have not, set this to 0.
#### Only BetaStare and BetaTV are supported with gpu_enabled = 0, 
#### BetaVision is only supported with gpu_enabled=1.
#是否开启gpu加速，开启为1，关闭为0。
gpu_enabled=0

###################################################################
#### Neural Net input size.  This controls how the net views the image.
#### Different values may be best for different photos.
#### 
#### [ 1280 ]: recommended, good performance and speed tradeoff
#### [ 640 ]:  much faster, decent performance when model fills the frame
#### [ 2560 ]: slower, better for collage or big group photos.  
#### 
#### [ 1280, 2560 ]: combine the two, may be safer than either alone
####
#### [ 0 ]: not recommended, uses photos as-is to neural net.
#神经网络输入图片大小，推荐1280。640较快，2560精细，0为原始图片适合图片打码。
picture_sizes = [ 1280 ] 

##################################################################
#### BetaTV does not run the neural net on every single frame of a
#### video.  Instead, it runs a few times per second, and then 
#### uses those to censor the whole video.  Set the neural net
#### runs per second here.  More means better accuracy, but slower.
#### 5 is a good sweet spot.  You can drop to 3 if 5 is too slow,
#### but you should probably increase the time_safety values
#### in the censoring config if you do.  15 is what I use for
#### best accuracy.
#视频帧率，指每秒截几张图并运行神经网络的次数。帧数越高打码效果越流畅，也越消耗性能。
video_censor_fps = 15

###################################################################
#反转模式。0为正常模式，1为反转模式。反转模式选中部分不会打码，其余部分打码。
reverse_mode = 1

### Items to censor
### Put a # in front of a line to turn it off.
#选择需要打码（显露）的部位，选中的删除前面的#，不选的前面加#。
items_to_censor = [
    #'exposed_anus',#屁眼
    #'exposed_vulva',#逼
    #'exposed_breast',#奶子
    #'exposed_buttocks',#屁股
    #'covered_vulva',#穿着内裤的逼
    #'covered_breast',#穿着胸罩的奶子
    #'covered_buttocks',#穿着内裤的屁股
    'face_femme',#女人的脸
    #'exposed_belly',#肚子
    #'covered_belly',#穿着衣服的肚子
    'exposed_feet',#裸足
    'covered_feet',#穿鞋的脚
    'exposed_armpits',#胳肢窝
    #'exposed_penis',#鸡巴
    #'exposed_chest',#胸（男）
    #'face_masc',#男人的脸
]

# censor style: whether the default censor should be black bars, pixelate, or blur.  Uncomment
# one of the below (by removing the #).
# you can override these per item below in Item Overrides
#default_censor_style = [ 'bar', (0,0,0) ] # second item is the color of the bar, in RGB code 
#默认打码方式。bar为条，blur为模糊，pixel为像素化。
#default_censor_style = [ 'bar', (255,255,255) ] # 第二项是条形的颜色，以 RGB 代码表示
default_censor_style = [ 'blur', 10 ] # 第二项是模糊程度。数值越高，模糊程度越强。一般为20。
#default_censor_style = [ 'pixel', 1 ] # 第二项是像素化程度。值越高，像素化程度越高。10 表示将 200x400 像素的区域像素化为 20x40 像素。
#default_censor_style = [ 'img', 'beta' ] # 第二项是图片名称。
#default_censor_style = [ 'line', 5 ] # 轮廓线模式，第二项是阈值（0-255）。

eyes_bar = [ 'on', (0,0,0), 1]  # 状态、RGB颜色

#是否开启边缘框。第二项为框颜色rgb，第三项为宽度。
enable_area_bar = ['on', (255,0,0), 6]  # 状态、RGB颜色、线宽

#打码形状（rectangle为矩形，circle为圆形）
default_censor_shape = 'rectangle'

# min_prob: how confident are we that the item is identified before blocking
default_min_prob = 0.4 #0.50 means 50% certainty

# area_safety: do we want to censor a safety region around the identified region
# note that in the overrides below, this can be set independently for width and height
##以下为打码范围控制，指神经网络识别的区域是否扩张或缩小。也可单独设置高（height）扩张或缩小和宽（width）扩张或缩小。
#默认打码范围，0.2表示扩张20%，-0.2表示缩小20%。
default_area_safety = 0 # i.e., 0.2 means add 20% to width and 20% to height

# time_safety: how long before and after the identification do we want to censor?
##以下为打码时间控制，指神经网络识别后提前或延后打码的时间。
#默认打码时间，0为识别到的那一刻打码。0.4表示提前0.4秒打码，-0.4表示延后0.4秒打码。
default_time_safety = 0.4 #i.e., 0.3 means 0.3 seconds before and 0.3 seconds after

# Item Overrides: to override any of the above defaults for a specific item
#打码部位单独设置打码方式，如需使用默认参数或前项未选中的部位可将下面每项前加#，或删除。
item_overrides = {
        #example演示:
        #'上述需要打码的部位': {'打码方式': [ 'pixel', 10 ], 'min_prob': 0.40, '宽打码范围': 0.3, '高打码范围': 0.1, '打码时间': 0.6 }
        #face_masc : {'censor_style': [ 'pixel', 10 ], 'min_prob': 0.40, 'width_area_safety': 0.3, 'height_area_safety': 0.1, 'time_safety': 0.6 },

        #'exposed_anus':  {'min_prob': 0.40, 'width_area_safety': 0.4, 'height_area_safety': 0.4 },
        #'exposed_vulva':  {'min_prob': 0.40, 'width_area_safety': 0.4, 'height_area_safety': 0.4 },
        #'exposed_breast': {'min_prob': 0.40, 'width_area_safety': 0.4 },
        #'exposed_buttocks': {'min_prob': 0.40, 'width_area_safety': 0.4 },
        #'covered_vulva': {'min_prob': 0.40, 'width_area_safety': 0.4 },
        #'covered_breast': {'min_prob': 0.40, 'width_area_safety': 0.4 },
        #'covered_buttocks': {'min_prob': 0.40, 'width_area_safety': 0.4 },
        'face_femme': {'censor_style': [ 'bar', (0,0,0) ], 'min_prob': 0.60, 'width_area_safety': 0.3, 'height_area_safety': 0.4, 'shape': 'circle' },
        #'exposed_belly': {'min_prob': 0.40, 'width_area_safety': 0.4 },
        #'covered_belly': {'min_prob': 0.40, 'width_area_safety': 0.4 },
        'exposed_feet': {'min_prob': 0.2, 'width_area_safety': -0.1, 'height_area_safety': 0 },
        'covered_feet': {'min_prob': 0.2, 'width_area_safety': -0.1, 'height_area_safety': 0 },
        #'exposed_armpits': {'min_prob': 0.40, 'width_area_safety': 0.4 },
        #'exposed_penis': {'min_prob': 0.40, 'width_area_safety': 0.4 },
        #'exposed_chest': {'min_prob': 0.40, 'width_area_safety': 0.4 },
        #'face_masc': {'min_prob': 0.40, 'width_area_safety': 0.4 },
        
}

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
#是否删除未打码的文件，0为不删除，1为删除。
input_delete_probability = 0

###################################################################
#### Miscellaneous

# If you have more than one graphics card and you care which one
# runs the neural net, change this
# You probably don't need to change this
#如有多个显卡，选择使用的显卡。
cuda_device_id = 0

# this determines how various censor styles deal with overlap
# You probably don't need to change this
#当多个打码部位重叠时，选择使用哪种方式。
censor_overlap_strategy = {
        'blur': 'single-pass',
        'bar': 'none',
        'pixel': 'single-pass',
        'debug': 'none',
        'line': 'single-pass',  # 添加line打码方式的重叠处理策略
        }

# this determines how pixel and blur censors 
# "scale" with image size.  'feature' is recommended
#像素化和模糊化的计算方式，推荐使用feature。
censor_scale_strategy = 'feature' # scales N->1 by min feature dimension, with a 100 base (so a 200x400 feature would be 2N->1)
#censor_scale_strategy = 'image' # scales N->1 by max image dimension, with 1000 base (so a 2000x1200 image would be 2N->1)
#censor_scale_strategy = 'none' # uses N -> 1 reduction

### Debug mode, if you are trying to solve issues
#是否开启调试模式。
debug_mode = 0
#debug_mode = 1 # just show debug information
#debug_mode = 3 # also save debug info to BetaSuite folder

### Enable BetaSuite Watermark
### I add a small watermark to the upper-left corner
### of censored content.  This is to make clear that the
### content has been altered, which I think is important.
### If you're sharing the content, I ask that you keep
### the watermark.  However, you can turn it off here,
### if you decide to.
#是否开启水印。
#enable_betasuite_watermark = True
enable_betasuite_watermark = False
#水印内容。
watermark_text = "made by telegram" #暂不支持中文
#水印颜色（bgr）
watermark_text_color = (0,0,0)
