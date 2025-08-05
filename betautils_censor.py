import cv2
import math
import numpy as np

import betautils_config as bu_config
from betautils_model import get_eye_boxes

import betaconst
import betaconfig

betaconfig.enable_area_bar[1] = tuple( reversed( betaconfig.enable_area_bar[1] ) )

def censor_scale_for_image_box( image, feature_w, feature_h ):  
    if betaconfig.censor_scale_strategy == 'none':
        return(1)
    if betaconfig.censor_scale_strategy == 'feature':
        return( min( feature_w, feature_h ) / 100 )
    if betaconfig.censor_scale_strategy == 'image':
        (img_h,img_w,_) = image.shape
        return( max( img_h, img_w ) / 1000 )

def pixelate_image( image, x, y, w, h, factor, shape ): # factor 10 means 100x100 area becomes 10x10
    factor *= censor_scale_for_image_box( image, w, h )
    new_w = math.ceil(w/factor)
    new_h = math.ceil(h/factor)
    if shape == 'circle':
        image[y:y+h,x:x+w] = cv2.resize( cv2.resize( image[y:y+h,x:x+w], (max(new_w,new_h), max(new_w,new_h)), interpolation = cv2.BORDER_DEFAULT ), (max(w,h), max(w,h)), interpolation = cv2.INTER_NEAREST )
    else:
        image[y:y+h,x:x+w] = cv2.resize( cv2.resize( image[y:y+h,x:x+w], (new_w, new_h), interpolation = cv2.BORDER_DEFAULT ), (w,h), interpolation = cv2.INTER_NEAREST )
    return( image )

def blur_image( image, x, y, w, h, factor, shape ):
    factor = 2*math.ceil( factor * censor_scale_for_image_box( image, w, h )/2 ) + 1
    if shape == 'circle':
        #image[y:y+max(h,w),x:x+max(w,h)] = cv2.GaussianBlur( image[y:y+max(h,w),x:x+max(w,h)], (factor, factor), 0 )
        image[y:y+max(h,w),x:x+max(w,h)] = cv2.blur( image[y:y+max(h,w),x:x+max(w,h)], (factor, factor), cv2.BORDER_DEFAULT )
    else:
        image[y:y+h,x:x+w] = cv2.GaussianBlur( image[y:y+h,x:x+w], (factor, factor), 0 )
    return( image )

def bar_image( image, x, y, w, h, color, shape ):
    color = tuple( reversed( color ) )
    image = np.ascontiguousarray( image )
    if shape == 'rectangle':
        image = cv2.rectangle( image, (x,y), (x+w,y+h), color, cv2.FILLED )
    if shape == 'circle':
        image = cv2.circle( image, (x+w//2,y+h//2), max(w,h)//2, color, cv2.FILLED )
    return( image )

def line_image( image, x, y, w, h, factor, shape ):
    if shape == 'circle':
        roi = cv2.cvtColor(image[y:y+max(h,w),x:x+max(w,h)], cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        roi = np.sqrt(sobelx**2 + sobely**2)
        roi = cv2.convertScaleAbs(roi, factor)
        roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        image[y:y+max(h,w),x:x+max(w,h)] = roi
    else:   
        roi = cv2.cvtColor(image[y:y+h,x:x+w], cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        roi = np.sqrt(sobelx**2 + sobely**2)
        roi = cv2.convertScaleAbs(roi, factor)
        roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        image[y:y+h,x:x+w] = roi
    return( image )

def create_background_image(image, censor_style):
    """创建指定审查样式的背景图像"""
    h, w = image.shape[:2]
    if 'blur' == censor_style[0]:
        factor = 2*math.ceil( censor_style[1] * censor_scale_for_image_box( image, w, h )/2 ) + 1
        return cv2.GaussianBlur( image, (factor, factor), 0 )
    elif 'pixel' == censor_style[0]:
        factor = censor_style[1] * censor_scale_for_image_box( image, w, h )
        new_w = math.ceil(w/factor)
        new_h = math.ceil(h/factor)
        return cv2.resize( cv2.resize( image, (new_w, new_h), interpolation = cv2.BORDER_DEFAULT ), (w,h), interpolation = cv2.INTER_NEAREST )
    elif 'bar' == censor_style[0]:
        background_color = tuple(reversed(censor_style[1]))
        return np.full_like(image, background_color, dtype=np.uint8)
    elif 'line' == censor_style[0]:
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(imgray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(imgray, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = cv2.convertScaleAbs(sobel, 8)
        sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
    return image.copy()
    
def debug_image( image, box ):
    x = box['x']
    y = box['y']
    w = box['w']
    h = box['h']
    color = tuple( reversed( box['censor_style'][1] ) )
    image = np.ascontiguousarray( image )
    image = cv2.rectangle( image, (x,y), (x+w,y+h), color, 3 )
    image = cv2.putText( image, '(%d,%d)'%(x,y),     (x+10,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 )
    image = cv2.putText( image, '(%d,%d)'%(x+w,y+h), (x+10,y+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 )
    image = cv2.putText( image, box['label'],        (x+10,y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 )
    image = cv2.putText( image, '%.2f %.1f %.1f'%(box['score'],box['start'],box['end'] ), (x+10, y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1 )
    return( image )

def censor_image( image, box ):
        if 'blur' == box['censor_style'][0]:
            return( blur_image( image, box['x'], box['y'], box['w'], box['h'], box['censor_style'][1], box['shape'] ) )
        if 'pixel' == box['censor_style'][0]:
            return( pixelate_image( image, box['x'], box['y'], box['w'], box['h'], box['censor_style'][1], box['shape'] ) )
        if 'bar' == box['censor_style'][0]:
            return( bar_image( image, box['x'], box['y'], box['w'], box['h'], box['censor_style'][1], box['shape'] ) )
        if 'line' == box['censor_style'][0]:
            return( line_image( image, box['x'], box['y'], box['w'], box['h'], box['censor_style'][1], box['shape'] ) )
        if 'debug' == box['censor_style'][0]:
            return( debug_image( image, box ) )  

def watermark_image( image ):
    if betaconfig.enable_betasuite_watermark:
        image = np.ascontiguousarray( image )
        (h,w,_) = image.shape
        scale = max( min(w/750,h/750), 1 )
        return( cv2.putText( image, betaconfig.watermark_text, (20,math.ceil(20*scale)), cv2.FONT_HERSHEY_PLAIN, scale, betaconfig.watermark_text_color, math.floor(scale) ) )
    else:
        return( image )

def process_raw_box( raw, vid_w, vid_h ):
    parts_to_blur = bu_config.get_parts_to_blur()
    label = betaconst.classes[int(raw['class_id'])][0]
    if label in parts_to_blur and raw['score'] > parts_to_blur[label]['min_prob']:
        x_area_safety = parts_to_blur[label]['width_area_safety']
        y_area_safety = parts_to_blur[label]['height_area_safety']
        time_safety = parts_to_blur[label]['time_safety']
        safe_x = math.floor( max( 0, raw['x'] - raw['w']*x_area_safety/2 ) )
        safe_y = math.floor( max( 0, raw['y'] - raw['h']*y_area_safety/2 ) )
        safe_w = math.ceil( min( vid_w-safe_x, raw['w']*(1+x_area_safety) ) )
        safe_h = math.ceil( min( vid_h-safe_y, raw['h']*(1+y_area_safety) ) )
        return( {
            "start": max( raw['t']-time_safety/2,0 ),
            "end":   raw['t']+time_safety/2,
            "x": safe_x, 
            "y": safe_y, 
            "w": safe_w, 
            "h": safe_h ,
            'censor_style': parts_to_blur[label]['censor_style'],
            'label': label,
            'score': raw['score'],
            'shape': parts_to_blur[label]['shape'],
        } )
    
def rectangles_intersect( box1, box2 ):
    if box1['x']+box1['w'] < box2['x']:
        return( False )

    if box1['y']+box1['h'] < box2['y']:
        return( False )

    if box1['x'] > box2['x']+box2['w']:
        return( False )

    if box1['y'] > box2['y']+box2['h']:
        return( False )

    return( True )

def censor_style_sort( censor_style ):
    if censor_style[0] == 'blur':
        return( 1 + 1/censor_style[1] )
    if censor_style[0] == 'bar':
        return( 2 + 1/(2+255*3-sum(censor_style[1])))
    if censor_style[0] == 'pixel':
        return( 3 + censor_style[1] )
    if censor_style[0] == 'debug':
        return( 99 )
            
def collapse_boxes_for_style( piece ):
    style = piece[0]['censor_style'][0]
    strategy = betaconfig.censor_overlap_strategy[style]
    if strategy == 'none':
        return( piece )
    
    if strategy == 'single-pass':
        segments = []
        for box in piece:
            found = False
            for i,segment in enumerate(segments):
                if rectangles_intersect( box, segment ):
                    x = min( segments[i]['x'], box['x'] )
                    y = min( segments[i]['y'], box['y'] )
                    w = max( segments[i]['x']+segments[i]['w'], box['x']+box['w'] ) - x
                    h = max( segments[i]['y']+segments[i]['h'], box['y']+box['h'] ) - y
                    segments[i]['x']=x
                    segments[i]['y']=y
                    segments[i]['w']=w
                    segments[i]['h']=h
                    found = True
                    break
            if not found:
                segments.append( box )

    return( segments )

def censor_img_for_boxes( image, boxes ):
    reverse_mode = betaconfig.reverse_mode
    # 如果是反向模式，创建黑色蒙版
    if reverse_mode == 0:
        # 创建与原图相同大小的白色蒙版
        mask = np.ones_like(image)
        # 获取默认的条形颜色作为白色区域颜色
        black_color = tuple( reversed( (0,0,0) ) )
        # 创建背景图（原图）
        background = image.copy()
    # 如果是反向模式，创建黑色蒙版
    if reverse_mode == 1:
        # 创建与原图相同大小的黑色蒙版
        mask = np.zeros_like(image)
        # 获取默认的条形颜色作为白色区域颜色
        white_color = tuple( reversed( (255,255,255) ) )
        # 创建背景图（使用默认条形颜色）
        background = create_background_image(image, betaconfig.default_censor_style)
    
    boxes.sort( key=lambda x: ( x['label'], censor_style_sort(x['censor_style']) ) )
    pieces = []
    for box in boxes:
        if len( pieces ) and pieces[-1][0]['label'] == box['label'] and pieces[-1][0]['censor_style']==box['censor_style']:
            pieces[-1].append(box)
        else:
            pieces.append([box])

    for piece in pieces:
        collapsed_boxes = collapse_boxes_for_style( piece )
        for collapsed_box in collapsed_boxes:
            if reverse_mode == 0:
                # 正常模式：直接处理图像
                image = censor_image( image, collapsed_box )
                if collapsed_box['shape'] == 'rectangle':
                    mask = cv2.rectangle(mask, (collapsed_box['x'], collapsed_box['y']), (collapsed_box['x']+collapsed_box['w'], collapsed_box['y']+collapsed_box['h']), black_color, cv2.FILLED)
                if collapsed_box['shape'] == 'circle':
                    mask = cv2.circle(mask, (collapsed_box['x']+collapsed_box['w']//2, collapsed_box['y']+collapsed_box['h']//2), max(collapsed_box['w'],collapsed_box['h'])//2, black_color, cv2.FILLED)
            
            if reverse_mode == 1:
                # 反向模式：在蒙版上绘制白色区域
                if collapsed_box['shape'] == 'rectangle':
                    mask = cv2.rectangle(mask, (collapsed_box['x'], collapsed_box['y']), (collapsed_box['x']+collapsed_box['w'], collapsed_box['y']+collapsed_box['h']), white_color, cv2.FILLED)
                if collapsed_box['shape'] == 'circle':
                    mask = cv2.circle(mask, (collapsed_box['x']+collapsed_box['w']//2, collapsed_box['y']+collapsed_box['h']//2), max(collapsed_box['w'],collapsed_box['h'])//2, white_color, cv2.FILLED)

    if reverse_mode == 0:
        image = np.where(mask == black_color, image, background)
    # 反向模式处理：将蒙版区域从原图提取出来
    if reverse_mode == 1:
        # 将蒙版中的白色区域对应的原图内容复制到背景图
        image = np.where(mask == white_color, image, background)
    
    # 统一绘制边框
    if 'on' == betaconfig.enable_area_bar[0]:
        # 修复线宽计算顺序，确保至少为1px
        w_bar = max(1, (min(image.shape[0], image.shape[1]) // 1000) * betaconfig.enable_area_bar[2])
        for piece in pieces:  # 遍历所有piece确保完整绘制
            collapsed_boxes = collapse_boxes_for_style(piece)
            for collapsed_box in collapsed_boxes:
                image = draw_border(image, collapsed_box, w_bar, betaconfig.enable_area_bar[1])

    if 'on' == betaconfig.eyes_bar[0]:
        # 获取眼睛位置（左眼、右眼）
        eye_positions = get_eye_boxes()
        for (lx, ly), (rx, ry) in eye_positions:
            # 计算两眼距离
            distance_between_eyes = int(math.hypot(rx - lx, ry - ly)) * 1.32 * betaconfig.eyes_bar[2]
            # 宽度按比例缩放，至少为1
            w_bar = int(max(1, distance_between_eyes // 2.8))
            # 计算中点
            cx = (lx + rx) / 2
            cy = (ly + ry) / 2
            # 计算旋转角度（单位：度）
            angle = math.degrees(math.atan2(ry - ly, rx - lx))
            # 构造旋转矩形（中心坐标、尺寸、角度）
            rotated_rect = ((cx, cy), (distance_between_eyes, w_bar), angle)
            # 获取矩形四个角点
            box = cv2.boxPoints(rotated_rect)
            box = box.astype(np.intp)
            # 绘制矩形
            image = cv2.drawContours(image, [box], 0, betaconfig.eyes_bar[1], -1)  # -1 填充
    
    image = watermark_image( image )

    return( image )

# 绘制边框辅助函数
def draw_border(image, box, w_bar, color):
    if box['shape'] == 'rectangle':
        return cv2.rectangle(image, (box['x'], box['y']), (box['x']+box['w'], box['y']+box['h']), color, w_bar)
    elif box['shape'] == 'circle':
        center = (box['x']+box['w']//2, box['y']+box['h']//2)
        radius = max(box['w'], box['h'])//2
        return cv2.circle(image, center, radius, color, w_bar)
    return ( image )
