import cv2
import logging
import math
import numpy as np

import betaconst
import betaconfig

def censor_scale_for_image_box( image, feature_w, feature_h ):
    if betaconfig.censor_scale_strategy == 'none':
        return(1)
    if betaconfig.censor_scale_strategy == 'feature':
        return( min( feature_w, feature_h ) / 100 )
    if betaconfig.censor_scale_strategy == 'image':
        (img_h,img_w,_) = image.shape
        return( max( img_h, img_w ) / 1000 )

def pixelate_image( image, x, y, w, h, factor ): 
    factor = max(1, factor * censor_scale_for_image_box( image, w, h ))
    new_w = max(1, math.ceil(w/factor))
    new_h = max(1, math.ceil(h/factor))
    image[y:y+h,x:x+w] = cv2.resize( cv2.resize( image[y:y+h,x:x+w], (new_w, new_h), interpolation = cv2.INTER_LANCZOS4 ), (w,h), interpolation = cv2.INTER_NEAREST ) 
    return( image )

def blur_image( image, x, y, w, h, factor ):
    factor = 2*math.ceil( factor * censor_scale_for_image_box( image, w, h )/2 ) + 1
    image[y:y+h,x:x+w] = cv2.blur( image[y:y+h,x:x+w], (factor, factor), borderType=cv2.BORDER_DEFAULT )
    return( image )

def bar_image( image, x, y, w, h, color ):
    color = tuple( reversed( color ) )
    image = np.ascontiguousarray( image )
    image = cv2.rectangle( image, (x,y), (x+w,y+h), color, cv2.FILLED )
    return( image )

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
    
    eyes = box.get('eyes')
    if eyes is not None and isinstance(eyes, list) and len(eyes) == 2:
        left_eye, right_eye = eyes[0], eyes[1]
        if left_eye is not None and len(left_eye) == 2:
            lx, ly = int(left_eye[0]), int(left_eye[1])
            image = cv2.circle(image, (lx, ly), 6, color, -1)
            image = cv2.putText(image, 'L', (lx-5, ly-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        if right_eye is not None and len(right_eye) == 2:
            rx, ry = int(right_eye[0]), int(right_eye[1])
            image = cv2.circle(image, (rx, ry), 6, color, -1)
            image = cv2.putText(image, 'R', (rx-5, ry-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
    return( image )

def censor_image( image, box ):
    shape_style = box['shape_style']
    
    if shape_style == 'circle':
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        if w <= 0 or h <= 0:
            return image
            
        roi = image[y:y+h, x:x+w].copy()
        
        if 'blur' == box['censor_style'][0]:
            factor = 2*math.ceil( box['censor_style'][1] * censor_scale_for_image_box( image, w, h )/2 ) + 1
            roi_censored = cv2.blur( roi, (factor, factor), borderType=cv2.BORDER_DEFAULT )
        elif 'pixel' == box['censor_style'][0]:
            factor = max(1, box['censor_style'][1] * censor_scale_for_image_box( image, w, h ))
            new_w = max(1, math.ceil(w/factor))
            new_h = max(1, math.ceil(h/factor))
            roi_censored = cv2.resize( cv2.resize( roi, (new_w, new_h), interpolation = cv2.INTER_LANCZOS4 ), (w,h), interpolation = cv2.INTER_NEAREST )
        elif 'bar' == box['censor_style'][0]:
            color = tuple( reversed( box['censor_style'][1] ) )
            roi_censored = np.zeros_like(roi)
            roi_censored[:] = color
        elif 'debug' == box['censor_style'][0]:
            return debug_image( image, box )
        else:
            return image
            
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
        mask_inv = cv2.bitwise_not(mask)
        
        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        img_fg = cv2.bitwise_and(roi_censored, roi_censored, mask=mask)
        image[y:y+h, x:x+w] = cv2.add(img_bg, img_fg)
        return image
    else:
        if 'blur' == box['censor_style'][0]:
            return( blur_image( image, box['x'], box['y'], box['w'], box['h'], box['censor_style'][1] ) )
        if 'pixel' == box['censor_style'][0]:
            return( pixelate_image( image, box['x'], box['y'], box['w'], box['h'], box['censor_style'][1] ) )
        if 'bar' == box['censor_style'][0]:
            return( bar_image( image, box['x'], box['y'], box['w'], box['h'], box['censor_style'][1] ) )
        if 'debug' == box['censor_style'][0]:
            return( debug_image( image, box ) )

def watermark_image( image ):
    if betaconfig.enable_betasuite_watermark:
        image = np.ascontiguousarray( image )
        (h,w,_) = image.shape
        scale = max( min(w/750,h/750), 1 )
        return( cv2.putText( image, betaconfig.watermark_text, (20,math.ceil(20*scale)), cv2.FONT_HERSHEY_PLAIN, scale, tuple(reversed(betaconfig.watermark_text_color)), math.floor(scale) ) )
    else:
        return( image )

def process_raw_box( raw, vid_w, vid_h, parts_to_blur ):
    label = betaconst.classes[int(raw['class_id'])][0]
    if betaconfig.debug_mode == 1:
        logging.debug('Detected region: %s', label)
    if label in parts_to_blur and raw['score'] > parts_to_blur[label]['min_prob']:
        x_area_safety = parts_to_blur[label]['width_area_safety']
        y_area_safety = parts_to_blur[label]['height_area_safety']
        time_safety = parts_to_blur[label]['time_safety']
        safe_x = max(0, math.floor(raw['x'] - raw['w'] * x_area_safety / 2))
        safe_y = max(0, math.floor(raw['y'] - raw['h'] * y_area_safety / 2))
        right = min(vid_w, math.ceil(raw['x'] + raw['w'] * (1 + x_area_safety / 2)))
        bottom = min(vid_h, math.ceil(raw['y'] + raw['h'] * (1 + y_area_safety / 2)))
        safe_w, safe_h = right - safe_x, bottom - safe_y
        if safe_w <= 0 or safe_h <= 0:
            return None
        return( {
            "start": max( raw['t']-time_safety/2,0 ),
            "end":   raw['t']+time_safety/2,
            "x": safe_x, 
            "y": safe_y, 
            "w": safe_w, 
            "h": safe_h ,
            'censor_style': parts_to_blur[label]['censor_style'],
            'shape_style': parts_to_blur[label]['shape_style'], 
            'label': label,
            'score': raw['score'],
            'eyes': raw.get('eyes')
        } )

def compute_overlap_rate(box1, box2):
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x'] + box1['w'], box2['x'] + box2['w'])
    y2 = min(box1['y'] + box1['h'], box2['y'] + box2['h'])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    inter_area = (x2 - x1) * (y2 - y1)
    area1 = box1['w'] * box1['h']
    area2 = box2['w'] * box2['h']
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

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
            for i, segment in enumerate(segments):
                if compute_overlap_rate(box, segment) >= betaconfig.censor_overlap_rate:
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
                segments.append( box.copy() )

    return( segments )

def apply_eyes_bar(image, eyes, color):
    left_eye, right_eye = eyes[0], eyes[1]
    if left_eye is None or right_eye is None:
        return image
    lx, ly = left_eye
    rx, ry = right_eye
    cx = (lx + rx) / 2.0
    cy = (ly + ry) / 2.0
    dx = rx - lx
    dy = ry - ly
    D = math.hypot(dx, dy)
    if D < 1:
        return image
    extend_ratio = 0.2
    thickness_ratio = 0.3
    W = D * (1.0 + 2.0 * extend_ratio)
    H = D * thickness_ratio
    ux = dx / D
    uy = dy / D
    vx = -uy
    vy = ux
    hw = W / 2.0
    hh = H / 2.0
    p1 = (int(cx - hw * ux + hh * vx), int(cy - hw * uy + hh * vy))
    p2 = (int(cx + hw * ux + hh * vx), int(cy + hw * uy + hh * vy))
    p3 = (int(cx + hw * ux - hh * vx), int(cy + hw * uy - hh * vy))
    p4 = (int(cx - hw * ux - hh * vx), int(cy - hw * uy - hh * vy))
    pts = np.array([p1, p2, p3, p4], dtype=np.int32)
    color_bgr = tuple(reversed(color))
    image = np.ascontiguousarray(image)
    cv2.fillPoly(image, [pts], color_bgr)
    return image

# =====================================================================
# 【新增】绘制区域边框 (Area Bar)
# =====================================================================
def apply_area_bars(image, boxes, config):
    status, color_rgb, coeff = config
    if status != 'on':
        return image

    img_h, img_w = image.shape[:2]
    
    # 动态线宽：以 1080P 为基准等比例缩放
    t = max(1, round(coeff * (img_h / 1080.0)))
    offset = max(0, (t - 1) // 2)

    color_bgr = tuple(reversed(color_rgb))
    image = np.ascontiguousarray(image)

    for box in boxes:
        # 【新增屏蔽逻辑】如果是 debug 模式，跳过画区域边界框
        if box.get('censor_style', [''])[0] == 'debug':
            continue

        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        if w <= 0 or h <= 0:
            continue

        shape = box.get('shape_style', 'rect')

        if shape == 'circle':
            cx, cy = x + w // 2, y + h // 2
            rx, ry = w // 2 + offset, h // 2 + offset
            cv2.ellipse(image, (cx, cy), (rx, ry), 0, 0, 360, color_bgr, t)
        else:
            pt1 = (x - offset, y - offset)
            pt2 = (x + w + offset, y + h + offset)
            cv2.rectangle(image, pt1, pt2, color_bgr, t)

    return image


def restore_regions(image, background, boxes):
    height, width = image.shape[:2]
    for box in boxes:
        x, y, w, h = box['x'], box['y'], box['w'], box['h']
        if w <= 0 or h <= 0 or box.get('censor_style', [''])[0] == 'debug':
            continue
        left, top = max(0, x), max(0, y)
        right, bottom = min(width, x + w + 1), min(height, y + h + 1)
        if left >= right or top >= bottom:
            continue
        source = image[top:bottom, left:right]
        destination = background[top:bottom, left:right]
        if box.get('shape_style') == 'circle':
            mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
            center = (x + w // 2 - left, y + h // 2 - top)
            cv2.ellipse(mask, center, (w // 2, h // 2), 0, 0, 360, 255, -1)
            cv2.copyTo(source, mask, destination)
        else:
            destination[:] = source


def censor_img_for_boxes( image, boxes ):
    # =====================================================================
    # 【最高优先级拦截】Debug 模式直通车
    # 只要触发 Debug（全局 debug_mode=1，或所有框均为 debug 样式）
    # 直接绕过所有打码逻辑（含反转模式的背景涂黑/模糊），仅执行标注和水印
    # =====================================================================
    is_global_debug = betaconfig.debug_mode & 1
    
    # 检查是否当前处理的全部是 debug 框
    is_all_boxes_debug = len(boxes) > 0 and all(box.get('censor_style', [''])[0] == 'debug' for box in boxes)

    if is_global_debug or is_all_boxes_debug:
        for box in boxes:
            style_name = box.get('censor_style', [''])[0]
            if style_name == 'debug':
                image = debug_image(image, box)
            else:
                # 兼容处理：全局 debug 模式下，哪怕原本是 blur 的框，也强制转为绿框标注，防止颜色元组报错
                temp_box = box.copy()
                temp_box['censor_style'] = ['debug', (0, 255, 0)]
                image = debug_image(image, temp_box)
                
        return watermark_image(image)

    # =====================================================================
    # 【新特性】一键反转模式（reverse_mode == 1）
    # =====================================================================
    if getattr(betaconfig, 'reverse_mode', 0) == 1:
        (img_h, img_w, _) = image.shape
        
        # 1. 动态放大全图系数，创建全图打码背景层
        style = betaconfig.default_censor_style[0]
        param = betaconfig.default_censor_style[1]
        
        if style == 'blur':
            global_factor = 2 * math.ceil(param * censor_scale_for_image_box(image, img_w, img_h) / 2) + 1
            img_censored = cv2.blur(image, (global_factor, global_factor), borderType=cv2.BORDER_DEFAULT)
        elif style == 'pixel':
            global_factor = max(1, param, param * censor_scale_for_image_box(image, img_w, img_h))
            new_w = max(1, math.ceil(img_w / global_factor))
            new_h = max(1, math.ceil(img_h / global_factor))
            img_censored = cv2.resize(cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        elif style == 'bar':
            color = tuple(reversed(param))
            img_censored = np.zeros_like(image)
            img_censored[:] = color
        else:
            raise ValueError(f'Unsupported background style: {style}')
            
        restore_regions(image, img_censored, boxes)
        image = img_censored

    else:
        # =====================================================================
        # 正常局部打码模式逻辑
        # =====================================================================
        boxes.sort( key=lambda x: ( x['label'], censor_style_sort(x['censor_style']), x.get('shape_style', 'rect') ) )
        pieces = []
        for box in boxes:
            if (len( pieces ) and 
                pieces[-1][0]['label'] == box['label'] and 
                pieces[-1][0]['censor_style'] == box['censor_style'] and
                pieces[-1][0].get('shape_style', 'rect') == box.get('shape_style', 'rect')):
                pieces[-1].append(box)
            else:
                pieces.append([box])

        for piece in pieces:
            collapsed_boxes = collapse_boxes_for_style( piece )
            for collapsed_box in collapsed_boxes:
                image = censor_image( image, collapsed_box )

    # =====================================================================
    # 公共后处理流：在所有涂层计算完毕后，统一置顶绘制附属 UI
    # =====================================================================
    if hasattr(betaconfig, 'eyes_bar') and betaconfig.eyes_bar[0] == 'on':
        bar_color = betaconfig.eyes_bar[1] if len(betaconfig.eyes_bar) > 1 else (0, 0, 0)
        for box in boxes:
            # 同样拦截 debug 模式画眼部粗条
            if box.get('censor_style', [''])[0] == 'debug':
                continue
                
            eyes = box.get('eyes')
            if eyes is not None and isinstance(eyes, list) and len(eyes) == 2:
                image = apply_eyes_bar(image, eyes, bar_color)

    # 同样拦截 debug 模式画区域外框
    if hasattr(betaconfig, 'enable_area_bar'):
        image = apply_area_bars(image, boxes, betaconfig.enable_area_bar)

    image = watermark_image( image )
    return( image )
