import cv2
import numpy as np
import torch

def masked_image(img, mask, cls2color: dict = {1: (0, 255, 0), 2: (0, 0, 255)}):

    '''
    img: np.ndarray; shape (H, W); pixel_value: [0, 255]
    mask: np.ndarray in onehot format; shape (H, W, 3)
    cls2color: for example, {1: (0, 255, 0), 2: (0, 0, 255)}    (in BGR format)

    return:
        weighted image for visualization
    '''

    assert img.shape[:2] == mask.shape[:2], "Image and mask must be in same shape!"
    mask_cls = np.argmax(mask, axis=2)   # (H, W)

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.uint8)

    colored_mask1 = np.zeros_like(img_color)
    colored_mask2 = np.zeros_like(img_color)

    colored_mask1[mask_cls==1] = np.array(cls2color[1])  # green
    colored_mask2[mask_cls==2] = np.array(cls2color[2])  # red
        
    vis_img = cv2.addWeighted(img_color, 1, colored_mask1, 0.5, 0)
    vis_img = cv2.addWeighted(vis_img, 1, colored_mask2, 0.5, 0)

    return vis_img

def tensor2array(t: torch.Tensor):
    '''
    (C, H, W) -> (H, W, C)
    '''
    if t.device.type == "cuda":
        t_ = np.array(t.detach().cpu())
    else:
        t_ = np.array(t)
    
    if t_.ndim == 2:
        return t_
    elif t_.ndim == 3:
        return t_.transpose(1, 2, 0)
    else:
        raise NotImplementedError