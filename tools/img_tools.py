from PIL import Image
import numpy as np



def adjust_contrast(img, contrast=0):
    # img: 0 ~ 255
    # value: -100 ~ 100

    img = img * 1.0
    thre = img.mean()
    img_out = img * 1.0
    if contrast <= -255.0:
        img_out = (img_out >= 0) + thre - 1
    elif contrast > -255.0 and contrast <= 0:
        img_out = img + (img - thre) * contrast / 255.0
    elif contrast <= 255.0 and contrast > 0:
        new_con = 255.0 * 255.0 / (256.0 - contrast) - 255.0
        img_out = img + (img - thre) * new_con / 255.0
    else:
        mask_1 = img > thre
        img_out = mask_1 * 255.0

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1
    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2

    img_out = img_out * 255.

    return img_out



def adjust_luminance(img, increment=0):

    # img: 0 ~ 255

    img = img * 1.0
    I = (img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3.0 + 0.001
    mask_1 = I > 128.0
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    rhs = (r * 128.0 - (I - 128.0) * 256.0) / (256.0 - I)
    ghs = (g * 128.0 - (I - 128.0) * 256.0) / (256.0 - I)
    bhs = (b * 128.0 - (I - 128.0) * 256.0) / (256.0 - I)
    rhs = rhs * mask_1 + (r * 128.0 / I) * (1 - mask_1)
    ghs = ghs * mask_1 + (g * 128.0 / I) * (1 - mask_1)
    bhs = bhs * mask_1 + (b * 128.0 / I) * (1 - mask_1)
    I_new = I + increment - 128.0
    mask_2 = I_new > 0.0
    R_new = rhs + (256.0 - rhs) * I_new / 128.0
    G_new = ghs + (256.0 - ghs) * I_new / 128.0
    B_new = bhs + (256.0 - bhs) * I_new / 128.0
    R_new = R_new * mask_2 + (rhs + rhs * I_new / 128.0) * (1 - mask_2)
    G_new = G_new * mask_2 + (ghs + ghs * I_new / 128.0) * (1 - mask_2)
    B_new = B_new * mask_2 + (bhs + bhs * I_new / 128.0) * (1 - mask_2)
    img_out = img * 1.0
    img_out[:, :, 0] = R_new
    img_out[:, :, 1] = G_new
    img_out[:, :, 2] = B_new
    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1
    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2

    img_out = img_out * 255.

    return img_out


def adjust_saturation(img, increment=0):
    # img: 0 ~ 255
    # increment: -1 ~ 1

    img = img * 1.0
    img_out = img * 1.0

    img_min = img.min(axis=2)
    img_max = img.max(axis=2)

    Delta = (img_max - img_min) / 255.0
    value = (img_max + img_min) / 255.0
    L = value / 2.0

    mask_1 = L < 0.5

    s1 = Delta / (value + 0.001)
    s2 = Delta / (2 - value + 0.001)
    s = s1 * mask_1 + s2 * (1 - mask_1)

    if increment >= 0:
        temp = increment + s
        mask_2 = temp > 1
        alpha_1 = s
        alpha_2 = s * 0 + 1 - increment
        alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)
        alpha = 1 / (alpha + 0.001) - 1
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha

    else:
        alpha = increment
        img_out[:, :, 0] = L * 255.0 + (img[:, :, 0] - L * 255.0) * (1 + alpha)
        img_out[:, :, 1] = L * 255.0 + (img[:, :, 1] - L * 255.0) * (1 + alpha)
        img_out[:, :, 2] = L * 255.0 + (img[:, :, 2] - L * 255.0) * (1 + alpha)

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2

    img_out = img_out * 255.

    return img_out