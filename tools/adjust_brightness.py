from tools.utils import *
import numpy as np
import cv2


def read_img(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    assert len(img.shape) == 3
    return img


# Calculates the average brightness in the specified irregular image
def calculate_average_brightness(img):
    # Average value of three color channels
    R = img[..., 0].mean()
    G = img[..., 1].mean()
    B = img[..., 2].mean()

    brightness = 0.299 * R + 0.587 * G + 0.114 * B
    return brightness, B, G, R


# adjusting the average brightness of the target image to the average brightness of the source image
def adjust_brightness_from_src_to_dst(dst, src, path=None, if_show=None, if_info=None):
    brightness1, B1, G1, R1 = calculate_average_brightness(src)
    brightness2, B2, G2, R2 = calculate_average_brightness(dst)
    brightness_difference = brightness1 / brightness2

    if if_info:
        print('Average brightness of original image', brightness1)
        print('Average brightness of target', brightness2)
        print('Brightness Difference between Original Image and Target', brightness_difference)

    dstf = dst * brightness_difference
    dstf = np.clip(dstf, 0, 255)
    dstf = np.uint8(dstf)

    ma, na, _ = src.shape
    mb, nb, _ = dst.shape
    result_show_img = np.zeros((max(ma, mb), 3 * max(na, nb), 3))
    result_show_img[:mb, :nb, :] = dst
    result_show_img[:ma, nb:nb + na, :] = src
    result_show_img[:mb, nb + na:nb + na + nb, :] = dstf
    result_show_img = result_show_img.astype(np.uint8)

    if if_show:
        cv2.imshow('-', cv2.cvtColor(result_show_img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if path != None:
        cv2.imwrite(path, cv2.cvtColor(result_show_img, cv2.COLOR_BGR2RGB))

    return dstf






