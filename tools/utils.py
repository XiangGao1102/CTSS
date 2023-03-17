import tensorflow as tf
from tensorflow.contrib import slim
from tools.adjust_brightness import adjust_brightness_from_src_to_dst, read_img
import os, cv2
import numpy as np
from tools.img_tools import adjust_contrast, adjust_luminance, adjust_saturation


def load_test_data(image_path, size):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocessing(img, size)
    img = np.expand_dims(img, axis=0)
    return img


def preprocessing(img, size):
    h, w = img.shape[:2]
    if h <= size[0]:
        h = size[0]
    else:
        x = h % 32
        h = h - x

    if w <= size[1]:
        w = size[1]
    else:
        y = w % 32
        w = w - y
    # the cv2 resize func : dsize format is (W ,H)
    img = cv2.resize(img, (w, h))
    return img / 127.5 - 1.0   # -1 ~ 1



def save_images(images, dataset_name, image_path, photo_path=None):
    images = inverse_transform(images.squeeze())
    adjust_config = None
    if dataset_name == 'TWR':
        adjust_config = [0.6, 50, 10]
    elif dataset_name == 'CSC':
        adjust_config = [0.5, 50, 30]
    elif dataset_name == 'DB':
        adjust_config = [0.4, 20, 20]
    else:
        raise ValueError('invalid dataset name.')
    if photo_path:
        images = adjust_brightness_from_src_to_dst(images, read_img(photo_path))
        images = adjust_saturation(images, adjust_config[0])
        images = adjust_contrast(images, adjust_config[1])
        images = adjust_luminance(images, adjust_config[2])
        images = images.astype(np.uint8)
        return imsave(images, image_path)
    else:
        images = images.astype(np.uint8)
        return imsave(images, image_path)



def inverse_transform(images):
    images = (images + 1.) / 2 * 255   # -1 ~ 1 --> 0 ~ 255
    images = np.clip(images, 0, 255)
    return images


def imsave(images, path):
    return cv2.imwrite(path, cv2.cvtColor(images, cv2.COLOR_BGR2RGB))


def show_all_variables():
    print('G:')
    slim.model_analyzer.analyze_vars([var for var in tf.trainable_variables() if var.name.startswith('generator')], print_info=True)
    print('D:')
    slim.model_analyzer.analyze_vars([var for var in tf.trainable_variables() if var.name.startswith('discriminator')], print_info=True)


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in ('true')


