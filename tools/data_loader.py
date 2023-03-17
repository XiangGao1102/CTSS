import os
import tensorflow as tf
import cv2
import numpy as np


class ImageGenerator(object):

    # generator to feed model images


    def __init__(self, image_dir, batch_size, num_cpus=16):
        self.paths = self.get_image_paths_train(image_dir)
        self.num_images = len(self.paths)
        self.num_cpus = num_cpus
        self.batch_size = batch_size


    def get_image_paths_train(self, image_dir):

        image_dir = os.path.join(image_dir)

        paths = []

        if not os.path.exists(image_dir):
            return paths

        for path in os.listdir(image_dir):
            # Check extensions of filename
            if path.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'gif']:
                continue

            # Construct complete path to anime image
            path_full = os.path.join(image_dir, path)

            # Validate if colorized image exists
            if not os.path.isfile(path_full):
                continue

            paths.append(path_full)

        return paths


    def read_image(self, img_path):
        img = cv2.imread(img_path.decode()).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)              # float32   0 ~ 255
        return img


    def load_image(self, img_path):
        img = self.read_image(img_path)    #  0 ~ 255
        img = img / 127.5 - 1.0            # -1 ~ 1 RGB
        return img                         # -1 ~ 1


    def load_images(self):

        dataset = tf.data.Dataset.from_tensor_slices(self.paths)

        # Repeat indefinitely
        dataset = dataset.repeat()

        # Uniform shuffle
        dataset = dataset.shuffle(buffer_size=len(self.paths))

        # Map path to image 
        dataset = dataset.map(lambda img: tf.py_func(
            self.load_image, [img], tf.float32), self.num_cpus)

        dataset = dataset.batch(self.batch_size)

        img = dataset.make_one_shot_iterator().get_next()

        return img


class PairedImageGenerator(object):

    # generator to feed model image and blurred image pairs

    def __init__(self, image_dir, batch_size, num_cpus=16):
        self.paths = self.get_image_paths_train(image_dir)
        self.num_images = len(self.paths)
        self.num_cpus = num_cpus
        self.batch_size = batch_size


    def get_image_paths_train(self, image_dir):

        image_dir = os.path.join(image_dir)

        paths = []

        for path in os.listdir(image_dir):
            # Check extensions of filename
            if path.split('.')[-1] not in ['jpg', 'jpeg', 'png', 'gif']:
                continue

            # Construct complete path to anime image
            path_full = os.path.join(image_dir, path)

            # Validate if colorized image exists
            if not os.path.isfile(path_full):
                continue

            paths.append(path_full)

        return paths


    def read_image(self, img_path):
        img = cv2.imread(img_path.decode()).astype(np.float32)
        blur = cv2.GaussianBlur(img, (9, 9), 0)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # float32   0 ~ 255
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)  # float32   0 ~ 255

        return img, blur


    def load_image(self, img_path):
        img, blur = self.read_image(img_path)  # 0 ~ 255

        img = img / 127.5 - 1.0     # -1 ~ 1 RGB
        blur = blur / 127.5 - 1.0   # -1 ~ 1 RGB

        blur = blur + np.random.normal(0, 0.2, blur.shape)
        blur = np.clip(blur, -1., 1.).astype(np.float32)

        return img, blur            # -1 ~ 1


    def load_images(self):

        dataset = tf.data.Dataset.from_tensor_slices(self.paths)

        # Repeat indefinitely
        dataset = dataset.repeat()

        # Uniform shuffle
        dataset = dataset.shuffle(buffer_size=len(self.paths))

        # Map path to image
        dataset = dataset.map(lambda img: tf.py_func(
            self.load_image, [img], [tf.float32, tf.float32]), self.num_cpus)

        dataset = dataset.batch(self.batch_size)

        img, blur = dataset.make_one_shot_iterator().get_next()

        return img, blur
