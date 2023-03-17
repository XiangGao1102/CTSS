import tensorflow as tf
from tools import edge_extracter
from tools.guided_filter import guided_filter



def extract_patches(img, patch_size, stride):
    channel = img.get_shape().as_list()[-1]
    patches = tf.extract_image_patches(img, ksizes=[1, patch_size, patch_size, 1], strides=[1, stride, stride, 1],
            rates=[1, 1, 1, 1], padding='VALID')     # (batch_size, h_patch_num, w_patch_num, patch_size x patch_size x 3)
    b, h_patch_num, w_patch_num, elements = patches.get_shape().as_list()
    patches = tf.reshape(patches, shape=[b * h_patch_num * w_patch_num, patch_size, patch_size, channel])
    return patches



def extract_top_k_img_patches_by_sum(img, patch_size, stride, k):
    '''
    :param img: input image of shape b, h, w, 3  -1 ~ 1 float32
    :param patch_size: the size of each extracted patch
    :param stride: the stride to slide window on original image
    :param k: the number of patches to extract from img
    :return: image patches with shape  k, patch_size, patch_size, 3  -1 ~ 1
    '''
    img_blur = guided_filter(img, img, 5, 0.2)
    edge_map = edge_extracter.edge_map(img_blur, enhance=True)   # 0 ~ 1    b, h, w, 1
    img_edge = tf.concat([img, edge_map], axis=-1)               # b, h, w, 4
    img_edge_patches = extract_patches(img_edge, patch_size, stride)
    print(img_edge_patches.get_shape())
    edge_patches = img_edge_patches[..., -1]
    img_patches = img_edge_patches[..., 0:3]
    edge_intensity_list = tf.reduce_sum(edge_patches, axis=[1, 2])
    top_k = tf.nn.top_k(edge_intensity_list, k=k, sorted=True, name='top_k')[1]
    top_k_img_patches = tf.gather(img_patches, top_k)            # k, patch_size, patch_size, 3
    return top_k_img_patches



def extract_top_k_img_patches_by_max(img, patch_size, stride, k):
    '''
    :param img: input image of shape b, h, w, 3  -1 ~ 1 float32
    :param patch_size: the size of each extracted patch
    :param stride: the stride to slide window on original image
    :param k: the number of patches to extract from img
    :return: image patches with shape  k, patch_size, patch_size, 3  -1 ~ 1
    '''
    edge_map = edge_extracter.edge_map(img, enhance=True)      # 0 ~ 1    b, h, w, 1
    img_edge = tf.concat([img, edge_map], axis=-1)             # b, h, w, 4
    img_edge_patches = extract_patches(img_edge, patch_size, stride)
    edge_patches = img_edge_patches[..., -1]
    img_patches = img_edge_patches[..., 0:3]
    edge_intensity_list = tf.reduce_max(edge_patches, axis=[1, 2])
    top_k = tf.nn.top_k(edge_intensity_list, k=k, sorted=True, name='top_k')[1]
    top_k_img_patches = tf.gather(img_patches, top_k)          # k, patch_size, patch_size, 3
    return top_k_img_patches



def extract_top_k_img_patches_with_edges(img, patch_size, stride, k):
    '''
    :param img: input image of shape b, h, w, 3  -1 ~ 1 float32
    :param patch_size: the size of each extracted patch
    :param stride: the stride to slide window on original image
    :param k: the number of patches to extract from img
    :return: image patches with shape  k, patch_size, patch_size, 3  -1 ~ 1
    '''
    edge_map = edge_extracter.edge_map(img, enhance=True)       # 0 ~ 1    b, h, w, 1
    img_edge = tf.concat([img, edge_map], axis=-1)                         # b, h, w, 4
    img_edge_patches = extract_patches(img_edge, patch_size, stride)
    edge_patches = img_edge_patches[..., -1]
    img_patches = img_edge_patches[..., 0:3]
    edge_intensity_list = tf.reduce_sum(edge_patches, axis=[1, 2])
    top_k = tf.nn.top_k(edge_intensity_list, k=k, sorted=True, name='top_k')[1]
    top_k_img_patches = tf.gather(img_patches, top_k)                     # k, patch_size, patch_size, 3
    top_k_edge_patches = tf.gather(edge_patches, top_k)
    return top_k_img_patches, top_k_edge_patches

