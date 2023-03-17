import tensorflow as tf


def edge_map(img, enhance=True):

    '''
    :param img: -1 ~ 1
    :param enhance:
    :return:
    '''

    def high_pass_filter(img, d, n):
        '''
        :param img: 0 ～ 1
        :return:
        '''
        return (1 - 1 / (1 + (img / d) ** n))


    # 1, H, W, 3   -1 ~ 1

    v_kernel = tf.transpose(tf.constant(
        [[[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
         [[0.0, -2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
         [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], shape=[3, 3, 3, 1], dtype=tf.float32), perm=(1, 2, 0, 3))

    h_kernel = tf.transpose(tf.constant(
        [[[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [-2.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]], shape=[3, 3, 3, 1], dtype=tf.float32), perm=(1, 2, 0, 3))

    v_center_kernel = tf.transpose(tf.constant(
        [[[0.0, -1.0, 0.0], [0.0, 2.0, 0.0], [0.0, -1.0, 0.0]],
         [[0.0, -2.0, 0.0], [0.0, 4.0, 0.0], [0.0, -2.0, 0.0]],
         [[0.0, -1.0, 0.0], [0.0, 2.0, 0.0], [0.0, -1.0, 0.0]]], shape=[3, 3, 3, 1], dtype=tf.float32),
        perm=(1, 2, 0, 3))

    h_center_kernel = tf.transpose(tf.constant(
        [[[0.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [-2.0, 4.0, -2.0], [0.0, 0.0, 0.0]],
         [[0.0, 0.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 0.0, 0.0]]], shape=[3, 3, 3, 1], dtype=tf.float32),
        perm=(1, 2, 0, 3))

    kernel = tf.concat([v_kernel, h_kernel, v_center_kernel, h_center_kernel], axis=-1)
    edge = tf.reduce_sum(tf.abs(tf.nn.conv2d(
        tf.pad(img, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT'), kernel, [1, 1, 1, 1], padding='VALID')), axis=-1, keep_dims=True)

    # edge = edge**0.5

    # normalize to 0 ~ 1
    edge = (edge - tf.reduce_min(edge)) / (tf.reduce_max(edge) - tf.reduce_min(edge))

    if enhance:
        edge = high_pass_filter(edge, d=0.2, n=2)

    return edge




