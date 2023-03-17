import tensorflow.contrib as tf_contrib
import tensorflow as tf
import tensorflow.contrib.slim as slim


def layer_norm(x, scope='layer_norm'):
    return tf_contrib.layers.layer_norm(x, center=True, scale=True, scope=scope)


def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def Conv2D(inputs, filters, kernel_size=3, strides=1, scope='Conv2d'):
    if kernel_size == 3 and strides == 1:
        inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
    if kernel_size == 7 and strides == 1:
        inputs = tf.pad(inputs, [[0, 0], [3, 3], [3, 3], [0, 0]], mode="REFLECT")
    if kernel_size == 3 and strides == 2:
        inputs = tf.pad(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]], mode="REFLECT")
    return tf_contrib.layers.conv2d(
        inputs,
        num_outputs=filters,
        kernel_size=kernel_size,
        stride=strides,
        weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
        biases_initializer=None,
        normalizer_fn=None,
        activation_fn=None,
        padding='VALID',
        scope=scope)


def Conv2DNormLReLU(inputs, filters, kernel_size=3, strides=1, scope='conv_blk'):
    with tf.variable_scope(name_or_scope=scope):
        x = Conv2D(inputs, filters, kernel_size, strides)
        x = layer_norm(x)
        return lrelu(x)


def dwise_conv(input, k_h=3, k_w=3, channel_multiplier=1, name='dwise_conv'):
    input = tf.pad(input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
    with tf.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]
        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                            initializer=tf_contrib.layers.variance_scaling_initializer())
        conv = tf.nn.depthwise_conv2d(input=input, filter=w, strides=[1, 1, 1, 1], padding='VALID', name=name)
        return conv


def upsample(inputs, filters, scope='upsample'):
    '''
        An alternative to transposed convolution where we first resize, then convolve.
        See http://distill.pub/2016/deconv-checkerboard/
        For some reason the shape needs to be statically known for gradient propagation
        through tf.image.resize_images, but we only know that for fixed image size, so we
        plumb through a "training" argument
    '''
    new_H, new_W = 2 * tf.shape(inputs)[1], 2 * tf.shape(inputs)[2]
    inputs = tf.image.resize_images(inputs, [new_H, new_W])
    return Conv2DNormLReLU(inputs=inputs, filters=filters, scope=scope)


def InvertedRes_block(inputs, expansion_ratio, output_dim, scope='InvertedRes_block'):
    with tf.variable_scope(scope):
        bottleneck_dim = round(expansion_ratio * inputs.get_shape().as_list()[-1])
        net = Conv2DNormLReLU(inputs=inputs, filters=bottleneck_dim, kernel_size=1)

        net = dwise_conv(net)
        net = layer_norm(net, scope='layer_norm_1')
        net = lrelu(net)

        net = Conv2D(inputs=net, filters=output_dim, kernel_size=1)
        net = layer_norm(net, scope='layer_norm_2')

        if (int(inputs.get_shape().as_list()[-1]) == output_dim):
            net = inputs + net
        return net


def G_net(inputs):

    with tf.variable_scope('A'):
        x1 = Conv2DNormLReLU(inputs=inputs, filters=32, kernel_size=7, scope='conv_blk_1')              # 256，256，32
        x2 = Conv2DNormLReLU(inputs=x1, filters=64, kernel_size=3, strides=2, scope='conv_blk_2')   # 128，128，64
        x3 = Conv2DNormLReLU(inputs=x2, filters=64, kernel_size=3, strides=1, scope='conv_blk_3')   # 128，128，64

    with tf.variable_scope('B'):
        x4 = Conv2DNormLReLU(inputs=x3, filters=128, kernel_size=3, strides=2, scope='conv_blk_1')  # 64，64，128
        x5 = Conv2DNormLReLU(inputs=x4, filters=128, kernel_size=3, strides=1, scope='conv_blk_2')  # 64，64，128

    with tf.variable_scope('C'):
        x6 = Conv2DNormLReLU(inputs=x5, filters=128, kernel_size=3, strides=1, scope='conv_blk_1')  # 64，64，128
        x7 = InvertedRes_block(inputs=x6, expansion_ratio=2, output_dim=256, scope='r1')            # 64，64，256
        x8 = InvertedRes_block(inputs=x7, expansion_ratio=2, output_dim=256, scope='r2')            # 64，64，256
        x9 = InvertedRes_block(inputs=x8, expansion_ratio=2, output_dim=256, scope='r3')            # 64，64，256
        x10 = InvertedRes_block(inputs=x9, expansion_ratio=2, output_dim=256, scope='r4')            # 64，64，256
        x11 = Conv2DNormLReLU(inputs=x10, filters=128, kernel_size=3, strides=1, scope='conv_blk_2')  # 64，64，128

    with tf.variable_scope('D'):
        x12 = upsample(inputs=x11, filters=128, scope='upsample')                                     # 128，128，128

        x13 = Conv2DNormLReLU(inputs=x12, filters=128, scope='conv_blk')                              # 128，128，128

    with tf.variable_scope('E'):
        x14 = upsample(inputs=x13, filters=64, scope='upsample')                                      # 256，256，64
        x15 = Conv2DNormLReLU(inputs=x14, filters=64, scope='conv_blk_1')                             # 256，256，64
        x16 = Conv2DNormLReLU(inputs=x15, filters=32, kernel_size=7, scope='conv_blk_2')              # 256，256，32

    with tf.variable_scope('out_layer'):
        out = Conv2D(inputs=x16, filters=3, kernel_size=1)                                               # 256，256，3
        fake = tf.tanh(out)

    return fake


def G_net_unet(inputs):

    with tf.variable_scope('A'):
        x1 = Conv2DNormLReLU(inputs=inputs, filters=32, kernel_size=7, scope='conv_blk_1')              # 256，256，32
        x2 = Conv2DNormLReLU(inputs=x1, filters=64, kernel_size=3, strides=2, scope='conv_blk_2')   # 128，128，64
        x3 = Conv2DNormLReLU(inputs=x2, filters=64, kernel_size=3, strides=1, scope='conv_blk_3')   # 128，128，64

    with tf.variable_scope('B'):
        x4 = Conv2DNormLReLU(inputs=x3, filters=128, kernel_size=3, strides=2, scope='conv_blk_1')  # 64，64，128
        x5 = Conv2DNormLReLU(inputs=x4, filters=128, kernel_size=3, strides=1, scope='conv_blk_2')  # 64，64，128

    with tf.variable_scope('C'):
        x6 = Conv2DNormLReLU(inputs=x5, filters=128, kernel_size=3, strides=1, scope='conv_blk_1')  # 64，64，128
        x7 = InvertedRes_block(inputs=x6, expansion_ratio=2, output_dim=256, scope='r1')            # 64，64，256
        x8 = InvertedRes_block(inputs=x7, expansion_ratio=2, output_dim=256, scope='r2')            # 64，64，256
        x9 = InvertedRes_block(inputs=x8, expansion_ratio=2, output_dim=256, scope='r3')            # 64，64，256
        x10 = InvertedRes_block(inputs=x9, expansion_ratio=2, output_dim=256, scope='r4')            # 64，64，256
        x11 = Conv2DNormLReLU(inputs=x10, filters=128, kernel_size=3, strides=1, scope='conv_blk_2')  # 64，64，128

    with tf.variable_scope('D'):
        x12 = upsample(inputs=x11, filters=128, scope='upsample')                                     # 128，128，128
        x12 = tf.concat([x12, x3], axis=-1)
        x13 = Conv2DNormLReLU(inputs=x12, filters=128, scope='conv_blk')                              # 128，128，128

    with tf.variable_scope('E'):
        x14 = upsample(inputs=x13, filters=64, scope='upsample')                                      # 256，256，64
        x14 = tf.concat([x14, x1], axis=-1)
        x15 = Conv2DNormLReLU(inputs=x14, filters=64, scope='conv_blk_1')                             # 256，256，64
        x16 = Conv2DNormLReLU(inputs=x15, filters=32, kernel_size=7, scope='conv_blk_2')              # 256，256，32

    with tf.variable_scope('out_layer'):
        out = Conv2D(inputs=x16, filters=3, kernel_size=1)                                               # 256，256，3
        fake = tf.tanh(out)

    return fake


def unet_generator(inputs, channel=32, num_blocks=4):

    x0 = slim.convolution2d(inputs, channel, [7, 7], activation_fn=None)
    x0 = tf.nn.leaky_relu(x0)

    x1 = slim.convolution2d(x0, channel, [3, 3], stride=2, activation_fn=None)
    x1 = tf.nn.leaky_relu(x1)
    x1 = slim.convolution2d(x1, channel * 2, [3, 3], activation_fn=None)
    x1 = tf.nn.leaky_relu(x1)

    x2 = slim.convolution2d(x1, channel * 2, [3, 3], stride=2, activation_fn=None)
    x2 = tf.nn.leaky_relu(x2)
    x2 = slim.convolution2d(x2, channel * 4, [3, 3], activation_fn=None)
    x2 = tf.nn.leaky_relu(x2)

    for idx in range(num_blocks):
        x2 = resblock(x2, out_channel=channel * 4, name='block_{}'.format(idx))

    x2 = slim.convolution2d(x2, channel * 2, [3, 3], activation_fn=None)
    x2 = tf.nn.leaky_relu(x2)

    h1, w1 = tf.shape(x2)[1], tf.shape(x2)[2]
    x3 = tf.image.resize_bilinear(x2, (h1 * 2, w1 * 2))
    x3 = slim.convolution2d(x3 + x1, channel * 2, [3, 3], activation_fn=None)
    x3 = tf.nn.leaky_relu(x3)
    x3 = slim.convolution2d(x3, channel, [3, 3], activation_fn=None)
    x3 = tf.nn.leaky_relu(x3)

    h2, w2 = tf.shape(x3)[1], tf.shape(x3)[2]
    x4 = tf.image.resize_bilinear(x3, (h2 * 2, w2 * 2))
    x4 = slim.convolution2d(x4 + x0, channel, [3, 3], activation_fn=None)
    x4 = tf.nn.leaky_relu(x4)
    x4 = slim.convolution2d(x4, 3, [7, 7], activation_fn=None)
    output = tf.tanh(x4)
    return output


def resblock(inputs, out_channel=32, name='resblock'):
    with tf.variable_scope(name):
        x = slim.convolution2d(inputs, out_channel, [3, 3],
                               activation_fn=None, scope='conv1')
        x = tf.nn.leaky_relu(x)
        x = slim.convolution2d(x, out_channel, [3, 3],
                               activation_fn=None, scope='conv2')

        return x + inputs