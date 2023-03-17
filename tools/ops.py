import tensorflow as tf
import tensorflow.contrib as tf_contrib


weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=3, stride=1, sn=True, scope='conv_0'):
    with tf.variable_scope(scope):

        if stride == 1:
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        if stride == 2:
            x = tf.pad(x, [[0, 0], [0, 1], [0, 1], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=False)
        return x


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

def sigmoid(x) :
    return tf.sigmoid(x)

##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)


def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)


def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)



def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


def L2_loss(x, y):
    size = tf.size(x)
    return tf.nn.l2_loss(x-y) * 2 / tf.to_float(size)


def Huber_loss(x, y):
    return tf.losses.huber_loss(x, y)


def discriminator_loss(anime, fake):
    real_loss = tf.reduce_mean(tf.square(anime - 1.0))
    fake_loss = tf.reduce_mean(tf.square(fake))
    loss = real_loss + fake_loss
    return loss


def generator_loss(fake):
    fake_loss = tf.reduce_mean(tf.square(fake - 1.0))
    return fake_loss


def gram(x):
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)  # b, c, c


def inverse_transform(img):
    '''
    :param img: Tensor with value in -1 ~ 1
    :return: Tensor with value in 0 ~ 255
    '''
    return (img + 1.) / 2. * 255.  # 0 ~ 255


def con_loss(vgg, real, fake):

    vgg.build(real)
    real_feature_map = vgg.conv4_4_no_activation

    vgg.build(fake)
    fake_feature_map = vgg.conv4_4_no_activation

    loss = L1_loss(real_feature_map, fake_feature_map)
    return loss


def color_loss(con, fake):
    con = rgb2yuv(con)
    fake = rgb2yuv(fake)

    return L1_loss(con[:,:,:,0], fake[:,:,:,0]) + Huber_loss(con[:,:,:,1], fake[:,:,:,1]) + Huber_loss(con[:,:,:,2], fake[:,:,:,2])


def total_variation_loss(inputs):
    """
    A smooth loss in fact. Like the smooth prior in MRF.
    V(y) = || y_{n+1} - y_n ||_2
    """
    dh = inputs[:, :-1, ...] - inputs[:, 1:, ...]
    dw = inputs[:, :, :-1, ...] - inputs[:, :, 1:, ...]
    size_dh = tf.size(dh)
    size_dw = tf.size(dw)
    return tf.nn.l2_loss(dh) / tf.cast(size_dh, tf.float32) + tf.nn.l2_loss(dw) / tf.cast(size_dw, tf.float32)


def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    rgb: -1 ~ 1
    """
    rgb = (rgb + 1.0) / 2.0
    return tf.image.rgb_to_yuv(rgb)



