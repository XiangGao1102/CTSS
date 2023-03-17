from tools.ops import *

def D_net(x_init, sn):

    x = conv(x_init, 32, kernel=3, stride=1, sn=sn, scope='conv_1')            # 256 x 256 x 32
    x = layer_norm(x, scope='l_norm_1')
    x = lrelu(x, 0.2)

    x = conv(x, 64, kernel=3, stride=2, sn=sn, scope='conv_2')                 # 128 x 128 x 64
    x = layer_norm(x, scope='l_norm_2')
    x = lrelu(x, 0.2)

    x = conv(x, 128, kernel=3, stride=2, sn=sn, scope='conv_3')                # 64 x 64 x 128
    x = layer_norm(x, scope='l_norm_3')
    x = lrelu(x, 0.2)

    x = conv(x, 256, kernel=3, stride=2, sn=sn, scope='conv_4')                # 32 x 32 x 256
    x = layer_norm(x, scope='l_norm_4')
    x = lrelu(x, 0.2)

    D_logit = conv(x, channels=1, kernel=3, stride=1, sn=sn, scope='D_logit')  # 32 x 32 x 1

    return D_logit


def patch_D_net(x_init, sn):

    x = conv(x_init, 16, kernel=3, stride=1, sn=sn, scope='conv_1')   # 96 x 96 x 16
    x = layer_norm(x, scope='l_norm_1')
    x = lrelu(x, 0.2)

    x = conv(x, 32, kernel=3, stride=2, sn=sn, scope='conv_2')        # 48 x 48 x 32
    x = layer_norm(x, scope='l_norm_2')
    x = lrelu(x, 0.2)

    x = conv(x, 64, kernel=3, stride=2, sn=sn, scope='conv_3')        # 24 x 24 x 64
    x = layer_norm(x, scope='l_norm_3')
    x = lrelu(x, 0.2)

    x = conv(x, 128, kernel=3, stride=2, sn=sn, scope='conv_4')       # 12 x 12 x 128
    x = layer_norm(x, scope='l_norm_4')
    x = lrelu(x, 0.2)

    D_logit = conv(x, channels=1, kernel=3, stride=1, sn=sn, scope='D_logit')  # 12 x 12 x 1

    return D_logit

