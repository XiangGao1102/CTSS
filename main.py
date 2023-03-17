from model import AnimeStyle
import argparse
from tools.utils import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


"""parsing and configuration"""

def parse_args():

    desc = "AnimeStyle"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--phase', type=str, default='test', help='train or test?')
    parser.add_argument('--dataset', type=str, default='TWR', help='dataset name')
    parser.add_argument('--g_adv_weight', type=float, default=300.0, help='weight of adversarial loss for generator')
    parser.add_argument('--d_adv_weight', type=float, default=300.0, help='weight of adversarial loss for discriminator')
    parser.add_argument('--con_weight', type=float, default=1.5, help='weight of content loss')
    parser.add_argument('--color_weight', type=float, default=15., help='weight of color reconstruction loss')
    parser.add_argument('--tv_weight', type=float, default=1.0, help='weight of total variation loss')


    parser.add_argument('--epoch', type=int, default=80, help='number of training epochs')
    parser.add_argument('--init_epoch', type=int, default=10, help='number of training epochs in initialization stage')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--save_freq', type=int, default=2, help='number of training epochs after every which model checkpoint is saved')
    parser.add_argument('--init_lr', type=float, default=2e-4, help='learning rate at initialization stage')
    parser.add_argument('--g_lr', type=float, default=2e-5, help='initial learning rate of the generator')
    parser.add_argument('--d_lr', type=float, default=1e-5, help='initial learning rate of the discriminator')
    parser.add_argument('--img_size', type=list, default=[256, 256], help='size of input image')
    parser.add_argument('--img_ch', type=int, default=3, help='number of image channel')
    parser.add_argument('--sn', type=str2bool, default=True, help='whether to use spectral norm')
    parser.add_argument('--val_freq', type=int, default=1, help='number of training epochs after every which validation is performed')


    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Name of checkpoint directory')
    parser.add_argument('--init_checkpoint_dir', type=str, default='init_checkpoint',
                        help='Name of initial checkpoint directory')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Name of directory to save generated test results')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Name of directory to save validation results during training')


    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    if args.phase == 'test':
        # --result_dir
        check_folder(args.result_dir)
    else:
        check_folder(args.init_checkpoint_dir)
        check_folder(args.sample_dir)

    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # open session
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, inter_op_parallelism_threads=8,
                               intra_op_parallelism_threads=8, gpu_options=gpu_options)) as sess:

        model = AnimeStyle(sess, args)

        # build graph
        model.build_model()

        # show network architecture
        show_all_variables()

        if args.phase == 'train':
            model.train()
            print(" [*] Training finished!")

        if args.phase == 'test':
            model.test()
            print(" [*] Test finished!")


if __name__ == '__main__':
    main()
