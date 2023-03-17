from tools.ops import *
from tools.utils import *
from glob import glob
import time
import numpy as np
from net.generator import G_net_unet
from net.discriminator import D_net, patch_D_net
from tools.data_loader import ImageGenerator
from tools.vgg19 import Vgg19
from tools.patch_extractor import extract_top_k_img_patches_by_sum
from os.path import basename
import os


class AnimeStyle(object):

    def __init__(self, sess, args):

        self.model_name = 'AnimeStyle'
        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.init_checkpoint_dir = args.init_checkpoint_dir
        self.result_dir = args.result_dir
        self.dataset_name = args.dataset

        self.init_epoch = args.init_epoch
        self.epoch = args.epoch

        self.batch_size = args.batch_size
        self.save_freq = args.save_freq

        self.init_lr = args.init_lr
        self.d_lr = args.d_lr
        self.g_lr = args.g_lr

        """ Weight """
        self.g_adv_weight = args.g_adv_weight
        self.d_adv_weight = args.d_adv_weight
        self.con_weight = args.con_weight
        self.color_weight = args.color_weight
        self.tv_weight = args.tv_weight

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        """ Discriminator """
        self.sn = args.sn
        self.val_freq = args.val_freq

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.real = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch], name='real')
        self.anime = tf.placeholder(tf.float32, [self.batch_size, self.img_size[0], self.img_size[1], self.img_ch], name='anime')
        self.test_real = tf.placeholder(tf.float32, [1, None, None, self.img_ch], name='test_input')

        self.real_image_generator = ImageGenerator('../dataset/train_photo', self.batch_size)
        self.anime_image_generator = ImageGenerator(f'../dataset/{self.dataset_name}', self.batch_size)
        self.dataset_num = max(self.real_image_generator.num_images, self.anime_image_generator.num_images)

        self.vgg = Vgg19()

        print()
        print("##### Information #####")
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", self.batch_size)
        print("# epoch : ", self.epoch)
        print("# init_epoch : ", self.init_epoch)
        print("# training image size [H, W] : ", self.img_size)
        print("# g_adv_weight,d_adv_weight,con_weight,color_weight,tv_weight: ", self.g_adv_weight, self.d_adv_weight, self.con_weight, self.color_weight, self.tv_weight)
        print("# init_lr,g_lr,d_lr: ", self.init_lr, self.g_lr, self.d_lr)
        print()


    def generator(self, x_init, reuse=False, scope='generator'):
        with tf.variable_scope(scope, reuse=reuse):
            return G_net_unet(x_init)


    def image_discriminator(self, x_init, reuse=False, scope='image_discriminator'):
        with tf.variable_scope(scope, reuse=reuse):
            return D_net(x_init, self.sn)


    def patch_discriminator(self, x_init, reuse=False, scope='patch_discriminator'):
        with tf.variable_scope(scope, reuse=reuse):
            return patch_D_net(x_init, self.sn)


    def build_model(self):

        """ Define Generator, Discriminator """
        self.generated = self.generator(self.real, reuse=False)                                           # -1 ～ 1  b, h, w, 3
        self.generated.set_shape(shape=[self.batch_size, self.img_size[0], self.img_size[1], self.img_ch])
        # self.recovered_img = self.generator(self.blur, reuse=True)

        self.test_generated = self.generator(self.test_real, reuse=True)     # -1 ～ 1

        self.anime_patches = extract_top_k_img_patches_by_sum(self.anime, 96, 48, self.batch_size * 4)           # 4b, patch_size, patch_size, 3
        self.generated_patches = extract_top_k_img_patches_by_sum(self.generated, 96, 72, self.batch_size * 4)   # 4b, patch_size, patch_size, 3

        self.anime_patches_gray = tf.reduce_sum(self.anime_patches, axis=-1, keep_dims=True)                     # 4b, patch_size, patch_size, 1
        self.generated_patches_gray = tf.reduce_sum(self.generated_patches, axis=-1, keep_dims=True)             # 4b, patch_size, patch_size, 1

        self.anime_patches_gray = (self.anime_patches_gray - tf.reduce_min(self.anime_patches_gray, axis=[1, 2], keep_dims=True)) / \
                                  (tf.reduce_max(self.anime_patches_gray, axis=[1, 2], keep_dims=True) - tf.reduce_min(self.anime_patches_gray, axis=[1, 2], keep_dims=True) + 1e-8)

        self.generated_patches_gray = (self.generated_patches_gray - tf.reduce_min(self.generated_patches_gray, axis=[1, 2], keep_dims=True)) / \
                                  (tf.reduce_max(self.generated_patches_gray, axis=[1, 2], keep_dims=True) - tf.reduce_min(self.generated_patches_gray, axis=[1, 2], keep_dims=True) + 1e-8)

        self.anime_img_logit = self.image_discriminator(self.anime, reuse=False)
        self.generated_img_logit = self.image_discriminator(self.generated, reuse=True)

        self.anime_patch_logit = self.patch_discriminator(self.anime_patches_gray, reuse=False)
        self.generated_patch_logit = self.patch_discriminator(self.generated_patches_gray, reuse=True)


        # init pharse
        self.init_loss = self.con_weight * con_loss(self.vgg, self.real, self.generated) * 5.

        # gan
        self.l_content = self.con_weight * con_loss(self.vgg, self.real, self.generated)
        self.l_tv = self.tv_weight * total_variation_loss(self.generated)
        self.l_color = self.color_weight * color_loss(self.real, self.generated)
        self.t_loss = self.l_content + self.l_tv + self.l_color

        self.g_img_loss = self.g_adv_weight * generator_loss(self.generated_img_logit)
        self.g_patch_loss = self.g_adv_weight * generator_loss(self.generated_patch_logit)
        self.g_loss = self.g_img_loss + self.g_patch_loss

        self.d_img_loss = self.d_adv_weight * discriminator_loss(self.anime_img_logit, self.generated_img_logit)
        self.d_patch_loss = self.d_adv_weight * discriminator_loss(self.anime_patch_logit, self.generated_patch_logit)
        self.d_loss = self.d_img_loss + self.d_patch_loss

        self.Generator_loss = self.t_loss + self.g_loss
        self.Discriminator_loss = self.d_loss


        """ Training """
        self.t_vars = tf.trainable_variables()
        self.G_vars = [var for var in self.t_vars if 'generator' in var.name]
        self.D_vars = [var for var in self.t_vars if 'discriminator' in var.name]

        self.init_optim = tf.train.AdamOptimizer(self.init_lr, beta1=0.5, beta2=0.999).minimize(self.init_loss, var_list=self.G_vars)
        self.G_optim = tf.train.AdamOptimizer(self.g_lr, beta1=0.5, beta2=0.999).minimize(self.Generator_loss, var_list=self.G_vars)
        self.D_optim = tf.train.AdamOptimizer(self.d_lr, beta1=0.5, beta2=0.999).minimize(self.Discriminator_loss, var_list=self.D_vars)



    def train(self):
        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=31)
        self.init_saver = tf.train.Saver(var_list=self.G_vars, max_to_keep=1)

        """ Input Image"""
        real_img_op, anime_img_op = self.real_image_generator.load_images(), self.anime_image_generator.load_images()

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = checkpoint_counter + 1
            print(" [*] Load SUCCESS")
        else:
            could_load, checkpoint_counter = self.load_init(self.init_checkpoint_dir)
            if could_load:
                start_epoch = checkpoint_counter + 1
                print(" [*] Load SUCCESS")

            else:
                start_epoch = 1
                print(" [!] Load failed...")

        # loop for epoch
        init_mean_loss = []
        mean_loss = []

        for epoch in range(start_epoch, self.epoch + 1):

            for idx in range(int(self.dataset_num / self.batch_size)):
                anime_img, real_img = self.sess.run([anime_img_op, real_img_op])


                train_feed_dict = {
                    self.real: real_img,
                    self.anime: anime_img,
                }

                if epoch <= self.init_epoch:
                    # Init G
                    start_time = time.time()


                    _, v_loss = self.sess.run([self.init_optim, self.init_loss], feed_dict=train_feed_dict)

                    init_mean_loss.append(v_loss)

                    print("Epoch: %3d Step: %5d / %5d  time: %f s init_v_loss: %.8f  mean_v_loss: %.8f" %
                          (epoch, idx, int(self.dataset_num / self.batch_size), time.time() - start_time, v_loss, np.mean(init_mean_loss)))

                    if (idx+1) % 200 == 0:
                        init_mean_loss.clear()

                else:
                    start_time = time.time()

                    # Update D
                    _, d_img_loss, d_patch_loss = self.sess.run([self.D_optim, self.d_img_loss, self.d_patch_loss], feed_dict=train_feed_dict)

                    # Update G
                    _, g_img_loss, g_patch_loss = self.sess.run([self.G_optim, self.g_img_loss, self.g_patch_loss], feed_dict=train_feed_dict)

                    mean_loss.append([d_img_loss, d_patch_loss, g_img_loss, g_patch_loss])

                    print("Epoch: %3d Step: %5d / %5d  time: %f s d_img_loss: %.8f, d_patch_loss: %.8f, g_img_loss: %.8f, "
                          "g_patch_loss: %.8f -- mean_d_img: %.8f, mean_d_patch: %.8f, mean_g_img: %.8f, mean_g_patch: %.8f" % (
                            epoch, idx, int(self.dataset_num / self.batch_size), time.time() - start_time, d_img_loss, d_patch_loss,
                            g_img_loss, g_patch_loss, np.mean(mean_loss, axis=0)[0],
                            np.mean(mean_loss, axis=0)[1], np.mean(mean_loss, axis=0)[2], np.mean(mean_loss, axis=0)[3]))

                    if (idx + 1) % 200 == 0:
                        mean_loss.clear()

            if epoch == self.init_epoch:
                self.save(self.init_saver, self.sess, 'init_model', self.init_checkpoint_dir, epoch)


            if epoch > self.init_epoch and np.mod(epoch, self.save_freq) == 0:
                self.save(self.saver, self.sess, self.model_name, self.checkpoint_dir, epoch)


            if epoch > self.init_epoch and np.mod(epoch, self.val_freq) == 0:
                """ Result Image """
                val_files = glob('../dataset/{}/*.*'.format('val'))
                save_path = './{}/{:03d}/'.format(self.sample_dir, epoch)
                check_folder(save_path)
                for i, sample_file in enumerate(val_files):
                    print('val: ' + str(i) + sample_file)
                    sample_image = np.asarray(load_test_data(sample_file, self.img_size))
                    test_real, test_generated = self.sess.run([self.test_real, self.test_generated], feed_dict={self.test_real: sample_image})

                    save_images(test_real, self.dataset_name, save_path + basename(sample_file).split('.')[0] + '_a.jpg', None)
                    # adjust_brightness_from_photo_to_fake
                    save_images(test_generated, self.dataset_name, save_path + basename(sample_file).split('.')[0] + '_b.jpg', sample_file)



    @property
    def model_dir(self):

        return "{}_{}_g{}_d{}_con{}_color{}_tv{}".format(self.model_name, self.dataset_name,
                                                          str(self.g_adv_weight), str(self.d_adv_weight),
                                                          str(self.con_weight), str(self.color_weight), str(self.tv_weight))


    def save(self, saver, sess, model_name, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver.save(sess, os.path.join(checkpoint_dir, model_name + '.model'), global_step=step)



    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)   # first line
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(counter)
            print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, ckpt_name)))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0



    def load_with_step(self, checkpoint_dir, step):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model' + '-' + str(step)))
        print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, self.model_name + '-' + str(step))))



    def load_init(self, init_checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(init_checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)   # first line
            self.init_saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, ckpt_name)))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0



    def test(self):
        # evaluate model given the specific checkpoint
        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        check_folder(self.result_dir)

        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        val_files = glob('../dataset/{}/*.*'.format('test'))
        save_path = self.result_dir + os.path.sep + self.model_dir + os.path.sep
        check_folder(save_path)
        for i, sample_file in enumerate(val_files):
            print('val: ' + str(i) + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, self.img_size))
            test_real, test_generated = self.sess.run([self.test_real, self.test_generated],
                                                      feed_dict={self.test_real: sample_image})

            save_images(test_real, self.dataset_name, save_path + basename(sample_file).split('.')[0] + '_a.jpg', None)
            # adjust_brightness_from_photo_to_fake
            save_images(test_generated, self.dataset_name, save_path + basename(sample_file).split('.')[0] + '_b.jpg', sample_file)



    def test_with_step(self, step):
        # evaluate model trained after a specific epoch
        self.saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        self.load_with_step(self.checkpoint_dir, step)

        val_files = glob('../dataset/{}/*.*'.format('test'))
        save_path = self.result_dir + os.path.sep + self.model_dir + os.path.sep + str(step) + os.path.sep
        check_folder(save_path)
        for i, sample_file in enumerate(val_files):
            print('val: ' + str(i) + sample_file)
            sample_image = np.asarray(load_test_data(sample_file, self.img_size))
            test_real, test_generated = self.sess.run([self.test_real, self.test_generated],
                                                      feed_dict={self.test_real: sample_image})

            save_images(test_real, self.dataset_name, save_path + basename(sample_file).split('.')[0] + '_a.jpg', None)
            # adjust_brightness_from_photo_to_fake
            save_images(test_generated, self.dataset_name, save_path + basename(sample_file).split('.')[0] + '_b.jpg', sample_file)



    def test_all_step(self):
        # evaluate model trained after all training epochs to select best results
        self.saver = tf.train.Saver()
        for file in os.listdir('./checkpoint/' + self.model_dir):
            if self.model_name in file:
                step = file.split('-')[1].split('.')[0]
                self.test_with_step(step)

