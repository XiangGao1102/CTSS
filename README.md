# Learning-to-Incorporate-Texture-Saliency-Adaptive-Attention-to-Image-Cartoonization
Code of paper "Learning to Incorporate Texture Saliency Adaptive Attention to Image Cartoonization", ICML 2022.

# Citation #
<pre>
<code>
@inproceedings{gao2022learning,
  title={Learning to Incorporate Texture Saliency Adaptive Attention to Image Cartoonization},
  author={Gao, Xiang and Zhang, Yuqi and Tian, Yingjie},
  booktitle={International Conference on Machine Learning},
  pages={7183--7207},
  year={2022},
  organization={PMLR}
}
</code>
</pre>

# Introduction
This is the code of the image cartoonization method proposed in paper "Learning to Incorporate Texture Saliency Adaptive Attention to Image Cartoonization" <https://proceedings.mlr.press/v162/gao22k/gao22k.pdf>. Below is the overall model architecture, please refer to the paper for more technical details.
<image src="images/architecture.jpg">

# Required environment
Tensorflow 1.X <br>
Numpy <br>
Opencv-python <br>
Pillow (PIL) <br>

# Required files and pretrained models
## 1. VGG19 model file ##
This model requires VGG19 model file for both training and inference (testing). The link to download VGG19 file is: <br> <https://drive.google.com/drive/folders/1LrQi-oJMqmE1--VjU8r03OEJd6XM4thc> <br>
Put the downloaded vgg19.npy file into **vgg19_weight** directory for training or inference.
## 2. Pretrained models ##
The pretrained models of different cartoon styles including "The Wind Rises" (TWR), "Dragon Ball" (DB), and "Crayon Shin-chan" (CSC) can be downloaded from <https://drive.google.com/drive/folders/1xtMNvpk7OonNbK-ZjINRSXyq0wYiAF40>. The checkpoint folders corresponding to different cartoon styles are: <br>
|  Styles   | Checkpoint folder names  |
|  -------------  | -------------  |
| TWR  | AnimeStyle_TWR_g300.0_d300.0_con1.5_color15.0_tv1.0 |
| DB  | AnimeStyle_DB_g300.0_d300.0_con2.5_color15.0_tv1.5 |
| CSC  | AnimeStyle_CSC_g300.0_d300.0_con1.5_color15.0_tv1.0 |

To use these pretrained models for direct inference (testing), put these checkpoint folders into **checkpoint** directory.

# How to train or test models #
The training or testing of the model is determined by the parsing argument **--phase** in main.py:
|  Phase   | Command  |
|  -------------  | -------------  |
| Train  | python main.py --phase train |
| Test  | python main.py --phase test | 

At training phase, the model checkpoint name is composed of model hyperparameters passed in via _argparse_. At testing phase, you should also pass in the same model hyperparameters as training phase to load the correct model. Some important model hyperparameters of different cartoon styles are listed below: <br>
|  Hyperparameters | description | TWR | DB | CSC |
|  --------------  | ------------|-----|----|-----|
| g_adv_weight     | weight of adversarial loss for generator | 300.0 | 300.0 | 300.0 |
| d_adv_weight     | weight of adversarial loss for discriminator | 300.0 | 300.0 | 300.0 |
| con_weight       | weight of content loss | 1.5 | 2.5 | 1.5 |
| color_weight     | weight of color reconstruction loss | 15.0 | 15.0 | 15.0 |
| tv_weight        | weight of total variation loss | 1.0 | 1.5 | 1.0 | 

# Results display #
Some cartoonization results of "The Wind Rises" (TWR) style are displayed below. The pictures are compressed for storage convenience, please refer to our paper or try our pretrained models for clearer high-resolution cartoonized results or results of other styles (e.g., DB and CSC). 
<center>
    <image src="images/twr_results.png" />
</center>