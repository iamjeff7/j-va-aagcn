## Credit to:
    1. [2s-AGCN]
    2. [View Adaptive Neural Networks (VA) for Skeleton-based Human Action Recognition]

# Introduction
    This is Skeleton-based action recognition project that combine 2 methods. 
    The first method is view adaptive subnetwork to learn the best suitable viewpoint.
    The second method is the attention-enhanced adative graph convolution network, which is a new way of tackling skeleton data by treating it as graph rather than sequnce.
    The second method supposed to have 2 streams but due time constraints, I was able to test with only one stream (joint)

    The codes are mainly from [2s-AGCN]. I convert VA subnetwork from Keras to Pytorch so that it can be integrated with 2s-agcn.
    -[Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1805.07694)
    -[Skeleton-Based Action Recognition with Multi-Stream Adaptive Graph Convolutional Networks](https://arxiv.org/abs/1912.06971)
    -[View Adaptive Neural Networks for High Performance Skeleton-based Human Action Recognition](https://arxiv.org/pdf/1804.07453.pdf). TPAMI, 2019.
    -[View Adaptive Recurrent Neural Networks for High Performance Human Action Recognition from Skeleton Data](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_View_Adaptive_Recurrent_ICCV_2017_paper.pdf). ICCV, 2017.

# Dependencies
    python >= 3.6
    $ python -m venv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt

# Data Preparation

 - Download the raw data from [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) and [Skeleton-Kinetics](https://github.com/yysijie/st-gcn). Then put them under the data directory:
 - Or just download already processed from [here](https://drive.google.com/drive/folders/17V0TWh4GuHZITnZONornEXlEwgkBKj8X?usp=sharing)

        -data\
          -kinetics_raw\
            -kinetics_train\
              ...
            -kinetics_val\
              ...
            -kinetics_train_label.json
            -keintics_val_label.json
          -nturgbd_raw\
            -nturgb+d_skeletons\
              ...
            -samples_with_missing_skeletons.txt

 - Preprocess the data with

    $ python data_gen/ntu_gendata.py


# Training
    $ python main.py --config ./config/nturgbd-cross-subject/train_joint_aagcn.yaml

# Testing
    $ python main.py --config ./config/nturgbd-cross-subject/test_joint_aagcn.yaml

# References
    @inproceedings{2sagcn2019cvpr,
        title     = {Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition},
        author    = {Lei Shi and Yifan Zhang and Jian Cheng and Hanqing Lu},
        booktitle = {CVPR},
        year      = {2019},
    }

    @article{shi_skeleton-based_2019,
        title = {Skeleton-{Based} {Action} {Recognition} with {Multi}-{Stream} {Adaptive} {Graph} {Convolutional} {Networks}},
        journal = {arXiv:1912.06971 [cs]},
        author = {Shi, Lei and Zhang, Yifan and Cheng, Jian and LU, Hanqing},
        month = dec,
        year = {2019},
	}
    @article{zhang2019view,
        title={View adaptive neural networks for high performance skeleton-based human action recognition},
        author={Zhang, Pengfei and Lan, Cuiling and Xing, Junliang and Zeng, Wenjun and Xue, Jianru and Zheng, Nanning},
        journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
        year={2019},
    }

    @inproceedings{zhang2017view,
        title={View adaptive recurrent neural networks for high performance human action recognition from skeleton data},
        author={Zhang, Pengfei and Lan, Cuiling and Xing, Junliang and Zeng, Wenjun and Xue, Jianru and Zheng, Nanning},
        booktitle={Proceedings of the IEEE International Conference on Computer Vision},
        pages={2117--2126},
        year={2017}
    }

[2s-AGCN]:https://github.com/lshiwjx/2s-AGCN
[View Adaptive Neural Networks (VA) for Skeleton-based Human Action Recognition]:https://github.com/microsoft/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition
