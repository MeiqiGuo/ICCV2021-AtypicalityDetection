# ICCV2021-AtypicalityDetection
This is a PyTorch implementation of the model proposed in our ICCV 2021 paper: [Detecting Persuasive Atypicality by Modeling Contextual Compatibility](https://openaccess.thecvf.com/content/ICCV2021/papers/Guo_Detecting_Persuasive_Atypicality_by_Modeling_Contextual_Compatibility_ICCV_2021_paper.pdf).

## Environment Setup
>python=3.6
>
>pytorch=1.6.0
>
>torchvision=0.7.0

## Self-supervised training with our models and detecting atypicality on test samples

1) Data:

Download the Ads dataset at [page](https://people.cs.pitt.edu/~kovashka/ads/#image). 

Extract the image features by the Faster R-CNN feature extractor demonstrated in ["Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering" (CVPR 2018)](https://arxiv.org/abs/1707.07998). You may refer to these github ([1](https://github.com/peteanderson80/bottom-up-attention), [2](https://github.com/airsplay/lxmert#faster-r-cnn-feature-extraction), [3](https://github.com/violetteshev/bottom-up-features)) for the extraction. 

Please save them and adapt the structure of folders with our code. 

2) Self-supervised training:

First update the config file according to your data paths.

Use Spatial-Relative Transformer:
>python main.py --train --test --svte --cartesian

Use Transformer:
>python main.py --train --test --vte

3) Detecting atypicality by pre-trained model:

First update the config file according to your data paths and the saved model path.

Use Spatial-Relative Transformer:
>python main.py --atypical_test --svte --cartesian

Use Transformer:
>python main.py --atypical_test --vte

## Citation

If you make use of this code, please kindly cite our paper:
```
@InProceedings{Guo_2021_ICCV,
    author    = {Guo, Meiqi and Hwa, Rebecca and Kovashka, Adriana},
    title     = {Detecting Persuasive Atypicality by Modeling Contextual Compatibility},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {972-982}
}
```
