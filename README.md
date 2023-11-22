# Audio_GW
A repository which contains work on the data analysis for audio signals and gravitational waves signals.

## Speech Classification with NNs

* **plot_classif_torch.ipynb** [![View filled on Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=View%20On%20Github&color=lightgrey)](https://github.com/FrancescoSarandrea/Audio_GW/blob/8066f59117a5ab88dc76042f955d3e71f94ce054/plot_classif_torch.ipynb) <a target="_blank" href="https://colab.research.google.com/github/FrancescoSarandrea/Audio_GW/blob/8066f59117a5ab88dc76042f955d3e71f94ce054/plot_classif_torch.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> is a jupyter notebook that contains the application of the 1-D scattering transform to classify audio files with a simple logistic regression. It is used for the classification of spoken digits and various environmental sounds. 

* **SpeechClassification.ipynb** [![View filled on Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=View%20On%20Github&color=lightgrey)]( https://github.com/FrancescoSarandrea/Audio_GW/blob/10332e9399d1584cfeface7ccba9dac41cd6ad72/SpeechClassification.ipynb) <a target="_blank" href="https://colab.research.google.com/github/FrancescoSarandrea/Audio_GW/blob/10332e9399d1584cfeface7ccba9dac41cd6ad72/SpeechClassification.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> Speech classification of the [Speech Commands](https://arxiv.org/abs/1804.03209) dataset using pytorch. We tested three models:
  1. Two dimensional CNN trained on transformed waveforms using 1-D scattering transform.
  2. Softmax regression trained on the "log scattering transform", see: https://www.kymat.io/index.html
  3. One dimensional CNN applied to raw waveforms, see: [Tutorial](https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html) [Paper](https://arxiv.org/pdf/1610.00087.pdf)  

    
  We also reported some possible future developments.
 
  
# Datasets Description
O3a_scatt_light_strain_whitened.zip contains hdf5 files of scattered-light glitches from Virgo.

Gravity_Spy_Glitches_whitened_#.zip contain various glitches from LIGO. They were identified starting from a slightly modified Gravity Spy dataset which can be found [here](https://zenodo.org/record/1476551#.ZFIvTbvRZop). The dataset was used to obtain the classification of the glitches and their time coordinate. Once this was done, it was possible to retrieve the time-signal from the [GWOSC website](https://gwosc.org/) using the [gwpy package](https://gwpy.github.io/docs/stable/overview/).   

# Creation of the Dataset
* **Gravity_Spy_Analysis.ipynb** <a target="blank" href="https://colab.research.google.com/drive/1L-2LxuG8wUeiNn7qjP1vC7obRZWeJY3C#scrollTo=DK1cSZ51Zuke"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
 the notebook used to perform the whiteneing of the time-signals and to create the Gravity_Spy_Glitches_whitened_# files.
* **Read_Whitened_GW.ipynb** <a target="blank" href="https://colab.research.google.com/drive/1HEvTa0_oq_23qRvcREC9qjBxbqnVxJG1#scrollTo=57UfK5QF0Xm4">  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
a notebook showing how to read the Gravity_Spy_Glitches_whitened_# files.

# Gravity Spy Machine Learning
* [DEEP MULTI-VIEW MODELS FOR GLITCH CLASSIFICATION](https://arxiv.org/pdf/1705.00034.pdf), Bahaadini et al., 2017
* [Machine learning for Gravity Spy: Glitch classification and dataset](https://www.sciencedirect.com/science/article/pii/S0020025518301634), Bahaadini et al., 2018
* [Classifying the unknown: Discovering novel gravitational-wavedetector glitches using similarity learning](https://journals.aps.org/prd/pdf/10.1103/PhysRevD.99.082002), Coughlin et al., 2019.
*  [Discriminative Dimensionality Reduction usingDeep Neural Networks for Clustering of LIGO data](https://arxiv.org/pdf/2205.13672.pdf), Bahaadin et al., 2022
*  [Data quality up to the third observing run ofAdvanced LIGO: Gravity Spy glitch classifications](https://arxiv.org/pdf/2208.12849.pdf), Glanzer et al., 2023
*  Classification of raw data with pytorch: <a target="_blank" href="https://colab.research.google.com/github/FrancescoSarandrea/Audio_GW/blob/75285c21264473870679c183650ab82fbd9311c8/GS_torchNNr.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
*  [Kaggle dataset](https://www.kaggle.com/datasets/tentotheminus9/gravity-spy-gravitational-waves): Gravity spy dataset (Q-transform data), Pytorch CNN: [![View filled on Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=View%20On%20Github&color=lightgrey)](https://github.com/FrancescoSarandrea/Audio_GW/blob/068d31261c704e1f150478389a45c13d0a0a9de5/gravityspytorchqtransform.ipynb), Pytorch Vision Transformer [![View filled on Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=View%20On%20Github&color=lightgrey)](https://github.com/FrancescoSarandrea/Audio_GW/blob/aed80babd01a4d39f41bccf4b9dc29c08d37be65/transformergs.ipynb) 

## Glitch Generation:

* [Gengli](https://git.ligo.org/melissa.lopez/gengli): GAN for glitch generation, [paper1](https://arxiv.org/pdf/2203.06494.pdf) [paper2](https://arxiv.org/pdf/2205.09204.pdf). To do: build dataset
* [Wavenet](https://arxiv.org/pdf/1609.03499.pdf): a pytorch implementation: [Code](https://github.com/vincentherrmann/pytorch-wavenet). To do: build proper dataset. *Wavenet_Pytorch_WIP.ipynb* is a notebook in which we try to perform the generation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MAGk2XOvFDsMHIPjqR7WNillP2UX6DK4#scrollTo=mqWIda7CK1Qy)

* [Denoising Diffusion Probabilistic Model](https://arxiv.org/abs/2006.11239): best state-of-the-art generative models, pytorch implementation: [Code](https://github.com/lucidrains/denoising-diffusion-pytorch). To do: guess? ahah

* Scratchy CycleGAN adapted from [Kaggle notebook](https://www.kaggle.com/code/songseungwon/cyclegan-tutorial-from-scratch-monet-to-photo) [![View filled on Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=View%20On%20Github&color=lightgrey)](https://github.com/FrancescoSarandrea/Audio_GW/blob/d37a48af813bd09f5efb68309e4e1457fa69b6e3/cycleGan.ipynb) <a target="_blank" href="https://colab.research.google.com/github/FrancescoSarandrea/Audio_GW/blob/d37a48af813bd09f5efb68309e4e1457fa69b6e3/cycleGan.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Time series Forecasting:

some paper:
* [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/pdf/1912.09363.pdf), Lim et al., (2019)
* [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504), Zeng et al., (2022)
* [Long-term Forecasting with TiDE: Time-series Dense Encoder](https://arxiv.org/pdf/2304.08424.pdf), Kong et al., (2023)

some repos: 
* [PyTorch Forecasting](https://github.com/jdb78/pytorch-forecasting) forecasting
* [Darts](https://github.com/unit8co/darts) forecasting and anomaly detection. Some examples: [![View filled on Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=View%20On%20Github&color=lightgrey)](https://github.com/FrancescoSarandrea/Audio_GW/blob/12ef2271aa2ede2263a065711f8834a22e8e91b8/dartstest-v0.ipynb)

some codes:
* D,N and Vanilla Linear Forecasting Scratchy Pytorch: [![View filled on Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=View%20On%20Github&color=lightgrey)](https://github.com/FrancescoSarandrea/Audio_GW/blob/a75f3106b37a3fd9a5296c33b22201aaf3351538/d-n-vanilla-linear.ipynb)
