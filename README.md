# Audio_GW
A repository which contains work on the data analysis for audio signals and gravitational waves signals.

## Speech Classification with Wavelet Scattering Transform

* **plot_classif_torch.ipynb** [![View filled on Github](https://img.shields.io/static/v1.svg?logo=github&label=Repo&message=View%20On%20Github&color=lightgrey)](https://github.com/FrancescoSarandrea/Audio_GW/blob/8066f59117a5ab88dc76042f955d3e71f94ce054/plot_classif_torch.ipynb) <a target="_blank" href="https://colab.research.google.com/github/FrancescoSarandrea/Audio_GW/blob/8066f59117a5ab88dc76042f955d3e71f94ce054/plot_classif_torch.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> is a jupyter notebook that contains the application of the 1-D scattering transform to classify audio files with a simple logistic regression. It is used for the classification of spoken digits and various environmental sounds. 

* jupyter notebook  Speech classification of the [Speech Commands](https://arxiv.org/abs/1804.03209) dataset using pytorch. We tested three models:
  1. Two dimensional CNN trained on transformed waveform.
  2. Softmax regression trained on the "log scattering", see: https://www.kymat.io/index.html
  3. One dimensional CNN applied to raw waveform, see: [Tutorial](https://pytorch.org/tutorials/intermediate/speech_command_classification_with_torchaudio_tutorial.html) [Paper](https://arxiv.org/pdf/1610.00087.pdf)  
    
  We also report some possible future developments.
  
