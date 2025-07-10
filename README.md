# HETGEN: HETDEX Generative Transformer-VAE for Emission Line Detection

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/github/license/qezlou/hetgen)
![Status](https://img.shields.io/badge/status-in--progress-yellow)

> **Note**: This repository is under active development. A detailed publication describing the methodology and results will be released soon.

HETGEN is a research-grade, generative deep learning framework designed to detect faint Lyman-alpha (Lyα) emission lines in galaxy spectra. Built for the HETDEX (Hobby-Eberly Telescope Dark Energy Experiment) survey, this model combines the strengths of Transformer architectures and Variational Autoencoders (VAEs) to effectively model complex spectral features while mitigating the influence of noise, sky residuals, and other systematics.

## Key Features

- **Hybrid Transformer-VAE architecture**: Enables robust feature encoding with generative capabilities tailored for spectral data.
- **Targeted for HETDEX**: Specifically optimized for Integral Field Unit (IFU) spectroscopy data from the HETDEX survey.
- **False positive mitigation**: Designed to identify and reduce false detections of emission lines using learned latent representations.
- **Synthetic spectrum generation**: Capable of producing synthetic emission lines and modeling complex noise conditions.

## Use Cases

- Automated detection of Lyα emission lines in noisy spectral data
- Filtering of false positives in large IFU surveys
- Generative modeling and data augmentation of astronomical spectra
- Research in representation learning for astrophysical data

## Installation

```bash
conda create -n hetgen_env python=3.9 numpy scipy astropy h5py matplotlib pytorch torchvision -c pytorch -c conda-forge
conda activate hetgen_env

git clone https://github.com/your-username/hetgen.git
cd hetgen
pip install -r requirements.txt
```

## Getting Started

Example scripts and notebooks are provided in the `notebooks/` directory to demonstrate:
- Preprocessing of HETDEX spectral data
- Model training and evaluation
- Latent space exploration and synthetic data generation

## Citation

If you use this codebase in your work, please cite:

```
@misc{hetgen2025,
  title={HETGEN: A Generative Transformer-VAE for HETDEX},
  author={Mahdi (Sum) Qezlou},
  year={2025},
  url={https://github.com/qezlou/hetgen}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.