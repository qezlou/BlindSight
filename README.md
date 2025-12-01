# BlindSight: GenAI for trillion-resolution element blind surveys
![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/github/license/qezlou/hetgen)
![Status](https://img.shields.io/badge/status-in--progress-yellow)

> **Note**: This repository is under active development. A detailed publication describing the methodology and results will be released soon.

![]()
HETGEN is a research-grade, generative deep learning framework designed to detect faint Lyman-alpha (Lyα) emission lines in galaxy spectra. Built for the HETDEX (Hobby-Eberly Telescope Dark Energy Experiment) survey, this model combines the strengths of Transformer architectures and Variational Autoencoders (VAEs) to effectively model complex spectral features while mitigating the influence of noise, sky residuals, and other systematics.

## Key Features

- **Hybrid Transformer-VAE architecture**: Enables robust feature encoding with generative capabilities tailored for spectral data.
- **Targeted for HETDEX**: Specifically optimized for Integral Field Unit (IFU) spectroscopy data from the HETDEX survey.
- **False positive mitigation**: Designed to identify and reduce false detections of emission lines using learned latent representations.
- **Synthetic spectrum generation**: Capable of producing synthetic emission lines and modeling complex noise conditions.

- 

## Use Cases

- Automated detection of Lyα emission lines in noisy spectral data
- Filtering of false positives in large IFU surveys
- Generative modeling and data augmentation of astronomical spectra
- Research in representation learning for astrophysical data

## Installation

```bash
conda create -n hetgen_env python=3.9 numpy scipy astropy h5py matplotlib tqdm pytorch torchvision -c pytorch -c conda-forge
conda activate hetgen_env

git clone https://github.com/your-username/hetgen.git
cd hetgen
python -m pip install -e .
```


## Getting Started

Example scripts and notebooks are provided in the `notebooks/` directory to demonstrate:
- Preprocessing of HETDEX spectral data
- Model training and evaluation
- Latent space exploration and synthetic data generation

## Input Description

The input to the HETGEN pipeline consists of observational data from the HETDEX IFU survey. These include:

- **Flux Spectrum**: The observed flux as a function of wavelength, containing both astronomical signal and sky background.
- **Noise estimates**: Estimated cross the spectrum
- **Sky Spectrum**: A model or measurement of the sky emission spectrum at the time of observation, used for identifying sky contamination.
- **Fiber Index**: Identifiers for the spatial position of each spectrum within the IFU field.

These inputs help train and evaluate the model's ability to detect true emission lines and reduce false positives.

## How to Run:
Write the config file for the hyperparameters
```py
from hetgen import training
# save a json  condig file training_config.json
training.create_example_config()
```
Write a script to call this:
```py
from hetgen.training import create_trainer_from_config

trainer = create_trainer_from_config('/scratch/06536/qezlou/encoder/submit/training_config.json')
trainer.fit(num_epochs=10)
```

Run training on as many GPU as you have, e.g. 4 here:

```sh
torchrun --nproc_per_node=4 run_train.py
```



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
