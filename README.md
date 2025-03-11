# Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields

Official implementation of the paper by Alexander Becker*, Rodrigo Daudt*, Dominik Narnhofer, Torben Peters, Nando Metzger, Jan Dirk Wegner and Konrad Schindler (* equal contribution)

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2311.17643)
[![Page](https://img.shields.io/badge/Project-Page-green)](https://therasr.github.io)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

<p align="center">
  <img src="assets/teaser_dark.svg#gh-dark-mode-only" alt="teaser" width=98%"/>
  <img src="assets/teaser_light.svg#gh-light-mode-only" alt="teaser" width=98%"/>
</p>
<p align="center">
  <emph>Thera</emph> is the first arbitrary-scale super-resolution method with a built-in physical observation model.
</p>

## Setup environment
You need a Python 3.10 environment (e.g., installed via conda) on Linux as well as an NVIDIA GPU. Then install packages via pip:
```bash
> pip install --upgrade pip
> pip install -r requirements.txt
```

## Use with pre-trained models
> Checkpoints will be released soon.


Super-resolve any image with:
```bash
> ./super_resolve.py IN_FILE OUT_FILE --scale 3.14 --checkpoint thera-rdn-pro.pkl
```

You can evaluate the models on datasets using the `run_eval.py` script, e.g.:

```bash
> python run_eval.py --checkpoint thera-rdn-pro.pkl --data-dir path_to_data_parent_folder --eval-sets data_folder_1 data_folder_2 ...
```

Check the arguments in `args.py` (bottom of file) for all testing options.

## Training
> Training code will be released soon.

## Useful XLA flags
* Disable pre-allocation of entire VRAM: `XLA_PYTHON_CLIENT_PREALLOCATE=false`
* Disable jitting for debugging: `JAX_DISABLE_JIT=1`

## Citation

> Citation coming soon.
