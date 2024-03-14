# Neural Fields with Thermal Activations for Arbitrary-Scale Super-Resolution

Official implementation of the paper "Neural Fields with Thermal Activations for Arbitrary-Scale Super-Resolution" by
Alexander Becker*, Rodrigo Daudt*, Nando Metzger, Jan Dirk Wegner, Konrad Schindler (* equal contribution)

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2311.17643)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

<p align="center">
  <img src="assets/teaser_dark.png#gh-dark-mode-only" alt="teaser" width=85%"/>
  <img src="assets/teaser_light.png#gh-light-mode-only" alt="teaser" width=85%"/>
</p>

## Setup environment
You need a Python 3.10 environment (e.g., installed via conda) on Linux as well as an NVIDIA GPU (or cloud TPU). Then install packages via pip:
```bash
> pip install --upgrade pip
> pip install -r requirements_cu11.txt  # CUDA 11
# or
> pip install -r requirements_cu12.txt  # CUDA 12
# or
> pip install -r requirements_tpu.txt  # TPU
```

## Use with pre-trained checkpoints
Download checkpoints [[here]](https://drive.google.com/drive/folders/1x1cwfG-DG2LXm3nKPZxRibvZTesqimiw?usp=sharing). Super-resolve any image with, e.g.:

```bash
> ./super_resolve.py IN_FILE OUT_FILE --scale 3.14 --checkpoint checkpoints/thera-L-swin-ir.pkl --backbone swin-ir --model-size L
```

You can evaluate the models on datasets using the `run_eval.py` script, e.g.:

```bash
> python run_eval.py --checkpoint checkpoints/thera-M-edsr-baseline.pkl --data-dir path_to_data_parent_folder --eval-sets data_folder_1, data_folder_2, ...
```

Check the arguments in `args.py` (bottom of file) for all testing options.

## Training
Train and evaluate using

```bash
> python run_train_and_eval.py --data-dir path_to_data_parent_folder --train-set train_data_folder --val-set val_data_folder
```

Check the arguments in `args.py` for all training options. Our implementation will automatically shard over all available devices, this can be overwritten by manually setting `--n-devices` or `CUDA_VISIBLE_DEVICES`.

## Useful XLA flags
* Disable pre-allocation of entire VRAM: XLA_PYTHON_CLIENT_PREALLOCATE=false
* Force GPU determinism (slow): XLA_FLAGS=--xla_gpu_deterministic_ops=true
* Disable jitting for debugging: JAX_DISABLE_JIT=1

## Citation

Please cite our paper if you found our work helpful:

```bibtex
@article{becker2023neural,
  title={Neural Fields with Thermal Activations for Arbitrary-Scale Super-Resolution},
  author={Becker, Alexander and Daudt, Rodrigo Caye and Metzger, Nando and Wegner, Jan Dirk and Schindler, Konrad},
  journal={arXiv preprint arXiv:2311.17643},
  year={2023}
}
```
