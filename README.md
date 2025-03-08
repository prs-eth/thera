# Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields

Official implementation of the paper by Alexander Becker*, Rodrigo Daudt*, Dominik Narnhofer, Torben Peters, Nando Metzger, Jan Dirk Wegner, Konrad Schindler (* equal contribution)

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

## Setup environment
You need a Python > 3.10 environment (e.g., installed via conda) on Linux as well as an NVIDIA GPU (or cloud TPU). Then install packages via pip:
```bash
> pip install --upgrade pip
> pip install -r requirements.txt
```

## Use with pre-trained models
> Checkpoints will be released soon.


Super-resolve any image with:
```bash
> ./super_resolve.py IN_FILE OUT_FILE --scale 3.14 --checkpoint checkpoints/thera-pro-edsr-baseline.pkl
```

You can evaluate the models on datasets using the `run_eval.py` script, e.g.:

```bash
> python run_eval.py --checkpoint checkpoints/thera-plus-edsr-baseline.pkl --data-dir path_to_data_parent_folder --eval-sets data_folder_1, data_folder_2, ...
```

Check the arguments in `args.py` (bottom of file) for all testing options.

## Training
> Training code will be released soon.

## Useful XLA flags
* Disable pre-allocation of entire VRAM: `XLA_PYTHON_CLIENT_PREALLOCATE=false`
* Force GPU determinism (slow): `XLA_FLAGS=--xla_gpu_deterministic_ops=true`
* Disable jitting for debugging: `JAX_DISABLE_JIT=1`

## Citation

> Citation coming soon.
