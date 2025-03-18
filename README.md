<div align="center">
  
# Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields

**Alexander Becker<sup>❄️🔥</sup>, Rodrigo Daudt<sup>❄️🔥</sup>, Dominik Narnhofer<sup>🔥</sup>, Torben Peters<sup>🔥</sup>, Nando Metzger<sup>🔥</sup>, Jan Dirk Wegner<sup>🌶️</sup>, Konrad Schindler<sup>🔥</sup>**  
<br>
<sup>❄️</sup> Equal contribution  
<sup>🔥</sup> Photogrammetry and Remote Sensing, ETH Zurich  
<sup>🌶️</sup> Department of Mathematical Modeling and Machine Learning, University of Zurich  


[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2311.17643)
[![Page](https://img.shields.io/badge/Project-Page-green)](https://therasr.github.io)
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/prs-eth/thera)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)
</div>


<p align="center">
  <img src="assets/teaser_dark.svg#gh-dark-mode-only" alt="teaser" width=98%"/>
  <img src="assets/teaser_light.svg#gh-light-mode-only" alt="teaser" width=98%"/>
</p>
<p align="center">
  <emph>Thera</emph> is the first arbitrary-scale super-resolution method with a built-in physical observation model.
</p>

## News
**2025-03-15**: We are #1 on Hacker News 🎉<br>
**2025-03-14**: Interactive Hugging Face Space is online<br>
**2025-03-12**: Pre-trained checkpoints are released

## Setup environment
You need a Python 3.10 environment (e.g., installed via conda) on Linux as well as an NVIDIA GPU. Then install packages via pip:
```bash
> pip install --upgrade pip
> pip install -r requirements.txt
```

## Use with pre-trained models
Download checkpoints:
<table>
    <tr>
        <td><strong>Backbone</strong></td>
        <td><strong>Variant</strong></td>
        <td><strong>Download</strong></td>
    </tr>
    <tr>
        <td rowspan="3">EDSR-baseline</td>
        <td>Air</td>
        <td><a href="https://huggingface.co/prs-eth/thera-edsr-air">🤗 Hugging Face</a> &#124; <a href="https://drive.google.com/file/d/18_XYU65ZYQuQOrfnlYAoee2KjjUhXuay/view?usp=sharing">Google Drive</a></td>
    </tr>
    <tr>
        <td>Plus</td>
        <td><a href="https://huggingface.co/prs-eth/thera-edsr-plus">🤗 Hugging Face</a> &#124; <a href="https://drive.google.com/file/d/1ydYspibEQUskn67-CTc5IsTysEWsdO8Q/view?usp=sharing">Google Drive</a></td>
    </tr>
    <tr>
        <td>Pro</td>
        <td><a href="https://huggingface.co/prs-eth/thera-edsr-pro">🤗 Hugging Face</a> &#124; <a href="https://drive.google.com/file/d/18slBa-dv-Z6SCTzL65MtmIryksfzdcnE/view?usp=sharing">Google Drive</a></td>
    </tr>
    <tr>
        <td rowspan="3">RDN</td>
        <td>Air</td>
        <td><a href="https://huggingface.co/prs-eth/thera-rdn-air">🤗 Hugging Face</a> &#124; <a href="https://drive.google.com/file/d/1EzJaexc_OoxinaLZYXs6BRJMQ1XgZRwO/view?usp=sharing">Google Drive</a></td>
    </tr>
    <tr>
        <td>Plus</td>
        <td><a href="https://huggingface.co/prs-eth/thera-rdn-plus">🤗 Hugging Face</a> &#124; <a href="https://drive.google.com/file/d/1mnn3XUSeWs-yBEpWcXSV7_jep_4nOdSo/view?usp=sharing">Google Drive</a></td>
    </tr>
    <tr>
        <td>Pro</td>
        <td><a href="https://huggingface.co/prs-eth/thera-rdn-pro">🤗 Hugging Face</a> &#124; <a href="https://drive.google.com/file/d/1h6MPs6HSx5kVx3m703gZNbE-d5EMV5CD/view?usp=sharing">Google Drive</a></td>
    </tr>
</table>


Super-resolve any image with:
```bash
> ./super_resolve.py IN_FILE OUT_FILE --scale 3.14 --checkpoint thera-rdn-pro.pkl
```

You can evaluate the models on datasets using the `run_eval.py` script, e.g.:

```bash
> python run_eval.py --checkpoint thera-rdn-pro.pkl --data-dir path_to_data_parent_folder --eval-sets data_folder_1 data_folder_2 ...
```

You can run `python run_eval.py -h` to display all testing options.

## Use with interactive Gradio app
You can also host a local version of our [Hugging Face demo](https://huggingface.co/spaces/prs-eth/thera). To do so, clone the dedicated demo repo:

```bash
> git clone https://huggingface.co/spaces/prs-eth/thera thera-demo
```

Instructions for running the demo locally can be found in [the repo's README.md](https://huggingface.co/spaces/prs-eth/thera/blob/main/README.md).

## Training
> Training code will be released soon.

## Useful XLA flags
* Disable pre-allocation of entire VRAM: `XLA_PYTHON_CLIENT_PREALLOCATE=false`
* Disable jitting for debugging: `JAX_DISABLE_JIT=1`

## Citation

If you found our work helpful, consider citing our paper 😊:

```
@article{becker2025thera,
  title={Thera: Aliasing-Free Arbitrary-Scale Super-Resolution with Neural Heat Fields},
  author={Becker, Alexander and Daudt, Rodrigo Caye and Narnhofer, Dominik and Peters, Torben and Metzger, Nando and Wegner, Jan Dirk and Schindler, Konrad},
  journal={arXiv preprint arXiv:2311.17643},
  year={2025}
}
```
