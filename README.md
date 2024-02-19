<h1 align="center">
<a href="https://arxiv.org/pdf/2401.14401.pdf">Range-Agnostic Multi-View Depth Estimation With Keyframe Selection</a>
<div style="margin-top: 5px; font-size: 0.5em">aka</div>
<div style="font-size: 0.8em">
    <img src="https://github.com/andreaconti/ramdepth/blob/main/media/icon.png" style="vertical-align: middle;" width="40px"/>
    <span style="vertical-align: middle;">RAMDepth</span>
    <img src="https://github.com/andreaconti/ramdepth/blob/main/media/icon.png" style="vertical-align: middle;" width="40px"/>
</div>
</h1>

<p>
<div align="center">
    <a href="https://andreaconti.github.io">Andrea Conti</a>
    &middot;
    <a href="https://mattpoggi.github.io">Matteo Poggi</a>
    &middot;
    <a href="">Valerio Cambareri</a>
    &middot;
    <a href="http://vision.deis.unibo.it/~smatt/Site/Home.html">Stefano Mattoccia</a>
</div>
<div align="center">
    <a href="https://arxiv.org/abs/2401.14401">[Arxiv]</a>
    <a href="https://andreaconti.github.io/projects/range_agnostic_multi_view_depth">[Project Page]</a>
</div>
</p>

![](https://github.com/andreaconti/ramdepth/blob/main/media/teaser.png)

Multi-View 3D reconstruction techniques process a set of source views and a reference view to yield an estimated depth map for the latter. Unluckily, state-of-the-art frameworks

1. require to know _a priori_ the depth range of the scene, in order to sample a set of _depth hypotheses_ and build a meaningful _cost volume_.
2. do not take into account the _keyframes selection_.

In this paper, we propose a novel framework **free from prior knowledge of the scene depth range** and capable of **distinguishing the most meaningful source frames**. The proposed method unlocks the capability to apply multi-view depth estimation to a wider range of scenarios like large-scale outdoor environments, top-view buildings and large-scale outdoor environments.

Our method relies on an **iterative approach**: starting from a zero-initialized depth map we extract geometrical correlation cues and update the prediction. At each iteration we feed also information extracted from the reference view only (the one on which we desire to compute depth). 
Moreover, at each iteration we use a different source view to exploit multi-view information in a round-robin fashion. For more details please refer to the [paper](https://arxiv.org/pdf/2401.14401.pdf).

## Citation

```bibtex
@InProceedings{Conti_2024_3DV,
    author    = {Conti, Andrea and Poggi, Matteo and Cambareri, Valerio and Mattoccia, Stefano},
    title     = {Range-Agnostic Multi-View Depth Estimation With Keyframe Selection},
    booktitle = {International Conference on 3D Vision},
    month     = {March},
    year      = {2024},
}
```

## Evaluation Code

In this repo we provide __evaluation__ code for our paper, it allows to load the pre-trained models on Blended and TartanAir and test them. Please note that <u>we do not provide the source code of our models</u> but only compiled binaries to perform inference.

Dependencies can be installed with `conda` or `mamba` as follows:

```bash
$ # first of all clone the repo and build the conda environment
$ git clone https://github.com/andreaconti/ramdepth.git
$ cd ramdepth
$ conda env create -f environment.yml  # use mamba if conda is too slow
$ conda activate ramdepth
$ # then, download and install the wheel containing the pretrained models, available for linux, windows and macos
$ pip install https://github.com/andreaconti/ramdepth/releases/download/wheels%2Fv0.1.0/ramdepth-0.1.0-cp310-cp310-linux_x86_64.whl --no-deps
```

Then you can run the [evaluate.ipynb](https://github.com/andreaconti/ramdepth/blob/main/evaluate.ipynb) to select the dataset and pre-trained model you want to test. Results may be slightly different with respect to the results in the main paper due to small differences in dataloaders and framework due to the packaging process.
