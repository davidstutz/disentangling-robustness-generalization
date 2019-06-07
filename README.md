# Disentangling Adversarial Robustness and Generalization

This repository contains data and code corresponding to

D. Stutz, M.Hein, B.Schiele. **Disentangling Adversarial Robustness and Generalization**. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

Please cite as:

    @inproceedings{Stutz2019CVPR,
      title = {Disentangling Adversarial Robustness and Generalization},
      author = {Stutz, David and Hein, Matthias and Schiele, Bernt},
      booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      publisher = {IEEE Computer Society},
      year = {2019}
    }

Also check the [project page](https://davidstutz.de/projects/adversarial-robustness/) for the final publication, code and data.

This repository includes PyTorch implementations of the PGD attack [1],
the C+W attack [2], adversarial training [1] as well as adversarial training
variants for adversarial deformations and on-manifold adversarial examples.

Additionally, the repository allows to easily reproduce the experiments
as reported in the paper.

![Disentangling Adversarial Robustness and Generalization.](screenshot.png?raw=true "Disentangling Adversarial Robustness and Generalization.")

## Getting Started

The main goal of the provided code is to reproduce the results reported in the paper.

### Overview

The code includes several modules, most of which are tailored to the reported experiments;
some, like the implemented attacks, can also be used standalone:

* `attacks`: implemented adversarial attacks, including [1] and [2]
* `common`: various utilities, including:
    * `latex.py`, `vis.py`, `plot.py`: utilities for plotting and visualization;
    * `cuda.py`, `torch.py`, `numpy.py`: PyTorch and NumPy utilities;
    * `ppca.py`: an implementation of probabilistic PCA [3];
    * `state.py`, `scheduler.py`, `timer.py`: utilities for training;
* `data/fonts`: generation code for the synthetic FONTS dataset;
* `experiments`: implemented experiments;
* `training`: command line tools for training, testing, attacking models.

    [1] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. ICLR, 2018.
    [2] Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In SP, 2017.
    [3] Michael E. Tipping, Chris M. Bishop. Probabilistic Principal Component Analysis. Journal of the Royal Statistical Society, 1999.

The experiments module will be the main focus of the following guide.

### Installation. Preparing Experiments

Before running experiments, the environment
needs to be set up, the datasets need to be downloaded, and the
pre-trained VAE-GAN models need to be downloaded

**Installation.** The experiments were run with on a custom Debian 9
with Tesla V100 GPUs and

* Python 3.5.3
* CUDA 9
* `torch` 0.4.0
* `torchvision` 0.2.1
* `h5py` 2.9.0
* `numpy` 1.16.3
* `scipy` 1.2.1
* `scikit-learn` 0.19.1
* `scikit-image` 0.14.2
* `umap-learn` 0.2.3
* `xgboost` 0.72
* `imageio` 2.5.0

**Data.** The data in the correct formats can be downloaded in the
following table:

| Dataset | Link
|---|---|
|FONTS|[cvpr2019_adversarial_robustness_fonts.tar.gz](https://datasets.d2.mpi-inf.mpg.de/cvpr2019-adversarial-robustness/cvpr2019_adversarial_robustness_fonts.tar.gz)|
|EMNIST|[cvpr2019_adversarial_robustness_emnist.tar.gz](https://datasets.d2.mpi-inf.mpg.de/cvpr2019-adversarial-robustness/cvpr2019_adversarial_robustness_emnist.tar.gz)|
|Fashion-MNIST|[cvpr2019_adversarial_robustness_fashion.tar.gz](https://datasets.d2.mpi-inf.mpg.de/cvpr2019-adversarial-robustness/cvpr2019_adversarial_robustness_fashion.tar.gz)|

Each datasets provides HDF5 files for training/test images and labels;
FONTS additionally provides test/training latent codes.

For FONTS, as example, these HDF5 files look as follows; images are stored
as floats in `[0,1]`:

* `database.h5`: `1000x10x28x28` (only for FONTS, prototype image for each of the 1000 fonts and 10 classes)
* `codes.h5`: `1120000x3` (only for FONTS, classes not splitted into train/test)
* `theta.h5`: `1120000x6` (only for FONTS, latent codes not splitted into train/test)
* `images.h5`: `1120000x28x28x1` (only for FONTS, images not splitted into train/test)
* `train_codes.h5`: `960000x3` (for FONTS, the class label is stored in the third column)
* `train_theta.h5`: `960000x6` (only for FONTS, latent codes)
* `train_images.h5`: `960000x28x28x1`
* `test_codes.h5`: `160000x3` (for FONTS, the class label is stored in the third column)
* `test_theta.h5`: `160000x6` (only for FONTS, latent codes)
* `test_images.h5`: `160000x28x28x1`

[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) needs to be converted
manually as the license does not allow redistribution. For this, `data/celeba`
includes two simple scripts, `convert_images.py` and `convert_labels.py`; for these
scripts, the dataset should be downloaded into `BASE_DATA` such that
the following directories/files exist:

    BASE_DATA/CelebA/img_align_celeba
    Base_DATA/CelebA/Anno/list_attr_celeba.txt

**VAE-GAN Models.** To approximate the manifolds of these datasets, we
use VAE-GANs; The models for all datasets can be downloaded here:

|Pre-trained VAE-GANs|
|---|
|[cvpr2019_adversarial_robustness_manifolds.tar.gz](https://datasets.d2.mpi-inf.mpg.de/cvpr2019-adversarial-robustness/cvpr2019_adversarial_robustness_manifolds.tar.gz)|

For each dataset, a class-agnostic VAE-GAN in the form of encoder,
decoder and classifier is provided and several class-specifiv VAE-GANs.

**Setting up Paths.** To automate the experiments, two main directories
are required: a data directory and an experiment directory. Both can be
set in `common/paths.py` by setting `BASE_DATA` and `BASE_EXPERIMENTS`.

`BASE_DATA` should include the directories for the datasets, for example
`fonts`, `emnist`, `fashion` and/or `CelebA` (mind capitalization!). If the
directories are named differently, the corresponding paths in `paths.py`
need to be changed.

In the end, run `python3 setup.py` to check the above steps.

### Running Experiments

The experiments on all datasets can be found in `experiments/`. The base experiment
is `verify_hypotheses.py` which can be used to reproduce our results for
all four claims:

1. adversarial examples leave the manifold;
2. on-manifold adversarial examples exist;
3. on-manifold robustness is essentially generalization;
4. and regular robustness is not necessarily contradicting generalization.

For the fourth claim, `verify_long_resnet_hypotheses.py` is useful, as well.

In the paper, we trained 5 networks, using different strategies, on
`N` training examples with `N` between `100` and `40000`. This can be done
using

    # Starting with the experiment on FONTS:
    cd experiments/fonts
    python3 verify_hypotheses.py -training_sizes=250,500,1000,1500,2000,2500,5000,7500,10000,15000,20000,30000,40000 -max_models=5

For smaller-scale experiments, less models can be trained, or the scale of
`N` can be reduced. For larger scale experiments, individual `N`s can be
run on different GPUs. Additionally, the model to start with can be controlled
using `-start_model`. This means `-start_model=4 -max_models=5` will only run
the fifth model for each `N`.

The experiment can be evaluated using `-mode=evaluate`. This will create a
directory `0_evaluation`, for example `BASE_EXPERIMENTS/VerifyHypotheses/Fonts/0_evaluation`
which contains the plots as reported in the paper. The plots are provided as images
and as LaTeX source. Note that for the paper, the LaTeX source was adapted
for better readability. The script creates plots for error rate and success rates
(off-manifold, on-manifold - true and learned manifold). On Fonts, the
on-manifold success rate might look as follows:

![On-Manifold Success Rate on FONTS.](plot.png?raw=true "On-Manifold Success Rate on FONTS.")

The following experiments are implemented for most datasets:

* `verify_hypotheses.py`: main experiment as reported in the paper,
  using `L_inf` adversarial examples and a simple LeNet-like architecture
  consisting of 3 to 4 convolutional layers with stride two followed
  by ReLU activations and batch normalization;
* `verify_l2_hypotheses.py`: main experiment as above but using `L_2`
  adversarial examples;
* `verify_[mlp|resnet|vgg2]_hypotheses.py`: main experiment using
  a multi-layer perceptron architecture, ResNet architecture or VGG
  architecture (without dropout);

## License

Licenses for source code and data corresponding to:

D. Stutz, M.Hein, B.Schiele. **Disentangling Adversarial Robustness and Generalization**. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

Note that the source code and/or data is based on other projects for which separate licenses apply:

* [MNIST](http://yann.lecun.com/exdb/mnist/) and [EMNIST](https://www.nist.gov/node/1298471/emnist-dataset);
* [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist);
* [Google Fonts](https://github.com/google/fonts);

### Source Code

Copyright (c) 2019 David Stutz, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").**

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the corresponding papers (see above) in documents and papers that report on research using the Software.

### Data

Copyright (c) 2019 David Stutz, Max-Planck-Gesellschaft

**Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use the data (the "Dataset").**

The authors grant you a non-exclusive, non-transferable, free of charge right: To download the Dataset and use it on computers owned, leased or otherwise controlled by you and/or your organisation; To use the Dataset for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

Without prior written approval from the authors, the Dataset, in whole or in part, shall not be further distributed, published, copied, or disseminated in any way or form whatsoever, whether for profit or not. This includes further distributing, copying or disseminating to a different facility or organizational unit in the requesting university, organization, or company.

THE DATASET IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE DATASET OR THE USE OR OTHER DEALINGS IN THE DATASET.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Dataset. The authors nevertheless reserve the right to update, modify, or discontinue the Dataset at any time.

You agree to cite the corresponding papers (see above) in documents and papers that report on research using the Dataset.
