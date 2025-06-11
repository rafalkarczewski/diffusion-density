# Density in Diffusion Models - Evaluation & Control

This repository contains supplementary code for the following two articles

* [1] [Diffusion Models as Cartoonists: The Curious Case of High Density Regions](https://arxiv.org/abs/2411.01293) - novel tools for likelihood estimation in stochastic diffusion models, as well as mode-tracking.
* [2] [Devil is in the Details: Density Guidance for Detail-Aware Generation with Flow Models](https://arxiv.org/abs/2502.05807) - explicit control over the log-density of generated samples.

For a high-level overview with examples and visualizations, please see our [blog post](https://rafalkarczewski.github.io/blog/2025/diffusion-density/). If you find our work useful in your research, please consider citing (expand for BibTeX):

<details>
<summary>
R. Karczewski, M. Heinonen, V. Garg. Diffusion Models as Cartoonists: The Curious Case of High Density Regions (ICLR 2025)
</summary>

```bibtex
@inproceedings{
karczewski2025diffusion,
title={Diffusion Models as Cartoonists: The Curious Case of High Density Regions},
author={Rafal Karczewski and Markus Heinonen and Vikas Garg},
booktitle={ICLR},
year={2025},
url={https://arxiv.org/abs/2411.01293}
}
```
</details>


<details>
<summary>
R. Karczewski, M. Heinonen, V. Garg. Devil is in the Details: Density Guidance for Detail-Aware Generation with Flow Models (ICML 2025)
</summary>

```bibtex
@inproceedings{
karczewski2025devil,
title={Devil is in the Details: Density Guidance for Detail-Aware Generation with Flow Models},
author={Rafal Karczewski and Markus Heinonen and Vikas Garg},
booktitle={ICML},
year={2025},
url={https://arxiv.org/abs/2502.05807}
}
```

</details>

## Setup

Install dependencies `pip install -r requirements.txt`.

## Code structure

#### Model Checkpoints

This repository uses [Git Large File Storage (Git LFS)](https://git-lfs.github.com/) to manage large model checkpoint files.

**To download model checkpoints correctly:**

1. **Install Git LFS** (if not already installed):

    ```sh
    # On Ubuntu/Debian
    sudo apt-get install git-lfs

    # On macOS (with Homebrew)
    brew install git-lfs

    # Or visit: https://git-lfs.github.com/
    ```

2. **Initialize Git LFS** (run once per machine):

    ```sh
    git lfs install
    ```

3. **Clone this repository:**

    ```sh
    git clone https://github.com/rafalkarczewski/diffusion-density.git
    cd diffusion-density
    ```

4. **Fetch all LFS files** (if needed):

    ```sh
    git lfs pull
    ```

> **Note:**  
> If you clone without Git LFS installed, large files (such as model checkpoints) will not be downloaded correctly. After installing Git LFS, you may need to run `git lfs pull` to retrieve these files.

#### Training

We provide CIFAR-10 model checkpoints in the `checkpoints` directory, but also provide a training script that can be used to train a diffusion model from scratch:

```sh
python train.py
```

#### Sampling

The `sampling.py` script contains key algorithms from the first paper [[1]](https://arxiv.org/abs/2411.01293), i.e.
* log-density tracking for SDE sampling (Theorem 1)
* high-density sampling (Algorithm 1)

Running 

```sh
python sampling.py
```
will perform SDE sampling with log-density tracking, and compare with ODE log-density estimates. Furthermore, it performs high-density sampling and shows how the average log-density changes with the threshold.

#### Guidance

The `guidance.py` script contains all key algorithms from the second paper [[2]](https://arxiv.org/abs/2502.05807). Specifically

* Density guidance (Eq23)
* Stochastic density guidance (Eq25)
* Prior Guidance (Section 3)

Running 

```sh
python guidance.py
```
will peform prior guidance, density guidance and stochastic density guidance, and compare log-density estimates.

#### Score alignment

The `score_aligment.py` script contains the implementation of the Score Alignment verification condition (Eq11 in [[2]](https://arxiv.org/abs/2502.05807)) implemented in two ways

* Assuming access to the score function at all `t` (Eq12 in [[2]](https://arxiv.org/abs/2502.05807))
* Without access to the score function apart from `t=T` (Eq13 in [[2]](https://arxiv.org/abs/2502.05807))
> Comparison between the two approaches can be found in Algorithm 1 in [[2]](https://arxiv.org/abs/2502.05807).

Running 

```sh
python score_alignment.py
```
will estimate the score_alignment condition on a sample of latent points using the two methods and compare the results.

### Note on $\lambda=\log\mathrm{SNR}(t)$ parametrization

Following [3], we parametrize our models with $\lambda_t=\log\mathrm{SNR}(t)$ instead of $t$, where $\mathrm{SNR}(t)=\frac{\alpha_t^2}{\sigma_t^2}$. Therefore, the reverse SDE in the $t$-parametrization:

$$dx_t = (f(t)x_t - g^2(t)\nabla \log p_t(x_t))dt + g(t)dW_t, \quad t \in (T, 0),$$

where $f(t) = \frac{d \log \alpha_t}{dt}$, $g^2(t) = -\frac{d \lambda_t}{dt}\sigma_t^2$ is equivalent to

$$dx_\lambda = (\tilde{f}(\lambda)x_\lambda + \sigma_\lambda^2 \nabla \log p_\lambda (x_\lambda))d\lambda + \sigma_\lambda dW_\lambda, \quad \lambda \in (\lambda_{\mathrm{min}}, \lambda_{\mathrm{max}}),$$

where $p_\lambda = p_{t(\lambda)}$, $\tilde{f}(\lambda) = \frac{d \log \alpha_\lambda}{d \lambda}$, $\alpha_\lambda = \alpha_{t(\lambda)}$, $\sigma_\lambda = \sigma_{t(\lambda)}$.

Similarly, the PF-ODE

$$\frac{d x_t}{dt} = f(t)x_t - \frac{1}{2} g^2(t)\nabla \log p_t(x_t), \quad t \in (T, 0)$$

becomes

$$\frac{d x_\lambda}{d\lambda} = \tilde{f}(\lambda)x_\lambda + \frac{1}{2} \sigma_\lambda^2 \nabla \log p_\lambda (x_\lambda), \quad \lambda \in (\lambda_{\mathrm{min}}, \lambda_{\mathrm{max}})$$

#### Examples

For VP-SDE, we have $\alpha_\lambda = \sqrt{\mathrm{sigmoid}(\lambda)}$, $\sigma_\lambda = \sqrt{\mathrm{sigmoid}(-\lambda)}$, and the PF-ODE becomes:

$$\frac{d x_\lambda}{d\lambda} = \frac{1}{2}\mathrm{sigmoid}(-\lambda) (x_\lambda + \nabla \log p_\lambda (x_\lambda)), \quad \lambda \in (\lambda_{\mathrm{min}}, \lambda_{\mathrm{max}})$$

For VE-SDE, we have $\alpha_\lambda = 1$, $\sigma_\lambda = \exp(-\lambda/2)$, and the PF-ODE is

$$\frac{d x_\lambda}{d\lambda} = \frac{1}{2}\exp(-\lambda) \nabla \log p_\lambda (x_\lambda), \quad \lambda \in (\lambda_{\mathrm{min}}, \lambda_{\mathrm{max}})$$

---

[3] Diederik P. Kingma, Ruiqi Gao. Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation (NeurIPS 2023)