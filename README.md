# Code: Critical initialisation for deep signal propagation in noisy rectifier neural networks

This repository provides the code to reproduce all the results in the paper: "Critical initialisation for deep signal propagation in noisy rectifier neural networks" (NIPS 2018).

The code was written by Elan Van Biljon, Arnu Pretorius and Herman Kamper. Large portions of the code was originally adapted from code that was made available in Poole et al. (2016) at https://github.com/ganguli-lab/deepchaos.

---

![Alt Text](https://github.com/ElanVB/noisy_signal_prop/blob/master/src/figures/example_figures-1.png)

---

## Basic steps for Figures 2-5

To reproduce Figures 2-5 in the paper please follow the steps below.

#### Step 1. Install [Conda](https://conda.io/docs/user-guide/install/index.html).

#### Step 2. Clone the research code repository. 

```bash
git clone https://github.com/ElanVB/noisy_signal_prop.git
```

#### Step 3. Activate the environment

```bash
cd noisy_signal_prop
conda env create -f environments/simple_env_gpu.yml
source activate noisy_signal_prop
```

If your machine does not have a GPU, please use `simple_env_cpu.yml` instead. Furthermore, full specs can be found in `specific_env.yml`.

#### Step 4. Run code in notebooks

Launch Jupyter server

```bash
jupyter notebook
```

and run the cells in the notebook corresponding to the Figure in the paper you wish to reproduce (e.g. `Figure_2_deep_noisy_signal_prop.ipynb`).

## Steps for larger scale experiments in Figure 6 (GPU required): 

Below are the instructions to reproduce the plots in Figure 6 using a docker image and the notebook provided.

#### Step 1. Install [Docker](https://docs.docker.com/engine/installation/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

#### Step 2. Obtain the research environment image from [Docker Hub](https://hub.docker.com/r/ufoym/deepo/).

```bash
docker pull ufoym/deepo:tensorflow
```
#### Step 3. Clone the research code repository. 
```bash
git clone https://github.com/ElanVB/noisy_signal_prop.git
```

#### Step 4. Generate experimental results (**Warning: this may take several hours to run.**)

```bash
cd noisy_signal_prop/src
docker run --runtime=nvidia -v "$(pwd)":/experiment -it ufoym/deepo:tensorflow bash experiment/start.sh
```

#### Step 5. Run plotting code in the notebook

Launch Jupyter server 

```bash
cd ..
jupyter notebook
```

and run the cells in the notebook `Figure_6_depth_scales_mnist_cifar10.ipynb`. 

### References

B. Poole, S. Lahiri, M. Raghu, J. Sohl-Dickstein, and S. Ganguli. Exponential expressivity in deep neural networks through transient chaos. Neural Information Processing Systems, 2016.

### NOTE

**This repository is still under construction, while all the code necessary to reproduce results in the paper is present, much of the supporting code for the notebooks is not yet in a user friendly state.**
**Expect this to change soon.** 
