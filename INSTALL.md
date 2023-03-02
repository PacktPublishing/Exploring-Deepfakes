# packt
Code and data for the Hands on Deepfakes book for Packt publishing.

## Installation:

We highly recommend using Anaconda/Miniconda from https://www.anaconda.com/ or https://conda.io/miniconda.html in order to create a separate python environment for each project you're working on.  This lets each project keep their own separate collection of Python libraries.  This helps you keep all the projects working by preventing interference between varied requirements.

Anaconda also provides their own package management system (conda) which lets you install different versions of libraries or tools and can even automate the installation of Cuda -- the Nvidia GPU acceleration software.  This makes running with GPU acceleration much easier.

Once you've installed anaconda and downloaded this repo (either by clicking )

## Setup 
### Anaconda

You can setup a contained Conda environment using the provided `conda-<version>.yml` for this project by running one of the following commands from an Anaconda Command Prompt:

### Nvidia:
```
conda env create -f conda-nvidia.yml
```

### CPU:
```
conda env create -f conda-cpu.yml
```

Once Conda has finished setting up the environment, enter one of the following commands to activate it before continuing on to the next section:

### Nvidia:
```
conda activate packt-deepfakes-nvidia
```

### CPU:
```
conda activate packt-deepfakes-cpu
```

## Usage

Each of the Chapters have detailed help information. To see them just add --help as an argument to the python file.  For example:

```
python C5-face_detection.py --help
```

This will list all the options and settings you can pass.
