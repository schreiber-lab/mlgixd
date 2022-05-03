# _mlgixd_
## Deep learning-based feature detection for GIXD data

The package provides the source code used in the following scientific publication: 

Starostin, V., Munteanu, V., Greco, A. et al. Tracking perovskite crystallization via deep learning-based feature detection on 2D X-ray scattering data. npj Comput Mater 8, 101 (2022). https://doi.org/10.1038/s41524-022-00778-8


## Installation

### Requirements

Python 3.6 or above and CUDA 10.2 or above are required.
It is also recommended installing PyTorch with torchvision by following 
the instructions from the official website:
https://pytorch.org/get-started/locally/ before installing
the _mlgixd_ package.

Other python dependencies can be installed automatically during the next step:
- numpy
- scipy
- tqdm
- PyYAML
- scikit-image

### Install the package 

To install the repository locally, execute the following command in the terminal to clone the
repository and install it via pip:

```shell
git clone git@github.com:schreiber-lab/mlgixd.git && cd mlgixd && pip install .
```

or if pip is not available:

```shell
git clone git@github.com:schreiber-lab/mlgixd.git && cd mlgixd && python setup.py install
```

To test that the package is installed correctly, execute in the 
terminal from the package folder:

```shell
mlgixd config/test_config.yaml
```
or
```shell
python -m mlgixd config/test_config.yaml
```

The command should start short model training, 
testing and saving to a file without errors. 


## Train & test the models 

To train and test one of the implemented models, 
use the corresponding configuration file
provided with the package (for instance, _config/our_model.yaml_):

```shell
mlgixd config/our_model.yaml
```

The results will be saved to a file in _saved_models_ folder specified in the configuration file 
 and can be opened in python:
```python
import torch

results = torch.load('saved_models/our_model.pt')
```

## Authors 

- Vladimir Starostin (Institute of Applied Physics, University of Tübingen)
- Valentin Munteanu (Institute of Applied Physics, University of Tübingen)
- Alessandro Greco (Institute of Applied Physics, University of Tübingen)
- Ekaterina Kneschaurek (Institute of Applied Physics, University of Tübingen)
- Alina Pleli (Institute of Applied Physics, University of Tübingen)
- Florian Bertram (Deutsches Elektronen-Synchrotron DESY)
- Alexander Gerlach (Institute of Applied Physics, University of Tübingen)
- Alexander Hinderhofer (Institute of Applied Physics, University of Tübingen)
- Frank Schreiber (Institute of Applied Physics, University of Tübingen)
