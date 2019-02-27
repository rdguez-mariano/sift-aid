# AID : An affine invariant descriptor for SIFT

This repository implements SIFT-AID, an affine invariant method for matching two images. The companion paper can be found [here](https://rdguez-mariano.github.io/pages/sift-aid).

## Prerequisites

Before starting to use SIFT-AID be sure of the following.

##### Creating a conda environment for SIFT-AID

```bash
conda create --name aid python=3.5.4

source activate aid

pip install --upgrade pip
pip install -r requirements.txt --ignore-installed
```

##### Compiling the C++ library

```bash
mkdir -p build && cd build && cmake .. && make
```

##### Deactivate the environment

```bash
conda deactivate
```

##### Delete the environment

```bash
conda-env remove -n aid
```

##### Possible install errors

If AttributeError: module 'cv2.cv2' has no attribute 'xfeatures2d' reinstall opencv-contrib

```bash
pip uninstall opencv-contrib-python
pip install opencv-python==3.4.2.16
```

## Launching SIFT-AID

The function `siftAID` in libLocalDesc.py will execute SIFT-AID with the network descritor loaded from [model.AID_simCos_BigDesc_dropout_75.hdf5](model-data/model.AID_simCos_BigDesc_dropout_75.hdf5).

```python
from libLocalDesc import *

img1 = cv2.cvtColor(cv2.imread('./acc-test/coca.1.png'),cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(cv2.imread('./acc-test/coca.2.png'),cv2.COLOR_BGR2GRAY)

_, good_HC, ET_KP, ET_M = siftAID(img1,img2, MatchingThres = 4000, Simi = 'SignProx', Visual=True)
print("SIFT-AID --> FilteredMatches = %d, KeypointsTime = %3.3f, MatchingTime = %3.3f" %(len(good_HC),ET_KP,ET_M))
```
An example of its use can be found in [gen-ICIP19-TeaserImages.py](py-tools/gen-ICIP19-TeaserImages.py).

## Training the descriptor network

##### Creating the needed directory structure

For training you need to specify three image datasets. These images will be used to generate optical affine views. Create default folders by typing:

```bash
mkdir -p imgs-train && mkdir -p imgs-val && mkdir -p imgs-test
```

Now, for example, you can download correspondent datasets from [MS-COCO](http://cocodataset.org) into the default folders.

Also, we need to create the following folders in order to storage tensorboard summaries and the resulting output images and data:

```bash
mkdir -p summaries && mkdir -p temp
```

##### Training

Once image datasets are available in *imgs-train* and *imgs-val* you can train the network.

```bash
python AID-train-model.py
```

Generated pairs of patches will be saved into their respective folders, e.g. (*db-gen-train-60*, *db-gen-val-60*). Those folders will have scattered files corresponding to patchs pairs. To create blocks of data that can be quickly (and automatically) reused by the trainer please launch also:

```bash
python py-tools/in2blocks.py
```

##### Some key variables in AID-train-model.py

Maximal viewpoint angle of the affine maps to be optically simulated.

```python
DegMax = 60
```

Set this variable to `False` if you want to really train. Use for test when modifying the code.

```python
Debug = True
```

Set this to `True` if no blocks of patch-data have been created yet. It will use all your CPU power for affine simulations. Deactivate back when you have created a sufficient amount of data blocks with [in2blocks.py](py-tools/in2blocks.py).

```python
Parallel = False
```

Set this variable to `True` if you want to cycle over blocks of patch-data.

```python
DoBigEpochs = True
```

Depending on you GPU, select the percentage of GPU memory to be used when training.

```python
config.gpu_options.per_process_gpu_memory_fraction = 0.1
```

## Summaries with tensorboard

Show all images (slider step = 1)

```bash
tensorboard --logdir="./" --samples_per_plugin="images=0"
```

If tensorboard crashes, try reinstalling it first !!! If locale error (unsupported locale) setting, do:

```bash
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
sudo dpkg-reconfigure locales
```

## Authors

* **Mariano Rodríguez** - [web page](https://rdguez-mariano.github.io/)
* **Gabriele Facciolo**
* **Rafael Grompone Von Gioi**
* **Pablo Musé**
* **Jean-Michel Morel** - [web page](https://sites.google.com/site/jeanmichelmorelcmlaenscachan/)
* **Julie Delon** - [web page](https://delon.wp.imt.fr/)


## License

The code is distributed under the permissive MIT License - see the [LICENSE](LICENSE) file for more details.

## Acknowledgements

##### Contributors

* **Jéremy Anger** [(web page)](http://dev.ipol.im/~anger/) and **Axel Davy** [(web page)](http://dev.ipol.im/~adavy/) pointed out the need of CPU vector operations to optimize time computations.

##### This project can optionally

* call libOrsa, libMatch and libNumerics. Copyright (C) 2007-2010, Lionel Moisan, distributed under the BSD license.
* call libUSAC. Copyright (c) 2012 University of North Carolina at Chapel Hill / See its [web page](http://www.cs.unc.edu/~rraguram/usac/) to see their specific associated licence.

## Github repository

<https://github.com/rdguez-mariano/sift-aid>
