# ATVFI
This repository is for paper "Arbitrary Timestep Video Frame Interpolation with Time-dependent Decoding".

## Dataset preparation

Download [GoPro dataset](https://seungjunnah.github.io/Datasets/gopro), and unzip it into path `./db/gopro`.
There should be 2 folders named "train" and "test" in it, and inside each of them, there are folders of frame images.
Download [Vimeo90k triplet dataset](http://toflow.csail.mit.edu/) and unzip into path`./db/vimeo_triplet`.
Download [UCF from QVI](https://github.com/xuxy09/QVI) and unzip into path `./test_input/ucf_extracted`.
Download [Middlebury](https://vision.middlebury.edu/flow/data/) and unzip into path `./test_input/middlebury_others` and `./test_input/middlebury_eval`.

## Training

First, install the following packages into your Python3 environment. There should be no strict requirements on the version of the packages.
- pytorch
- torchvision
- numpy
- cupy
- Pillow
- opencv
- tqdm

As an example, if you have installed [conda](https://conda.io), then you may create a new environment by run
```bash
conda create -n atvfi pytorch torchvision numpy cupy Pillow opencv tqdm -c conda-forge
```
and make sure you installed CUDA-enabled version of PyTorch (by examining whether "cu" appears in the version of pytorch package
in the installation plan of conda).

Then, run `python run_train.py --out_dir ./record/baseline_atvfi --gopro_use_augment --gpu_id 0 `to train the model.
The trained model files will be stored in the given directory `./record/baseline_atvfi`.
For other options, you may refer to the help messages in the code.

## Evaluation

After training, run `python evaluation.py` to show evaluation results. If you trained the model with command above, 
the default parameters would do the work and no extra parameters are needed.
