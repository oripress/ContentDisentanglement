## Emerging Disentanglement in Auto-Encoder Based Unsupervised Image Content Transfer

PyTorch implementation of "Emerging Disentanglement in Auto-Encoder Based Unsupervised Image Content Transfer" ([arxiv]()).


The network learns to disentangle content between a set and its subset. For example, given a set of people with glasses, and a set of people without, the network
learns to map to decompose a face into 2 parts: one that contains information about glasses and one that contains information about everything else.

We can then transfer one person's glasses to many different people. In the image below, the glasses from the people in
the left column are transferred to the people in the top row.
<img src="images/gls_mat_clear.png" width="1200px">

We can also do this for people who already have glasses, i.e. we replace their glasses with another pair:
<img src="images/gls_swap_clear.png" width="600px">


## Prerequisites
- Python 2.7 / 3.6
- Pytorch 0.4
- [argparse](https://docs.python.org/2/howto/argparse.html)
- [Pillow](https://pillow.readthedocs.io/en/5.3.x/)

## Get Started:
First, clone this repository by running:
```
git clone https://github.com/oripress/ContentDisentanglement
```
### Download and Prepare the Data
Download the dataset by running the following command:
```
bash celeba_downloader.sh
```
Contrary to the notation used in the paper, A is the larger set, for example, A is people with glasses and B is people without.
You can use the provided script ```preprocess.py``` to split celebA into the above format (with A and B based on the attribute of your choosing).
For example, you can run the script using the following command:
```
python preprocess.py --root ./img_align_celeba --attributes ./list_attr_celeba.txt --dest ./glasses_train
```
You can also use your own custom dataset, as long as it adheres to the following format:
```
root/
     trainA/
     trainB/
     testA/
     testB/
```
You can then run the preprocessing in the following manner:
```
python preprocess.py --root ./custom_dataset --dest ./custom_train --folders
```

### To Train
Run ```train.py```. You can use the following example to run
```
python train.py --root ./glasses_data --out ./glasses_experiment --sep 25 --discweight 0.001
```

### To Resume Training
Run ```train.py```. You can use the following example to run
```
python train.py --root ./glasses_data --out ./glasses_experiment --load ./glasses_experiment --sep 25 --discweight 0.001
```

### To Evaluate
Run ```eval.py```. You can use the following example to run
```
python eval.py --dataroot ./glasses_data --out ./glasses_eval --sep 25 --num_display 10
```

### Acknowledgements
The implementation is based on the architecture of [Fader Networks](https://github.com/facebookresearch/FaderNetworks).
Some of the code is also based on the implementations of [MUNIT](https://github.com/NVlabs/MUNIT) and [DRIT](https://github.com/HsinYingLee/DRIT), and [StarGAN](https://github.com/yunjey/StarGAN).
