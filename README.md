# Swin-Unet for Lung Segmentation

The code is reproduced from the repository (https://github.com/HuCaoFighting/Swin-Unet) for segmentation of lung based on a dataset from 

## 1. Download pre-trained swin transformer model (Swin-T)
* [Get pre-trained model in this link] (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/"

## 2. Prepare data


## 3. Environment

- Please prepare an environment with python=3.7, and then use the following command for the dependencies.

```bash
pip install torch torchvision numpy tqdm tensorboard tensorboardX scipy h5py timm einops opencv-python scikit-learn six yacs
```
## 4. Train/Test


- A trained ckpt from scratch is saved [here](assets/ckpt/epoch_100th.pth), which is trained based on 4 batch-size for 100 epochs with a step scheduler lr decent strategy.

- Or you can train it on your own:

```bash
python train.py --root_path your DATA_DIR --batch_size 24 --max_epochs 100 --base_lr 0.001
```

- Test 

```bash
python test.py --root_path your DATA_DIR 
```
- You can check the qualitative analysis in the tensorboard. (The parameter samples_per_plugin is for visualizing more images or scalars)

```bash
tensorboard --logdir your TENSORBOARD_LOG_DIR --port 6666 --samples_per_plugin=scalars=500,images=20
```
## References
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)