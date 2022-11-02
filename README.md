# supcon_wmpc
PyTorch implementation of the model described in the paper Supervised Contrastive Learning for Wafer Map Pattern Classification

## Components
- **data/** - directory where the dataset should be placed
- **src/pre_process.ipynb** - notebook for processing the raw dataset
- **src/models/vgg.py** - VGG16 model architecture, modified to perform contrastive learning
- **src/utils/dataset.py** - PyTorch dataset class for both datasets
- **src/utils/tools.py** - functions and class for augmentation, result parsing and metrics
- **src/loader.py** - Pytorch data loader
- **src/loss.py** - Supervised Contrastive Loss of proposed method
- **src/main.py** - main module
- **src/option.py** - argument parser
- **src/test.py** - inference function
- **src/train.py** - train function

## Data
- The datasets used in the paper can be downloaded from
    - **WM811k** - https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
    - **MixedWM38** - https://github.com/Junliangwangdhu/WaferMap

## Dependencies
- **Python**
- **PyTorch**
- **NumPy**
- **Tensorboard**
- **Pandas**
- **Scikit-learn**
- **Scikit-image**

## Example Run Code
- **WM-811k**

        python src/main.py --dataset wm811k --batch_size 128 --set_size 10000 --epochs 500 --patience 50 --exp_id 1234

- **WM-811k**

        python src/main.py --dataset mixedwm38 --batch_size 128 --set_size 10000 --epochs 500 --patience 50 --exp_id 1234