# ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection: Task 3

This project contains the source code used for the RECOD Titan's submission to ISIC
2018: Skin Lesion Analysis Towards Melanoma Detection (Task 3). This project was
forked from the [source](https://github.com/fabioperez/skin-data-augmentation)
of the paper 'Data Augmentation for Skin Lesion Analysis'.

## Project setup

1. Install OpenCV with `pip3 install opencv-python`.
2. Run `pip3 install -r requirements.txt`.
3. Download data from [ISIC 2017: Skin Lesion Analysis Towards Melanoma
   Detection](https://challenge.kitware.com/#challenge/583f126bcad3a51cc66c8d9a).


## Train

The project uses [Sacred](http://sacred.readthedocs.io) to organize the
experiments. The main script for training is in the `train.py` file. Check the
available settings by running `python3 train.py print_config`.

Possible values for `model_names`: `resnet152`, `inceptionv4`, `densenet161`.

#### Example: training ResNet-152 with split 1

```
TRAIN_ROOT=/path/to/dataset/images
TRAIN_CSV=splits/split_task3_train_full_1.txt
VAL_ROOT=/path/to/dataset/images
VAL_CSV=splits/split_task3_validation.txt

python3 train.py with \
    train_root=$TRAIN_ROOT train_csv=$TRAIN_CSV \
    val_root=$VAL_ROOT val_csv=$VAL_CSV \
    model_name='resnet152' \
    'aug={"color_contrast": 0.3, "color_saturation": 0.3, "color_brightness": 0.3, "color_hue": 0.1, "rotation": 90, "scale": (0.8, 1.2), "shear": 20, "vflip": True, "hflip": True, "random_crop": True}' \
    weighted_loss=True \
    --name resnet152-split-1
```

If everything goes well, Sacred will create a directory with a unique ID inside
`results` (e.g. `results/1` for the first run). Inside this directory, you will
find:

* `config.json`: Sacred configuration used in training.
* `cout.txt`: Entire stdout produced during the training.
* `run.json`: General metadata of the training.
* `train.csv`: CSV with metrics on train set.
* `val.csv`: CSV with metrics on validation set.
* `checkpoints/model_best.pth`: model with the best validation AUC.
* `checkpoints/model_last.pth`: model as in the last epoch.

### Telegram API

If you want to monitor the experiments with Telegram (receive a message when
the experiments start, finish, or fail), create a file `telegram.json` at the
root of the project:

```
$ cat telegram.json
{
    "token": "00000000:XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
    "chat_id": "00000000"
}
```

To configure the Telegram API, check
[this](https://stackoverflow.com/questions/32423837/telegram-bot-how-to-get-a-group-chat-id).


## Test

Each model file (i.e, `model_best.pth` or `model_last.pth`) contains the
PyTorch model, weights, and augmentation configuration (accessed through
`model.aug_params`). To load the model, use `torch.load`.

The `test.py` file will automatically infer the augmentation settings from the
model. Run `python3 test.py --help` to check all available options.

#### Example: get predictions for test set

```
TEST_ROOT=/path/to/dataset/images
TEST_CSV=splits/split_task3_testsubmission_challenge.txt
python3 test.py results/<SACRED_ID>/checkpoints/model_best.pth $TEST_ROOT $TEST_CSV -n 128 --output results_test.csv
```
