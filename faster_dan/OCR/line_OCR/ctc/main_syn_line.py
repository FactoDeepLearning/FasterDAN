from faster_dan.OCR.line_OCR.ctc.trainer_line_ctc import TrainerLineCTC
from faster_dan.OCR.line_OCR.ctc.models_line_ctc import Decoder
from faster_dan.basic.models import FCN_Encoder
from faster_dan.basic.transforms import line_aug_config
from faster_dan.basic.scheduler import exponential_dropout_scheduler, linear_scheduler
from faster_dan.OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
from torch.optim import Adam
import torch
import numpy as np
import random


def train_and_test(rank, params):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    params["training_params"]["ddp_rank"] = rank
    model = TrainerLineCTC(params)
    model.load_model()

    model.generate_syn_dataset("READ_2016_line_syn")


def main():
    dataset_name = "READ_2016"  # ["RIMES", "READ_2016", "MAURDOR_C3", "MAURDOR_C4", "MAURDOR_C3_C4"]
    dataset_level = "page"  # ["page", "double_page"]

    params = {
        "dataset_params": {
            "dataset_manager": OCRDatasetManager,
            "dataset_class": OCRDataset,
            "datasets": {
                dataset_name: "../../../Datasets/formatted/{}_{}".format(dataset_name, dataset_level),
            },
            "train": {
                "name": "{}-train".format(dataset_name),
                "datasets": [(dataset_name, "train"), ],
            },
            "valid": {
                "{}-valid".format(dataset_name): [(dataset_name, "valid"), ],
            },
            "config": {
                "load_in_memory": True,  # load the whole dataset in RAM
                "worker_per_gpu": 8,
                "width_divisor": 8,  # Image width will be divided by 8
                "height_divisor": 32,  # Image height will be divided by 32
                "padding_value": 0,  # Image padding value
                "padding_token": 1000,  # Label padding value (None: default value is chosen)
                "padding_mode": "br",
                "charset_mode": "CTC",  # add blank token
                "constraints": ["CTC_line", ],  # Padding for CTC requirements if necessary
                "normalize": True,
                # pad (bottom/right) to max image size in training dataset
                "padding": {
                    "min_height": "max",
                    "min_width": "max",
                    "min_pad": None,
                    "max_pad": None,
                    "mode": "br",
                    "train_only": False,
                },
                "preprocessings": [
                    {
                        "type": "to_RGB",
                        # if grayscale image, produce RGB one (3 channels with same value) otherwise do nothing
                    },
                ],
                "augmentation": line_aug_config(0.9, 0.1),  # data augmentation config
                "synthetic_data": {
                    "mode": "line_hw_to_printed",
                    # only use synthetic images for training
                    "init_proba": 1,
                    "end_proba": 1,
                    "num_steps_proba": 1e5,
                    "proba_scheduler_function": linear_scheduler,
                    "config": {
                        "background_color_default": (255, 255, 255),
                        "background_color_eps": 15,
                        "text_color_default": (0, 0, 0),
                        "text_color_eps": 15,
                        "font_size_min": 30,
                        "font_size_max": 50,
                        "color_mode": "RGB",
                        "padding_left_ratio_min": 0.02,
                        "padding_left_ratio_max": 0.1,
                        "padding_right_ratio_min": 0.02,
                        "padding_right_ratio_max": 0.1,
                        "padding_top_ratio_min": 0.02,
                        "padding_top_ratio_max": 0.2,
                        "padding_bottom_ratio_min": 0.02,
                        "padding_bottom_ratio_max": 0.2,
                        "max_width": 1500,
                    },
                },
                "charset": [],
            }
        },

        "model_params": {
            # Model classes to use for each module
            "models": {
                "encoder": FCN_Encoder,
                "decoder": Decoder,
            },
            "transfer_learning": None,
            "input_channels": 3,  # 1 for grayscale images, 3 for RGB ones (or grayscale as RGB)
            "enc_size": 256,  # encoder model size
            # curriculum dropout
            "dropout_scheduler": {
                "function": exponential_dropout_scheduler,
                "T": 5e4,
            },
            "dropout": 0.5,  # dropout for encoder
        },

        "training_params": {
            "output_folder": "FCN_Encoder_read_line_syn",  # folder names for logs and weigths
            "max_nb_epochs": 10000,  # max number of epochs for the training
            "max_training_time": 3600 * 24 * 1.9,  # max training time limit (in seconds)
            "load_epoch": "last",  # ["best", "last"], to load weights from best epoch or last trained epoch
            "interval_save_weights": None,  # None: keep best and last only
            "use_ddp": False,  # Use DistributedDataParallel
            "use_amp": True,  # Enable automatic mix-precision
            "nb_gpu": torch.cuda.device_count(),
            "batch_size": 16,  # mini-batch size per GPU
            "optimizers": {
                "all": {
                    "class": Adam,
                    "args": {
                        "lr": 0.0001,
                        "amsgrad": False,
                    }
                }
            },
            "lr_schedulers": None,  # Learning rate schedulers
            "eval_on_valid": True,  # Whether to eval and logs metrics on validation set during training or not
            "eval_on_valid_interval": 2,  # Interval (in epochs) to evaluate during training
            "focus_metric": "cer",  # Metrics to focus on to determine best epoch
            "expected_metric_value": "low",  # ["high", "low"] What is best for the focus metric value
            "set_name_focus_metric": "{}-valid".format(dataset_name),  # Which dataset to focus on to select best weights
            "train_metrics": ["loss", "cer", "wer"],  # Metrics name for training
            "eval_metrics": ["loss", "cer", "wer"],  # Metrics name for evaluation on validation set during training
            "force_cpu": False,  # True for debug purposes to run on cpu only
        },
    }

    train_and_test(0, params)


if __name__ == "__main__":
    main()