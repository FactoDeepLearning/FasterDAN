from torch.optim import Adam
from faster_dan.basic.transforms import aug_config
from faster_dan.OCR.ocr_dataset_manager import OCRDataset, OCRDatasetManager
from faster_dan.OCR.document_OCR.faster_dan.trainer_faster_dan import Manager
from faster_dan.OCR.document_OCR.faster_dan.models_faster_dan import GlobalHTADecoder
from faster_dan.basic.models import FCN_Encoder
from faster_dan.basic.scheduler import exponential_dropout_scheduler, linear_scheduler
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
    model = Manager(params)
    model.load_model()

    model.train()

    # load weights giving best CER on valid set
    model.params["training_params"]["load_epoch"] = "best"
    model.load_model()

    metrics = ["cer", "wer", "time", "map_cer",  "loer", ]

    for dataset_name in params["dataset_params"]["datasets"].keys():
        for set_name in ["test", "valid", "train"]:
            model.predict("{}-{}".format(dataset_name, set_name), [(dataset_name, set_name), ], metrics, output=True)


if __name__ == "__main__":

    dataset_name = "READ_2016"  # ["RIMES", "READ_2016", "MAURDOR_C3", "MAURDOR_C4", "MAURDOR_C3_C4]
    dataset_level = "page"  # ["page", "double_page"]
    dataset_variant = "_sem"

    # max number of lines for synthetic documents
    max_nb_lines = {
        "RIMES": 40,
        "READ_2016": 30,
        "MAURDOR_C3": 40,
        "MAURDOR_C4": 40,
        "MAURDOR_C3_C4": 40,
    }

    params = {
        "dataset_params": {
            "dataset_manager": OCRDatasetManager,
            "dataset_class": OCRDataset,
            "datasets": {
                dataset_name: "../../../Datasets/formatted/{}_{}{}".format(dataset_name, dataset_level, dataset_variant),
            },
            "train": {
                "name": "{}-train".format(dataset_name),
                "datasets": [(dataset_name, "train"), ],
            },
            "valid": {
                "{}-valid".format(dataset_name): [(dataset_name, "valid"), ],
            },
            "config": {
                "balance_datasets": True,  # use same number of images for all datasets (if multiple datasets are used)
                "load_in_memory": True,  # Load all images in CPU memory
                "worker_per_gpu": 4,  # Num of parallel processes per gpu for data loading
                "width_divisor": 8,  # Image width will be divided by 8
                "height_divisor": 32,  # Image height will be divided by 32
                "padding_value": 0,  # Image padding value
                "padding_token": None,  # Label padding value
                "charset_mode": "seq2seq",  # add end-of-transcription ans start-of-transcription tokens to charset
                "constraints": ["add_eot", "add_sot", "fdan_encoding"],  # add end-of-transcription ans start-of-transcription tokens in labels
                "normalize": True,  # Normalize with mean and variance of training dataset
                "preprocessings": [
                    {
                        "type": "to_RGB",
                        # if grayscaled image, produce RGB one (3 channels with same value) otherwise do nothing
                    },
                ],
                "augmentation": aug_config(0.9, 0.1),  # data augmentation config
                "synthetic_data": {
                    "mode": "document",
                    "page_syn_mode": "typed",  # use only synthetic text lines
                    "init_proba": 0.9,  # begin proba to generate synthetic document
                    "end_proba": 0.2,  # end proba to generate synthetic document
                    "num_steps_proba": 200000,  # linearly decrease the percent of synthetic document from 90% to 20% through 200000 samples
                    "proba_scheduler_function": linear_scheduler,  # decrease proba rate linearly
                    "start_scheduler_at_max_line": True,  # start decreasing proba only after curriculum reach max number of lines
                    "dataset_level": dataset_level,
                    "curriculum": True,  # use curriculum learning (slowly increase number of lines per synthetic samples)
                    "crop_curriculum": True,  # during curriculum learning, crop images under the last text line
                    "curr_start": 0,  # start curriculum at iteration
                    "curr_step": 10000,  # interval to increase the number of lines for curriculum learning
                    "min_nb_lines": 1,  # initial number of lines for curriculum learning
                    "max_nb_lines": max_nb_lines[dataset_name],  # maximum number of lines for curriculum learning
                    "padding_value": 255,

                    "max_char_per_line": 100,  # maximum number of characters per line
                    "mix_paragraphs": True,  # generate body from random text lines of different bodies
                    "rimes_sem_order": False,

                    # config for synthetic line generation
                    "config": {
                        "background_color_default": (255, 255, 255),
                        "background_color_eps": 15,
                        "text_color_default": (0, 0, 0),
                        "text_color_eps": 15,
                        "font_size_min": 35,
                        "font_size_max": 45,
                        "color_mode": "RGB",
                        "padding_left_ratio_min": 0.00,
                        "padding_left_ratio_max": 0.05,
                        "padding_right_ratio_min": 0.02,
                        "padding_right_ratio_max": 0.2,
                        "padding_top_ratio_min": 0.02,
                        "padding_top_ratio_max": 0.1,
                        "padding_bottom_ratio_min": 0.02,
                        "padding_bottom_ratio_max": 0.1,
                    },
                }
            }
        },

        "model_params": {
            "models": {
                "encoder": FCN_Encoder,
                "decoder": GlobalHTADecoder,
            },
            "transfer_learning": None,
            # "transfer_learning": {
            #     # model_name: [state_dict_name, checkpoint_path, learnable, strict]
            #     "encoder": ["encoder", "../../line_OCR/ctc/outputs/FCN_Encoder_read_line_syn/checkpoints/best_XXX.pt", True, True],
            #     "decoder": ["decoder", "../../line_OCR/ctc/outputs/FCN_Encoder_read_line_syn/checkpoints/best_XXX.pt", True, False],
            # },
            "transfered_charset": True,  # Transfer learning of the decision layer based on charset of the line HTR model
            "additional_tokens": 1,  # for decision layer = [<eot>, ], only for transfered charset

            "input_channels": 3,  # number of channels of input image
            "dropout": 0.5,  # dropout rate for encoder
            "enc_dim": 256,  # dimension of extracted features
            "nb_layers": 5,  # encoder
            "pe_h_max": 500,  # maximum height for encoder output (for 2D positional embedding)
            "pe_w_max": 1000,  # maximum width for encoder output (for 2D positional embedding)
            "l_max": 15000,  # max predicted sequence (for 1D positional embedding)
            "dec_num_layers": 8,  # number of transformer decoder layers
            "dec_num_heads": 4,  # number of heads in transformer decoder layers
            "dec_res_dropout": 0.1,  # dropout in transformer decoder layers
            "dec_pred_dropout": 0.1,  # dropout rate before decision layer
            "dec_att_dropout": 0.1,  # dropout rate in multi head attention
            "dec_dim_feedforward": 256,  # number of dimension for feedforward layer in transformer decoder layers
            "attention_win": 100,  # length of attention window
            "use_tokens_from_all_lines": True,
            "two_step_pos_enc_mode": "cat",  # ["cat" / "add"]
            "first_pass_factor": 1,
            # Curriculum dropout
            "dropout_scheduler": {
                "function": exponential_dropout_scheduler,
                "T": 5e4,
            }
        },

        "training_params": {
            "output_folder": "fdan_read_single_page",  # folder name for checkpoint and results
            "max_nb_epochs": 50000,  # maximum number of epochs before to stop
            "max_training_time": 3600 * 24 * 1.9,  # maximum time before to stop (in seconds)
            "load_epoch": "last",  # ["best", "last"]: last to continue training, best to evaluate
            "interval_save_weights": None,  # None: keep best and last only
            "batch_size": 1,  # mini-batch size for training
            "valid_batch_size": 4,  # mini-batch size for validation
            "test_batch_size": 1,  # mini-batch size for validation
            "use_amp": True,  # Enable automatic mix-precision
            "nb_gpu": torch.cuda.device_count(),
            "optimizers": {
                "all": {
                    "class": Adam,
                    "args": {
                        "lr": 0.0001,
                        "amsgrad": False,
                    }
                },
            },
            "lr_schedulers": None,  # Learning rate schedulers
            "eval_on_valid": True,  # Whether to eval and logs metrics on validation set during training or not
            "eval_on_valid_interval": 5,  # Interval (in epochs) to evaluate during training
            "focus_metric": "cer",  # Metrics to focus on to determine best epoch
            "expected_metric_value": "low",  # ["high", "low"] What is best for the focus metric value
            "set_name_focus_metric": "{}-valid".format(dataset_name),  # Which dataset to focus on to select best weights
            "train_metrics": ["loss", "cer", "wer", "syn_max_lines"],  # Metrics name for training
            "eval_metrics": ["cer", "wer", "map_cer", "cer_first_pass"],  # Metrics name for evaluation on validation set during training
            "force_cpu": False,  # True for debug purposes
            "max_line_pred": 100,  # max number of predicted lines
            "max_pred_per_line": 150,  # max number of char by predicted lines
            # Keep teacher forcing rate to 20% during whole training
            "teacher_forcing_scheduler": {
                "min_error_rate": 0.2,
                "max_error_rate": 0.2,
                "total_num_steps": 5e4
            },
        },
    }

    syn_config = params["dataset_params"]["config"]["synthetic_data"]
    params["training_params"]["start_valid_from_steps"] = syn_config["curr_start"] + syn_config["curr_step"] * (syn_config["max_nb_lines"] - syn_config["min_nb_lines"])

    train_and_test(0, params)