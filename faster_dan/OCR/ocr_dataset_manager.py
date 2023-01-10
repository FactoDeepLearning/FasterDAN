from faster_dan.basic.generic_dataset_manager import DatasetManager, GenericDataset
from faster_dan.basic.utils import pad_images, pad_image_width_right, resize_max, pad_image_width_random, pad_sequences_1D, pad_image
from faster_dan.basic.utils import randint, rand, rand_uniform
from faster_dan.basic.generic_dataset_manager import apply_preprocessing
from faster_dan.Datasets.dataset_formatters.read2016_formatter import SEM_TOKENS as READ_SEM_TOKENS
from faster_dan.Datasets.dataset_formatters.rimes_formatter import order_text_regions as order_text_regions_rimes
from faster_dan.Datasets.dataset_formatters.rimes_formatter import order_text_regions_sem as order_text_regions_rimes_sem
from faster_dan.Datasets.dataset_formatters.rimes_formatter import SEM_TOKENS as RIMES_SEM_TOKENS
from faster_dan.Datasets.dataset_formatters.rimes_formatter import SEM_MATCHING_TOKENS_STR as RIMES_MATCHING_TOKENS_STR
from faster_dan.OCR.ocr_utils import LM_str_to_ind
import numpy.random
import random
import cv2
import os
import copy
import pickle
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from faster_dan.basic.transforms import RandomRotation, apply_transform, Tightening
from fontTools.ttLib import TTFont


class OCRDatasetManager(DatasetManager):
    """
    Specific class to handle OCR/HTR tasks
    """

    def __init__(self, params):
        super(OCRDatasetManager, self).__init__(params)

        self.charset = params["charset"] if "charset" in params else self.get_merged_charsets()
        self.max_pred_ind = len(self.charset)

        if "synthetic_data" in self.params["config"] and self.params["config"]["synthetic_data"] and "config" in self.params["config"]["synthetic_data"]:
            self.char_only_set = self.charset.copy()
            self.sem_tokens = list()
            for token_dict in [RIMES_SEM_TOKENS, READ_SEM_TOKENS]:
                for char in token_dict.values():
                    if char in self.char_only_set:
                        self.char_only_set.remove(char)
                        self.sem_tokens.append(char)
                    if char.upper() in self.char_only_set:
                        self.char_only_set.remove(char.upper())
                        self.sem_tokens.append(char.upper())
            assert len(np.unique(self.sem_tokens)) == len(np.array(self.sem_tokens)), "Duplicated semantic tokens from different datasets"
            for token in ["\n", ]:
                if token in self.char_only_set:
                    self.char_only_set.remove(token)
            self.params["config"]["synthetic_data"]["config"]["valid_fonts"] = get_valid_fonts(self.char_only_set)

        if "new_tokens" in params:
            self.charset = sorted(list(set(self.charset).union(set(params["new_tokens"]))))

        self.tokens = {
            "pad": params["config"]["padding_token"],
        }
        if self.params["config"]["charset_mode"].lower() == "ctc":
            self.tokens["blank"] = self.max_pred_ind
            self.max_pred_ind += 1
        elif self.params["config"]["charset_mode"] == "seq2seq":
            self.tokens["end"] = self.max_pred_ind
            self.max_pred_ind += 1
            self.tokens["start"] = self.max_pred_ind
            self.max_pred_ind += 1
            if "pad_label_with_end_token" in self.params["config"].keys() and self.params["config"]["pad_label_with_end_token"]:
                self.params["config"]["padding_token"] = self.tokens["end"]
            else:
                self.params["config"]["padding_token"] = self.tokens["pad"]

        if "add_eot" in self.params["config"]["constraints"] and "end" not in self.tokens:
            self.tokens["end"] = self.max_pred_ind
            self.max_pred_ind += 1
        if "add_sot" in self.params["config"]["constraints"] and "start" not in self.tokens:
            self.tokens["start"] = self.max_pred_ind
            self.max_pred_ind += 1
        if not self.tokens["pad"]:
                self.tokens["pad"] = self.max_pred_ind
                self.max_pred_ind += 1
        self.params["config"]["padding_token"] = self.tokens["pad"]

    def get_merged_charsets(self):
        """
        Merge the charset of the different datasets used
        """
        datasets = self.params["datasets"]
        charset = set()
        for key in datasets.keys():
            with open(os.path.join(datasets[key], "labels.pkl"), "rb") as f:
                info = pickle.load(f)
                charset = charset.union(set(info["charset"]))
        if "\n" in charset and "remove_linebreaks" in self.params["config"]["constraints"]:
            charset.remove("\n")
        if "" in charset:
            charset.remove("")
        return sorted(list(charset))

    def apply_specific_treatment_after_dataset_loading(self, dataset):
        dataset.charset = self.charset
        if hasattr(self, "char_only_set"):
            dataset.char_only_set = self.char_only_set
        if hasattr(self, "sem_tokens"):
            dataset.sem_tokens = self.sem_tokens
        dataset.tokens = self.tokens
        dataset.convert_labels()
        if "READ_2016" in dataset.name and "augmentation" in dataset.params["config"] and dataset.params["config"]["augmentation"]:
            dataset.params["config"]["augmentation"]["fill_value"] = tuple([int(i) for i in dataset.mean])
        if "padding" in dataset.params["config"] and dataset.params["config"]["padding"]["min_height"] == "max":
            dataset.params["config"]["padding"]["min_height"] = max([s["img"].shape[0] for s in self.train_dataset.samples])
        if "padding" in dataset.params["config"] and dataset.params["config"]["padding"]["min_width"] == "max":
            dataset.params["config"]["padding"]["min_width"] = max([s["img"].shape[1] for s in self.train_dataset.samples])


class OCRDataset(GenericDataset):
    """
    Specific class to handle OCR/HTR datasets
    """

    def __init__(self, params, set_name, custom_name, paths_and_sets):
        super(OCRDataset, self).__init__(params, set_name, custom_name, paths_and_sets)
        self.charset = None
        self.tokens = None
        self.reduce_dims_factor = np.array([params["config"]["height_divisor"], params["config"]["width_divisor"], 1])
        self.collate_function = OCRCollateFunction
        self.synthetic_id = 0

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.samples[idx])

        if not self.load_in_memory:
            sample["img"] = self.get_sample_img(idx)
            sample = apply_preprocessing(sample, self.params["config"]["preprocessings"])

        if "synthetic_data" in self.params["config"] and self.params["config"]["synthetic_data"] and self.set_name == "train":
            sample = self.generate_synthetic_data(sample)

        # Data augmentation
        sample["img"], sample["applied_da"] = self.apply_data_augmentation(sample["img"])

        if "max_size" in self.params["config"] and self.params["config"]["max_size"]:
            max_ratio = max(sample["img"].shape[0] / self.params["config"]["max_size"]["max_height"], sample["img"].shape[1] / self.params["config"]["max_size"]["max_width"])
            if max_ratio > 1:
                new_h, new_w = int(np.ceil(sample["img"].shape[0] / max_ratio)), int(np.ceil(sample["img"].shape[1] / max_ratio))
                sample["img"] = cv2.resize(sample["img"], (new_w, new_h))

        # Normalization if requested
        if "normalize" in self.params["config"] and self.params["config"]["normalize"]:
            sample["img"] = (sample["img"] - self.mean) / self.std

        sample["img_shape"] = sample["img"].shape
        sample["img_reduced_shape"] = np.ceil(sample["img_shape"] / self.reduce_dims_factor).astype(int)

        # Padding to handle CTC requirements
        if self.set_name == "train":
            max_label_len = 0
            height = 1
            ctc_padding = False
            if "CTC_line" in self.params["config"]["constraints"]:
                max_label_len = sample["label_len"]
                ctc_padding = True
            if ctc_padding and 2 * max_label_len + 1 > sample["img_reduced_shape"][1]*height:
                sample["img"] = pad_image_width_right(sample["img"], int(np.ceil((2 * max_label_len + 1) / height) * self.reduce_dims_factor[1]), self.padding_value)
                sample["img_shape"] = sample["img"].shape
                sample["img_reduced_shape"] = np.ceil(sample["img_shape"] / self.reduce_dims_factor).astype(int)
            sample["img_reduced_shape"] = [max(1, t) for t in sample["img_reduced_shape"]]

        sample["img_position"] = [[0, sample["img_shape"][0]], [0, sample["img_shape"][1]]]
        # Padding constraints to handle model needs
        if "padding" in self.params["config"] and self.params["config"]["padding"]:
            if self.set_name == "train" or not self.params["config"]["padding"]["train_only"]:
                min_pad = self.params["config"]["padding"]["min_pad"]
                max_pad = self.params["config"]["padding"]["max_pad"]
                pad_width = randint(min_pad, max_pad) if min_pad is not None and max_pad is not None else None
                pad_height = randint(min_pad, max_pad) if min_pad is not None and max_pad is not None else None

                sample["img"], sample["img_position"] = pad_image(sample["img"], padding_value=self.padding_value,
                                          new_width=self.params["config"]["padding"]["min_width"],
                                          new_height=self.params["config"]["padding"]["min_height"],
                                          pad_width=pad_width,
                                          pad_height=pad_height,
                                          padding_mode=self.params["config"]["padding"]["mode"],
                                          return_position=True)
        sample["img_reduced_position"] = [np.ceil(p / factor).astype(int) for p, factor in zip(sample["img_position"], self.reduce_dims_factor[:2])]
        return sample

    def get_charset(self):
        charset = set()
        for i in range(len(self.samples)):
            charset = charset.union(set(self.samples[i]["label"]))
        return charset

    def convert_labels(self):
        """
        Label str to token at character level
        """
        for i in range(len(self.samples)):
            self.samples[i] = self.convert_sample_labels(self.samples[i])

    def convert_sample_labels(self, sample):
        label = sample["label"]
        line_labels = label.split("\n")
        if "remove_linebreaks" in self.params["config"]["constraints"]:
            full_label = label.replace("\n", " ").replace("  ", " ")
            word_labels = full_label.split(" ")
        else:
            full_label = label
            word_labels = label.replace("\n", " ").replace("  ", " ").split(" ")

        sample["label"] = full_label
        sample["token_label"] = LM_str_to_ind(self.charset, full_label)
        if "add_eot" in self.params["config"]["constraints"]:
            sample["token_label"].append(self.tokens["end"])
        sample["label_len"] = len(sample["token_label"])
        if "add_sot" in self.params["config"]["constraints"]:
            sample["token_label"].insert(0, self.tokens["start"])

        sample["line_label"] = line_labels
        sample["token_line_label"] = [LM_str_to_ind(self.charset, l) for l in line_labels]
        sample["line_label_len"] = [len(l) for l in line_labels]
        sample["nb_lines"] = len(line_labels)

        sample["word_label"] = word_labels
        sample["token_word_label"] = [LM_str_to_ind(self.charset, l) for l in word_labels]
        sample["word_label_len"] = [len(l) for l in word_labels]
        sample["nb_words"] = len(word_labels)
        if "fdan_encoding" in self.params["config"]["constraints"]:
            sample["fdan_encoding"] = self.split_tokens_two_step(sample["token_label"])
        return sample

    def split_tokens_two_step(self, token_label):
        first_pass_tokens = list()
        lines_tokens = list()
        line_tokens = list()
        for i in range(len(token_label)):
            if token_label[i] == self.tokens["pad"]:
                continue
            if token_label[i] >= len(self.char_only_set) or (i > 0 and (token_label[i-1] >= len(self.char_only_set) or token_label[i-1] == self.charset.index("\n"))):
                first_pass_tokens.append(token_label[i])
                if len(line_tokens) > 0:
                    line_tokens.append(self.tokens["end"])
                    lines_tokens.append(line_tokens)
                    line_tokens = list()
            elif token_label[i] == self.charset.index("\n"):
                line_tokens.append(self.tokens["end"])
                lines_tokens.append(line_tokens)
                line_tokens = list()
                continue
            if token_label[i] < len(self.char_only_set):
                line_tokens.append(token_label[i])
        if len(line_tokens) > 0:
            lines_tokens.append(line_tokens)
        mix_passes = lines_tokens.copy()
        mix_passes.insert(0, first_pass_tokens)
        line_indices = np.concatenate([(np.ones(len(mix_passes[i])) * i) for i in range(len(mix_passes))], axis=0)
        index_in_lines = np.concatenate([np.arange(0, len(mix_passes[i])) for i in range(len(mix_passes))], axis=0)
        mix_passes = first_pass_tokens.copy()
        for line in lines_tokens:
            mix_passes.extend(line)
        return {
            "first_pass": first_pass_tokens,
            "second_pass": lines_tokens,
            "mix_pass": mix_passes,
            "line_indices": line_indices,
            "index_in_lines": index_in_lines,
        }

    def generate_synthetic_data(self, sample):
        sample = copy.deepcopy(sample)
        config = self.params["config"]["synthetic_data"]

        if ("wait_for_max_lines" not in config or not config["wait_for_max_lines"]) or self.get_syn_max_lines() == config["max_nb_lines"]:
            if not (config["init_proba"] == config["end_proba"] == 1):
                nb_samples = self.training_info["step"] * self.params["batch_size"]
                if "start_scheduler_at_max_line" in config and config["start_scheduler_at_max_line"]:
                    max_step = config["num_steps_proba"]
                    current_step = max(0, min(nb_samples-config["curr_step"]*(config["max_nb_lines"]-config["min_nb_lines"]), max_step))
                    proba = config["init_proba"] if self.get_syn_max_lines() < config["max_nb_lines"] else \
                        config["proba_scheduler_function"](config["init_proba"], config["end_proba"], current_step, max_step)
                else:
                    proba = config["proba_scheduler_function"](config["init_proba"], config["end_proba"],
                                                           min(nb_samples, config["num_steps_proba"]),
                                                           config["num_steps_proba"])
                if rand() > proba:
                    return sample

        if "typed" in config["mode"]:
            bg_color, txt_color = generate_colors_from_config(config["config"])
            config["txt_color"] = txt_color
            config["bg_color"] = bg_color

        dataset_name = None
        if "rimes" in sample["name"].lower():
            dataset_name = "RIMES"
        elif "read" in sample["name"].lower():
            dataset_name = "READ_2016"
        elif "maurdor" in sample["name"].lower():
            dataset_name = "MAURDOR"
        if dataset_name.lower() in ["rimes", "read_2016", "maurdor"] and "dataset_level" in config and "page" in config["dataset_level"]:
            return self.generate_synthetic_page_sample(dataset_name)

        if self.params["config"]["synthetic_data"]["mode"] == "line_hw_to_printed":
            if "random_text" in self.params["config"]["synthetic_data"] and self.params["config"]["synthetic_data"]["random_text"]:
                label = "".join([self.charset[randint(0, len(self.charset))] for _ in range(randint(1, self.get_syn_max_chars()+1))])
            else:
                label = sample["label"]
            if "curriculum" in config and config["curriculum"]:
                label = label[:self.get_syn_max_chars()]
                sample["label"] = label
                sample["unchanged_label"] = label
                sample = self.convert_sample_labels(sample)
            label = sample["label"][:100]
            sample["label"] = label
            sample["unchanged_label"] = label
            sample = self.convert_sample_labels(sample)
            sample["img"] = self.generate_typed_text_line_image(label)
            return sample

    def get_syn_max_lines(self):
        config = self.params["config"]["synthetic_data"]
        if config["curriculum"]:
            nb_samples = self.training_info["step"]*self.params["batch_size"]
            max_nb_lines = min(config["max_nb_lines"], (nb_samples-config["curr_start"]) // config["curr_step"]+1)
            return max(config["min_nb_lines"], max_nb_lines)
        return config["max_nb_lines"]

    def get_syn_max_chars(self):
        config = self.params["config"]["synthetic_data"]
        if config["curriculum"]:
            nb_samples = self.training_info["step"]*self.params["batch_size"]
            max_nb_chars = min(config["max_nb_chars"], (nb_samples-config["curr_start"]) // config["curr_step"]+1)
            return max(config["min_nb_chars"], max_nb_chars)
        return config["max_nb_chars"]

    def generate_synthetic_page_sample(self, dataset_name):
        samples = [a for a in self.samples if dataset_name in a["name"]]
        config = self.params["config"]["synthetic_data"]
        max_nb_lines_per_page = self.get_syn_max_lines()
        crop = config["crop_curriculum"] and max_nb_lines_per_page < config["max_nb_lines"]
        sample = {
            "name": "synthetic_{}_{}".format(dataset_name, self.synthetic_id),
            "path": None
        }
        self.synthetic_id += 1
        nb_pages = 2 if "double" in config["dataset_level"] else 1
        background_sample = copy.deepcopy(samples[randint(0, len(samples))])
        pages = list()
        backgrounds = list()

        if config["page_syn_mode"] == "typed":
            h, w, c = background_sample["img"].shape
            page_width = w // 2 if nb_pages == 2 else w
            for i in range(nb_pages):
                nb_lines_per_page = randint(config["min_nb_lines"], max_nb_lines_per_page+1)
                background = np.ones((h, page_width, c), dtype=background_sample["img"].dtype) * 255
                if i == 0 and nb_pages == 2:
                    background[:, -2:, :] = 0
                backgrounds.append(background)
                if dataset_name == "READ_2016":
                    side = background_sample["pages_label"][i]["side"]
                    coords = {
                        "left": int(0.15 * page_width) if side == "left" else int(0.05 * page_width),
                        "right": int(0.95 * page_width) if side == "left" else int(0.85 * page_width),
                        "top": int(0.05 * h),
                        "bottom": int(0.85 * h),
                    }
                    pages.append(self.generate_synthetic_read2016_page(background, coords, side=side, crop=crop,
                                                                   nb_lines=nb_lines_per_page))
                elif dataset_name == "RIMES":
                    pages.append(self.generate_synthetic_rimes_page(background, nb_lines=nb_lines_per_page, crop=crop))
                elif "MAURDOR" in dataset_name:
                    pages.append(self.generate_synthetic_maurdor_page(background, nb_lines=nb_lines_per_page, crop=crop))
                else:
                    raise NotImplementedError
        else:
            if dataset_name == "READ_2016":
                left_page_width = background_sample["pages_label"][0]["page_width"] if nb_pages == 2 else background_sample["img"].shape[1]
                for i in range(nb_pages):
                    nb_lines_per_page = randint(config["min_nb_lines"], max_nb_lines_per_page+1)
                    background = background_sample["img"][:, :left_page_width, ...] if i == 0 else background_sample["img"][:, left_page_width:, ...]
                    side = background_sample["pages_label"][i]["side"]
                    coords = background_sample["pages_label"][i]
                    if i == 1:
                        coords["left"] = coords["left"] - left_page_width
                        coords["right"] = coords["right"] - left_page_width
                    top, bottom, left, right = coords["top"], coords["bottom"], coords["left"], coords["right"]
                    padding_value = np.median(background[bottom:bottom+10, left:right, ...], axis=[0, 1]).astype(background.dtype)
                    top = min(top, 75)
                    bottom = max(bottom, 1650)
                    if side == "left":
                        left = min(left, 200)
                        right = max(right, background.shape[1] - 25)
                    else:
                        left = min(left, 25)
                        right = max(right, background.shape[1] - 200)
                    coords["left"] = left
                    coords["bottom"] = bottom
                    coords["top"] = top
                    coords["right"] = right
                    background[top:bottom, left:right, ...] = padding_value
                    backgrounds.append(background)
                    pages.append(self.generate_synthetic_read2016_page(background, coords, side=side, crop=crop,
                                                                    nb_lines=nb_lines_per_page))
            else:
                raise NotImplementedError
        if nb_pages == 1:
            sample["img"] = pages[0][0]
            sample["label_raw"] = pages[0][1]["raw"]
            sample["label_begin"] = pages[0][1]["begin"]
            sample["label_sem"] = pages[0][1]["sem"]
            sample["label"] = pages[0][1]
            sample["nb_cols"] = pages[0][2]
        else:
            if pages[0][0].shape[0] != pages[1][0].shape[0]:
                max_height = max(pages[0][0].shape[0], pages[1][0].shape[0])
                backgrounds[0] = backgrounds[0][:max_height]
                backgrounds[0][:pages[0][0].shape[0]] = pages[0][0]
                backgrounds[1] = backgrounds[1][:max_height]
                backgrounds[1][:pages[1][0].shape[0]] = pages[1][0]
                pages[0][0] = backgrounds[0]
                pages[1][0] = backgrounds[1]
            sample["label_raw"] = pages[0][1]["raw"] + "\n" + pages[1][1]["raw"]
            sample["label_begin"] = pages[0][1]["begin"] + pages[1][1]["begin"]
            sample["label_sem"] = pages[0][1]["sem"] + pages[1][1]["sem"]
            sample["img"] = np.concatenate([pages[0][0], pages[1][0]], axis=1)
            sample["nb_cols"] = pages[0][2] + pages[1][2]
        sample["label"] = sample["label_raw"]
        if any([b.islower() and b in self.charset for b in self.sem_tokens]):
            sample["label"] = sample["label_begin"]
        if any([b.isupper() and b in self.charset for b in self.sem_tokens]):
            sample["label"] = sample["label_sem"]
        sample["unchanged_label"] = sample["label"]
        sample = self.convert_sample_labels(sample)
        return sample

    def generate_synthetic_maurdor_page(self, background, nb_lines, crop=False):
        img_height, img_width = background.shape[:2]
        page_label = list()
        min_top_crop = img_height
        max_bottom_crop = 0
        min_line_height = 30
        orientation = "landscape" if img_height < img_width else "portrait"
        if orientation == "portrait":
            max_len_line = 100
        else:
            max_len_line = 150
        last_pg = {
            "top": 0,
            "bottom": 0,
            "right": 0,
            "height": 0,
        }
        while nb_lines > 0:
            if img_width - last_pg["right"] > 0.25*img_width:
                block_mode = np.random.choice(["bottom", "right_side"], 1, p=[0.4, 0.6])[0]
            else:
                block_mode = "bottom"
            if block_mode == "bottom":
                min_top = last_pg["bottom"]
                min_left = 0
                max_left = int(0.6*img_width)
            else:
                min_top = min(img_height, last_pg["top"] + randint(20, 30))
                min_left = int(last_pg["right"] + 0.02*img_width)
                max_left = int(0.8*img_width)

            num_line_block = randint(1, nb_lines+1)
            num_line_block = min(num_line_block, (img_height-min_top)//min_line_height)
            if num_line_block == 0:
                break
            left = randint(min_left, max_left+1)
            max_len_line_block = max(1, int((img_width-left)/img_width * max_len_line))
            line_labels = self.get_lines_maurdor(num_line_block, max_len_line=max_len_line_block)

            max_height = img_height - min_top
            nb_lines -= num_line_block
            max_width = img_width-left
            max_top = max(min(img_height, img_height-min_line_height*num_line_block), min_top)
            if max_top < 0:
                continue

            top = randint(min_top, max_top+1)

            pg_img = self.generate_typed_text_paragraph_image(line_labels, padding_value=255, max_pad_left_ratio=0.1, same_font_size=True)
            pg_img = resize_max(pg_img, max_height=int(img_height-top), max_width=int(max_width))
            h, w = pg_img.shape[:2]
            bottom = top + h
            right = left + w
            last_pg = {
                "top": top,
                "bottom": bottom,
                "right": right,
                "height": h,
            }
            background[top:bottom, left:right, ...] = pg_img
            page_label.extend(line_labels)
            min_top_crop = min(min_top_crop, top)
            max_bottom_crop = max(max_bottom_crop, bottom)

        if crop and min_top_crop < max_bottom_crop:
            background = background[min_top_crop:max_bottom_crop]

        page_label = "\n".join(page_label)
        page_labels = {
            "sem": page_label,
            "raw": page_label,
            "begin": page_label
        }
        return [background, page_labels, 1]

    def get_lines_maurdor(self, num_lines,  min_len_line=1, max_len_line=100):
        line_labels = list()
        while len(line_labels) != num_lines:
            line_label = ""
            while len(line_label) < min_len_line or len(line_label) > max_len_line:
                line_label = []
                while np.sum([len(l) for l in line_label]) < min_len_line:
                    sample = self.samples[randint(0, len(self))]
                    line_label.append(sample["line_label"][randint(0, len(sample["line_label"]))])
                line_label = " ".join(line_label)
                if len(line_label) > max_len_line:
                    line_label = " ".join(line_label[:max_len_line].split(" ")[:-1])
            line_labels.append(line_label)
        return line_labels

    def generate_synthetic_rimes_page(self, background, nb_lines=20, crop=False):
        max_nb_lines = self.get_syn_max_lines()
        def larger_lines(label):
            lines = label.split("\n")
            new_lines = list()
            while len(lines) > 0:
                if len(lines) == 1:
                    new_lines.append(lines[0])
                    del lines[0]
                elif len(lines[0]) + len(lines[1]) < max_len:
                    new_lines.append("{} {}".format(lines[0], lines[1]))
                    del lines[1]
                    del lines[0]
                else:
                    new_lines.append(lines[0])
                    del lines[0]
            return "\n".join(new_lines)
        config = self.params["config"]["synthetic_data"]
        max_len = config["max_char_per_line"]
        matching_tokens_str = RIMES_MATCHING_TOKENS_STR
        h, w, c = background.shape
        num_lines = list()
        for s in [a for a in self.samples if "rimes" in a["name"].lower()]:
            l = sum([len(p["label"].split("\n")) for p in s["paragraphs_label"]])
            num_lines.append(l)
        stats = self.stat_sem_rimes()
        ordered_modes = ['Corps de texte', 'PS/PJ', 'Ouverture', 'Date, Lieu', 'Coordonnées Expéditeur', 'Coordonnées Destinataire', ]
        object_ref = ['Objet', 'Reference']
        random.shuffle(object_ref)
        ordered_modes = ordered_modes[:3] + object_ref + ordered_modes[3:]
        kept_modes = list()
        for mode in ordered_modes:
            if mode in stats and rand_uniform(0, 1) < stats[mode]:
                kept_modes.append(mode)

        paragraphs = dict()
        for mode in kept_modes:
            paragraphs[mode] = self.get_paragraph_rimes(mode=mode, mix="mix_paragraphs" in config and config["mix_paragraphs"])
            if mode == "Corps de texte" and rand_uniform(0, 1) < 0.2:
                nb_lines = min(nb_lines+10, max_nb_lines) if max_nb_lines < 30 else nb_lines+10
                concat_line = randint(0, 2) == 0
                if concat_line:
                    paragraphs[mode]["label"] = larger_lines(paragraphs[mode]["label"])
                while (len(paragraphs[mode]["label"].split("\n")) <= 30):
                    body2 = self.get_paragraph_rimes(mode=mode, mix="mix_paragraphs" in config and config["mix_paragraphs"])
                    paragraphs[mode]["label"] += "\n" + larger_lines(body2["label"]) if concat_line else body2["label"]
                    paragraphs[mode]["label"] = "\n".join(paragraphs[mode]["label"].split("\n")[:40])
        if rand_uniform(0, 1) < 0.1 and "Corps de texte" in paragraphs:
            paragraphs["Corps de texte"]["label"] = paragraphs["Corps de texte"]["label"].upper().replace("È", "E").replace("Ë", "E").replace("Û", "U").replace("Ù", "U").replace("Î", "I").replace("Ï", "I").replace("Â", "A").replace("Œ", "OE")
        if rand_uniform(0, 1) < 0.1 and "Corps de texte" in paragraphs:
            labels = paragraphs["Corps de texte"]["label"].split("\n")
            duplicated_label = labels[randint(0, len(labels))]
            labels.insert(randint(0, len(labels)), duplicated_label)
            paragraphs["Corps de texte"]["label"] = "\n".join(labels)
        if rand_uniform(0, 1) < 0.1 and "Corps de texte" in paragraphs:
            paragraphs["Corps de texte"]["label"] = larger_lines(paragraphs["Corps de texte"]["label"])
        for mode in paragraphs.keys():
            line_labels = paragraphs[mode]["label"].split("\n")
            if len(line_labels) == 0:
                print("ERROR")
            paragraphs[mode]["lines"] = list()
            for line_label in line_labels:
                if len(line_label) > 100:
                    for chunk in [line_label[i:i + max_len] for i in range(0, len(line_label), max_len)]:
                        paragraphs[mode]["lines"].append(chunk)
                else:
                    paragraphs[mode]["lines"].append(line_label)
        page_labels = {
            "raw": "",
            "begin": "",
            "sem": ""
        }
        top_limit = 0
        bottom_limit = h
        max_bottom_crop = 0
        min_top_crop = h
        has_opening = has_object = has_reference = False
        top_opening = top_object = top_reference = 0
        right_opening = right_object = right_reference = 0
        has_reference = False
        date_on_top = False
        date_alone = False
        for mode in kept_modes:
            pg = paragraphs[mode]
            if len(pg["lines"]) > nb_lines:
                pg["lines"] = pg["lines"][:nb_lines]
            nb_lines -= len(pg["lines"])
            pg_image = self.generate_typed_text_paragraph_image(pg["lines"], padding_value=255, max_pad_left_ratio=1, same_font_size=True)
            if rand_uniform(0, 1) < 0.1:
                pg_image = apply_transform(pg_image, Tightening(color=255, remove_proba=0.75))
            if rand_uniform(0, 1) < 0.1:
                pg_image = apply_transform(pg_image, RandomRotation(degrees=10, expand=True, fill=255))
            pg["added"] = True
            if mode == 'Corps de texte':
                pg_image = resize_max(pg_image, max_height=int(0.5*h), max_width=w)
                img_h, img_w = pg_image.shape[:2]
                min_top = int(0.4*h)
                max_top = int(0.9*h - img_h)
                top = randint(min_top, max_top + 1)
                left = randint(0, int(w - img_w) + 1)
                bottom_body = top + img_h
                top_body = top
                bottom_limit = min(top, bottom_limit)
            elif mode == "PS/PJ":
                pg_image = resize_max(pg_image, max_height=int(0.03*h), max_width=int(0.9*w))
                img_h, img_w = pg_image.shape[:2]
                min_top = bottom_body
                max_top = int(min(h - img_h, bottom_body + 0.15*h))
                top = randint(min_top, max_top + 1)
                left = randint(0, int(w - img_w) + 1)
                bottom_limit = min(top, bottom_limit)
            elif mode == "Ouverture":
                pg_image = resize_max(pg_image, max_height=int(0.03 * h), max_width=int(0.9 * w))
                img_h, img_w = pg_image.shape[:2]
                min_top = int(top_body - 0.05 * h)
                max_top = top_body - img_h
                top = randint(min_top, max_top + 1)
                left = randint(0, min(int(0.15*w), int(w - img_w)) + 1)
                has_opening = True
                top_opening = top
                right_opening = left + img_w
                bottom_limit = min(top, bottom_limit)
            elif mode == "Objet":
                pg_image = resize_max(pg_image, max_height=int(0.03 * h), max_width=int(0.9 * w))
                img_h, img_w = pg_image.shape[:2]
                max_top = top_reference - img_h if has_reference else top_opening - img_h if has_opening else top_body - img_h
                min_top = int(max_top - 0.05 * h)
                top = randint(min_top, max_top + 1)
                left = randint(0, min(int(0.15*w), int(w - img_w)) + 1)
                has_object = True
                top_object = top
                right_object = left + img_w
                bottom_limit = min(top, bottom_limit)
            elif mode == "Reference":
                pg_image = resize_max(pg_image, max_height=int(0.03 * h), max_width=int(0.9 * w))
                img_h, img_w = pg_image.shape[:2]
                max_top = top_object - img_h if has_object else top_opening - img_h if has_opening else top_body - img_h
                min_top = int(max_top - 0.05 * h)
                top = randint(min_top, max_top + 1)
                left = randint(0, min(int(0.15*w), int(w - img_w)) + 1)
                has_reference = True
                top_reference = top
                right_reference = left + img_w
                bottom_limit = min(top, bottom_limit)
            elif mode == 'Date, Lieu':
                pg_image = resize_max(pg_image, max_height=int(0.03 * h), max_width=int(0.45 * w))
                img_h, img_w = pg_image.shape[:2]
                if h - max_bottom_crop - 10 > img_h and randint(0, 10) == 0:
                    top = randint(max_bottom_crop, h)
                    left = randint(0, w-img_w)
                else:
                    min_top = top_body - img_h
                    max_top = top_body - img_h
                    min_left = 0
                    # Check if there is anough place to put the date at the right side of opening, reference or object
                    if object_ref == ['Objet', 'Reference']:
                        have = [has_opening, has_object, has_reference]
                        rights = [right_opening, right_object, right_reference]
                        tops = [top_opening, top_object, top_reference]
                    else:
                        have = [has_opening, has_reference, has_object]
                        rights = [right_opening, right_reference, right_object]
                        tops = [top_opening, top_reference, top_object]
                    for right_r, top_r, has_r in zip(rights, tops, have):
                        if has_r:
                            if right_r + img_w >= 0.95*w:
                                max_top = min(top_r - img_h, max_top)
                                min_left = 0
                            else:
                                min_left = max(min_left, right_r+0.05*w)
                                min_top = top_r - img_h if min_top == top_body - img_h else min_top
                    if min_left != 0 and randint(0, 5) == 0:
                        min_left = 0
                        for right_r, top_r, has_r in zip(rights, tops, have):
                            if has_r:
                                max_top = min(max_top, top_r-img_h)

                    max_left = max(min_left, w - img_w)

                    # No placement found at right-side of opening, reference or object
                    if min_left == 0:
                        # place on the top
                        if randint(0, 2) == 0:
                            min_top = 0
                            max_top = int(min(0.05*h, max_top))
                            date_on_top = True
                        # place just before object/reference/opening
                        else:
                            min_top = int(max(0, max_top - 0.05*h))
                            date_alone = True
                            max_left = min(max_left, int(0.1*w))

                    min_top = min(min_top, max_top)
                    top = randint(min_top, max_top + 1)
                    left = randint(int(min_left), max_left + 1)
                    if date_on_top:
                        top_limit = max(top_limit, top + img_h)
                    else:
                        bottom_limit = min(top, bottom_limit)
                    date_right = left + img_w
                    date_bottom = top + img_h
            elif mode == "Coordonnées Expéditeur":
                max_height = min(0.25*h, bottom_limit-top_limit)
                if max_height <= 0:
                    pg["added"] = False
                    print("ko", bottom_limit, top_limit)
                    break
                pg_image = resize_max(pg_image, max_height=int(max_height), max_width=int(0.45 * w))
                img_h, img_w = pg_image.shape[:2]
                top = randint(top_limit, bottom_limit-img_h+1)
                left = randint(0, int(0.5*w-img_w)+1)
            elif mode == "Coordonnées Destinataire":
                if h - max_bottom_crop - 10 > 0.2*h and randint(0, 10) == 0:
                    pg_image = resize_max(pg_image, max_height=int(0.2*h), max_width=int(0.45 * w))
                    img_h, img_w = pg_image.shape[:2]
                    top = randint(max_bottom_crop, h)
                    left = randint(0, w-img_w)
                else:
                    max_height = min(0.25*h, bottom_limit-top_limit)
                    if max_height <= 0:
                        pg["added"] = False
                        print("ko", bottom_limit, top_limit)
                        break
                    pg_image = resize_max(pg_image, max_height=int(max_height), max_width=int(0.45 * w))
                    img_h, img_w = pg_image.shape[:2]
                    if date_alone and w - date_right - img_w > 11:
                        top = randint(0, date_bottom-img_h+1)
                        left = randint(max(int(0.5*w), date_right+10), w-img_w)
                    else:
                        top = randint(top_limit, bottom_limit-img_h+1)
                        left = randint(int(0.5*w), int(w - img_w)+1)

            bottom = top+img_h
            right = left+img_w
            min_top_crop = min(top, min_top_crop)
            max_bottom_crop = max(bottom, max_bottom_crop)
            try:
                background[top:bottom, left:right, ...] = pg_image
            except:
                pg["added"] = False
                nb_lines = 0
            pg["coords"] = {
                "top": top,
                "bottom": bottom,
                "right": right,
                "left": left
            }

            if nb_lines <= 0:
                break
        if "rimes_sem_order" in config and config["rimes_sem_order"]:
            sorted_pg = order_text_regions_rimes_sem(paragraphs.values())
        else:
            sorted_pg = order_text_regions_rimes(paragraphs.values())
        for pg in sorted_pg:
            if "added" in pg.keys() and pg["added"]:
                pg_label = "\n".join(pg["lines"])
                mode = pg["type"]
                begin_token = matching_tokens_str[mode]
                end_token = begin_token.upper()
                page_labels["raw"] += pg_label
                page_labels["begin"] += begin_token + pg_label
                page_labels["sem"] += begin_token + pg_label + end_token
        if crop:
            if min_top_crop > max_bottom_crop:
                print("KO - min > MAX")
            elif min_top_crop > h:
                print("KO - min > h")
            else:
                background = background[min_top_crop:max_bottom_crop]
        return [background, page_labels, 1]

    def stat_sem_rimes(self):
        try:
            return self.rimes_sem_stats
        except:
            samples = [a for a in self.samples if "rimes" in a["name"].lower()]
            stats = dict()
            for sample in samples:
                for pg in sample["paragraphs_label"]:
                    mode = pg["type"]
                    if mode == 'Coordonnées Expéditeur':
                        if len(pg["label"]) < 50 and "\n" not in pg["label"]:
                            mode = "Reference"
                    if mode not in stats.keys():
                        stats[mode] = 1
                    else:
                        stats[mode] += 1
            for key in stats:
                stats[key] = max(0.10, stats[key]/len(samples))
            self.rimes_sem_stats = stats
            return stats

    def get_paragraph_rimes(self, mode="Corps de texte", mix=False):
        samples = [a for a in self.samples if "rimes" in a["name"].lower()]
        while True:
            sample = samples[randint(0, len(samples))]
            random.shuffle(sample["paragraphs_label"])
            for pg in sample["paragraphs_label"]:
                pg_mode = pg["type"]
                if pg_mode == 'Coordonnées Expéditeur':
                    if len(pg["label"]) < 50 and "\n" not in pg["label"]:
                        pg_mode = "Reference"
                if mode == pg_mode:
                    if mode == "Corps de texte" and mix:
                        return self.get_mix_paragraph_rimes(mode, min(5, len(pg["label"].split("\n"))))
                    else:
                        return pg

    def get_mix_paragraph_rimes(self, mode="Corps de texte", num_lines=10):
        samples = [a for a in self.samples if "rimes" in a["name"].lower()]
        res = list()
        while len(res) != num_lines:
            sample = samples[randint(0, len(samples))]
            random.shuffle(sample["paragraphs_label"])
            for pg in sample["paragraphs_label"]:
                pg_mode = pg["type"]
                if pg_mode == 'Coordonnées Expéditeur':
                    if len(pg["label"]) < 50 and "\n" not in pg["label"]:
                        pg_mode = "Reference"
                if mode == pg_mode:
                    lines = pg["label"].split("\n")
                    res.append(lines[randint(0, len(lines))])
                    break
        return {
            "label": "\n".join(res),
            "type": mode,
        }

    def generate_synthetic_read2016_page(self, background, coords, side="left", nb_lines=20, crop=False):
        config = self.params["config"]["synthetic_data"]
        two_column = False
        sem_token = READ_SEM_TOKENS
        page_labels = {
            "raw": "",
            "begin": sem_token["page"],
            "sem": sem_token["page"],
        }
        area_top = 0 if crop else coords["top"]
        area_left = coords["left"]
        area_right = coords["right"]
        area_bottom = coords["bottom"]
        min_lines_per_body = 1
        max_lines_per_ann = 6
        if config["page_syn_mode"] == "typed":
            num_page_text_label = str(randint(0, 1000))
            num_page_img = self.generate_typed_text_line_image(num_page_text_label)
        else:
            num_page_sample = self.samples[randint(0, len(self))]
            num_page_label = num_page_sample["pages_label"][randint(0, len(num_page_sample["pages_label"]))]["paragraphs"][0]
            top, bottom, left, right = num_page_label["top"], num_page_label["bottom"], num_page_label["left"], num_page_label["right"]
            num_page_img = num_page_sample["img"][top:bottom, left:right]
            num_page_text_label = num_page_label["lines"][0]["text"]
        if side == "left":
            background[area_top:area_top+num_page_img.shape[0], area_left:area_left+num_page_img.shape[1]] = num_page_img
        else:
            background[area_top:area_top + num_page_img.shape[0], area_right-num_page_img.shape[1]:area_right] = num_page_img
        for key in ["sem", "begin"]:
            page_labels[key] += sem_token["number"]
        for key in page_labels.keys():
            page_labels[key] += num_page_text_label
        page_labels["sem"] += sem_token["number"].upper()
        area_top = area_top + num_page_img.shape[0]
        nb_lines -= 1
        if nb_lines > 0:
            area_top += randint(1, 25)
        ratio_ann = rand_uniform(0.6, 0.7)
        full_page = rand() > 0.85
        while nb_lines > 0:
            nb_body_lines = randint(min_lines_per_body, max(min_lines_per_body, nb_lines)+1)
            if full_page:
                nb_lines = nb_body_lines
            max_ann_lines = max(0, min(nb_body_lines, nb_lines-nb_body_lines))
            body_labels = list()
            body_imgs = list()
            while nb_body_lines > 0:
                if config["page_syn_mode"] == "typed":
                    current_nb_lines = 1
                    label, img = self.get_printed_line_read_2016("body")
                else:
                    current_nb_lines = randint(1, min(config["max_successive_lines"]+1, nb_body_lines+1))
                    label, img = self.get_n_consecutive_lines_read_2016(current_nb_lines, "body")

                nb_body_lines -= current_nb_lines
                body_labels.append(label)
                body_imgs.append(img)
            nb_ann_lines = randint(0, min(max_lines_per_ann, max_ann_lines+1))
            ann_labels = list()
            ann_imgs = list()
            while nb_ann_lines > 0:
                if config["page_syn_mode"] == "typed":
                    current_nb_lines = 1
                    label, img = self.get_printed_line_read_2016("annotation")
                else:
                    current_nb_lines = randint(1, min(3, nb_ann_lines+1))
                    label, img = self.get_n_consecutive_lines_read_2016(current_nb_lines, "ann")

                nb_ann_lines -= current_nb_lines
                ann_labels.append(label)
                ann_imgs.append(img)
            max_width_body = int(np.floor(ratio_ann*(area_right-area_left)))
            max_width_ann = area_right-area_left-max_width_body
            for img_list, max_width in zip([body_imgs, ann_imgs], [max_width_body, max_width_ann]):
                for i in range(len(img_list)):
                    if img_list[i].shape[1] > max_width:
                        ratio = max_width/img_list[i].shape[1]
                        new_h = int(np.floor(ratio*img_list[i].shape[0]))
                        new_w = int(np.floor(ratio*img_list[i].shape[1]))
                        img_list[i] = cv2.resize(img_list[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            body_top = area_top
            body_height = 0
            i_body = 0
            for i, (label, img) in enumerate(zip(body_labels, body_imgs)):
                remaining_height = area_bottom - body_top
                if img.shape[0] > remaining_height:
                    nb_lines = 0
                    break
                background[body_top:body_top+img.shape[0], area_left+max_width_ann:area_left+max_width_ann+img.shape[1]] = img
                body_height += img.shape[0]
                body_top += img.shape[0]
                if full_page and i < len(body_labels)-1:
                    interline_height = randint(int(0.9*img.shape[0]), int(2*img.shape[0]))
                    body_top += interline_height
                    body_height += interline_height
                nb_lines -= 1
                i_body += 1

            ann_height = int(np.sum([img.shape[0] for img in ann_imgs]))
            ann_top = area_top + randint(0, body_height-ann_height+1) if ann_height < body_height else area_top
            largest_ann = max([a.shape[1] for a in ann_imgs]) if len(ann_imgs) > 0 else max_width_ann
            pad_ann = randint(0, max_width_ann-largest_ann+1) if max_width_ann > largest_ann else 0

            ann_label_blocks = [list(), ]
            i_ann = 0
            ann_height = 0
            for (label, img) in zip(ann_labels, ann_imgs):
                remaining_height = body_top - ann_top
                if img.shape[0] > remaining_height:
                    break
                background[ann_top:ann_top+img.shape[0], area_left+pad_ann:area_left+pad_ann+img.shape[1]] = img
                ann_height += img.shape[0]
                ann_top += img.shape[0]
                nb_lines -= 1
                two_column = True
                ann_label_blocks[-1].append(ann_labels[i_ann])
                i_ann += 1
                if randint(0, 10) == 0:
                    ann_label_blocks.append(list())
                    ann_top += randint(0, max(15, body_top-ann_top-20))


            area_top = area_top + max(ann_height, body_height)
            if nb_lines > 0:
                area_top += randint(25, 100)

            ann_full_labels = {
                "raw": "",
                "begin": "",
                "sem": "",
            }
            for ann_label_block in ann_label_blocks:
                if len(ann_label_block) > 0:
                    for key in ["sem", "begin"]:
                        ann_full_labels[key] += sem_token["annotation"]
                    ann_full_labels["raw"] += "\n"
                    for key in ann_full_labels.keys():
                        ann_full_labels[key] += "\n".join(ann_label_block)
                    ann_full_labels["sem"] += sem_token["annotation"].upper()

            body_full_labels = {
                "raw": "",
                "begin": "",
                "sem": "",
            }
            if i_body > 0:
                for key in ["sem", "begin"]:
                    body_full_labels[key] += sem_token["body"]
                body_full_labels["raw"] += "\n"
                for key in body_full_labels.keys():
                    body_full_labels[key] += "\n".join(body_labels[:i_body])
                body_full_labels["sem"] += sem_token["body"].upper()

            section_labels = dict()
            for key in ann_full_labels.keys():
                section_labels[key] = ann_full_labels[key] + body_full_labels[key]
            for key in section_labels.keys():
                if section_labels[key] != "":
                    if key in ["sem", "begin"]:
                        section_labels[key] = sem_token["section"] + section_labels[key]
                    if key == "sem":
                        section_labels[key] = section_labels[key] + sem_token["section"].upper()
            for key in page_labels.keys():
                page_labels[key] += section_labels[key]

        if crop:
            background = background[:area_top]

        page_labels["sem"] += sem_token["page"].upper()

        for key in page_labels.keys():
            page_labels[key] = page_labels[key].strip()

        return [background, page_labels, 2 if two_column else 1]

    def get_n_consecutive_lines_read_2016(self, n=1, mode="body"):
        samples = [a for a in self.samples if "read_2016" in a["name"].lower()]
        while True:
            sample = samples[randint(0, len(samples))]
            paragraphs = list()
            for page in sample["pages_label"]:
                paragraphs.extend(page["paragraphs"])
                random.shuffle(paragraphs)
                for pg in paragraphs:
                    if ((mode == "body" and pg["mode"] == "body") or
                        (mode == "ann" and pg["mode"] == "annotation")) and len(pg["lines"]) >= n:
                        line_idx = randint(0, len(pg["lines"])-n+1)
                        lines = pg["lines"][line_idx:line_idx+n]
                        label = "\n".join([l["text"] for l in lines])
                        top = min([l["top"] for l in lines])
                        bottom = max([l["bottom"] for l in lines])
                        left = min([l["left"] for l in lines])
                        right = max([l["right"] for l in lines])
                        img = sample["img"][top:bottom, left:right]
                        return label, img

    def get_printed_line_read_2016(self, mode="body"):
        samples = [a for a in self.samples if "read_2016" in a["name"].lower()]
        while True:
            sample = samples[randint(0, len(samples))]
            for page in sample["pages_label"]:
                paragraphs = list()
                paragraphs.extend(page["paragraphs"])
                random.shuffle(paragraphs)
                for pg in paragraphs:
                    random.shuffle(pg["lines"])
                    for line in pg["lines"]:
                        if (mode == "body" and len(line["text"]) > 5) or (mode == "annotation" and len(line["text"]) < 15 and not line["text"].isdigit()):
                            label = line["text"]
                            img = self.generate_typed_text_line_image(label)
                            return label, img

    def generate_typed_text_line_image(self, text):
        return generate_typed_text_line_image(text, self.params["config"]["synthetic_data"]["config"])

    def generate_typed_text_paragraph_image(self, texts, padding_value=255, max_pad_left_ratio=0.1, same_font_size=False):
        config = self.params["config"]["synthetic_data"]["config"]
        if same_font_size:
            images = list()
            txt_color = config["text_color_default"]
            bg_color = config["background_color_default"]
            font_size = randint(config["font_size_min"], config["font_size_max"] + 1)
            for text in texts:
                font_path = config["valid_fonts"][randint(0, len(config["valid_fonts"]))]
                fnt = ImageFont.truetype(font_path, font_size)
                text_width, text_height = fnt.getsize(text)
                padding_top = int(rand_uniform(config["padding_top_ratio_min"], config["padding_top_ratio_max"]) * text_height)
                padding_bottom = int(rand_uniform(config["padding_bottom_ratio_min"], config["padding_bottom_ratio_max"]) * text_height)
                padding_left = int(rand_uniform(config["padding_left_ratio_min"], config["padding_left_ratio_max"]) * text_width)
                padding_right = int(rand_uniform(config["padding_right_ratio_min"], config["padding_right_ratio_max"]) * text_width)
                padding = [padding_top, padding_bottom, padding_left, padding_right]
                images.append(generate_typed_text_line_image_from_params(text, fnt, bg_color, txt_color, config["color_mode"], padding))
        else:
            images = [self.generate_typed_text_line_image(t) for t in texts]

        max_width = max([img.shape[1] for img in images])

        padded_images = [pad_image_width_random(img, max_width, padding_value=padding_value, max_pad_left_ratio=max_pad_left_ratio) for img in images]
        return np.concatenate(padded_images, axis=0)


class OCRCollateFunction:
    """
    Merge samples data to mini-batch data for OCR task
    """

    def __init__(self, config):
        self.img_padding_value = float(config["padding_value"])
        self.label_padding_value = config["padding_token"]
        self.config = config

    def __call__(self, batch_data):
        names = [batch_data[i]["name"] for i in range(len(batch_data))]
        ids = [int(batch_data[i]["name"].split("/")[-1].split("_")[-1].split(".")[0]) for i in range(len(batch_data))]
        applied_da = [batch_data[i]["applied_da"] for i in range(len(batch_data))]

        labels = [batch_data[i]["token_label"] for i in range(len(batch_data))]
        labels = pad_sequences_1D(labels, padding_value=self.label_padding_value)
        labels = torch.tensor(labels).long()
        reverse_labels = [[batch_data[i]["token_label"][0], ] + batch_data[i]["token_label"][-2:0:-1] + [batch_data[i]["token_label"][-1], ] for i in range(len(batch_data))]
        reverse_labels = pad_sequences_1D(reverse_labels, padding_value=self.label_padding_value)
        reverse_labels = torch.tensor(reverse_labels).long()
        labels_len = [batch_data[i]["label_len"] for i in range(len(batch_data))]

        raw_labels = [batch_data[i]["label"] for i in range(len(batch_data))]
        unchanged_labels = [batch_data[i]["unchanged_label"] for i in range(len(batch_data))]

        nb_cols = [batch_data[i]["nb_cols"] for i in range(len(batch_data))]
        nb_lines = [batch_data[i]["nb_lines"] for i in range(len(batch_data))]
        line_raw = [batch_data[i]["line_label"] for i in range(len(batch_data))]
        line_token = [batch_data[i]["token_line_label"] for i in range(len(batch_data))]
        pad_line_token = list()
        line_len = [batch_data[i]["line_label_len"] for i in range(len(batch_data))]
        for i in range(max(nb_lines)):
            current_lines = [line_token[j][i] if i < nb_lines[j] else [self.label_padding_value] for j in range(len(batch_data))]
            pad_line_token.append(torch.tensor(pad_sequences_1D(current_lines, padding_value=self.label_padding_value)).long())
            for j in range(len(batch_data)):
                if i >= nb_lines[j]:
                    line_len[j].append(0)
        line_len = [i for i in zip(*line_len)]

        nb_words = [batch_data[i]["nb_words"] for i in range(len(batch_data))]
        word_raw = [batch_data[i]["word_label"] for i in range(len(batch_data))]
        word_token = [batch_data[i]["token_word_label"] for i in range(len(batch_data))]
        pad_word_token = list()
        word_len = [batch_data[i]["word_label_len"] for i in range(len(batch_data))]
        for i in range(max(nb_words)):
            current_words = [word_token[j][i] if i < nb_words[j] else [self.label_padding_value] for j in range(len(batch_data))]
            pad_word_token.append(torch.tensor(pad_sequences_1D(current_words, padding_value=self.label_padding_value)).long())
            for j in range(len(batch_data)):
                if i >= nb_words[j]:
                    word_len[j].append(0)
        word_len = [i for i in zip(*word_len)]

        padding_mode = self.config["padding_mode"] if "padding_mode" in self.config else "br"
        imgs = [batch_data[i]["img"] for i in range(len(batch_data))]
        imgs_shape = [batch_data[i]["img_shape"] for i in range(len(batch_data))]
        imgs_reduced_shape = [batch_data[i]["img_reduced_shape"] for i in range(len(batch_data))]
        imgs_position = [batch_data[i]["img_position"] for i in range(len(batch_data))]
        imgs_reduced_position= [batch_data[i]["img_reduced_position"] for i in range(len(batch_data))]
        imgs = pad_images(imgs, padding_value=self.img_padding_value, padding_mode=padding_mode)
        imgs = torch.tensor(imgs).float().permute(0, 3, 1, 2)

        formatted_batch_data = {
            "names": names,
            "ids": ids,
            "nb_lines": nb_lines,
            "nb_cols": nb_cols,
            "labels": labels,
            "reverse_labels": reverse_labels,
            "raw_labels": raw_labels,
            "unchanged_labels": unchanged_labels,
            "labels_len": labels_len,
            "imgs": imgs,
            "imgs_shape": imgs_shape,
            "imgs_reduced_shape": imgs_reduced_shape,
            "imgs_position": imgs_position,
            "imgs_reduced_position": imgs_reduced_position,
            "line_raw": line_raw,
            "line_labels": pad_line_token,
            "line_labels_len": line_len,
            "nb_words": nb_words,
            "word_raw": word_raw,
            "word_labels": pad_word_token,
            "word_labels_len": word_len,
            "applied_da": applied_da,
        }

        if "fdan_encoding" in batch_data[0]:
            fdan_first_pass = [batch_data[i]["fdan_encoding"]["first_pass"] for i in range(len(batch_data))]
            formatted_batch_data["fdan_first_pass"] = fdan_first_pass
            fdan_second_pass = [batch_data[i]["fdan_encoding"]["first_pass"] for i in range(len(batch_data))]
            formatted_batch_data["fdan_second_pass"] = fdan_second_pass

            fdan_labels = [batch_data[i]["fdan_encoding"]["mix_pass"] for i in range(len(batch_data))]
            fdan_labels = pad_sequences_1D(fdan_labels, padding_value=self.label_padding_value)
            fdan_labels = torch.tensor(fdan_labels).long()
            formatted_batch_data["fdan_labels"] = fdan_labels

            fdan_line_indices = [batch_data[i]["fdan_encoding"]["line_indices"] for i in range(len(batch_data))]
            fdan_line_indices = pad_sequences_1D(fdan_line_indices, padding_value=self.label_padding_value)
            fdan_line_indices = torch.tensor(fdan_line_indices).long()
            formatted_batch_data["fdan_line_indices"] = fdan_line_indices

            fdan_index_in_lines = [batch_data[i]["fdan_encoding"]["index_in_lines"] for i in range(len(batch_data))]
            fdan_index_in_lines = pad_sequences_1D(fdan_index_in_lines, padding_value=self.label_padding_value)
            fdan_index_in_lines = torch.tensor(fdan_index_in_lines).long()
            formatted_batch_data["fdan_index_in_lines"] = fdan_index_in_lines

        return formatted_batch_data

def generate_colors_from_config(config):
    background_color = config["background_color_default"]
    eps = randint(-config["background_color_eps"], config["background_color_eps"] + 1)
    background_color = tuple([min(255, max(0, c + eps)) for c in background_color])

    text_color = config["text_color_default"]
    eps = randint(-config["text_color_eps"], config["text_color_eps"] + 1)
    text_color = tuple([min(255, max(0, c + eps)) for c in text_color])
    return background_color, text_color


def generate_typed_text_line_image(text, config, bg_color=(255, 255, 255), txt_color=(0, 0, 0)):
    if text == "":
        text = " "
    if "text_color_default" in config:
        txt_color = config["text_color_default"]
    if "background_color_default" in config:
        bg_color = config["background_color_default"]

    max_width = config["max_width"] if "max_width" in config else None
    text_width = max_width
    while text_width is None or (max_width and text_width >= max_width) or text_width == 0 or text_height == 0:
        font_path = config["valid_fonts"][randint(0, len(config["valid_fonts"]))]
        font_size = randint(config["font_size_min"], config["font_size_max"]+1)
        fnt = ImageFont.truetype(font_path, font_size)
        text_width, text_height = fnt.getsize(text)

    padding_top = int(rand_uniform(config["padding_top_ratio_min"], config["padding_top_ratio_max"])*text_height)
    padding_bottom = int(rand_uniform(config["padding_bottom_ratio_min"], config["padding_bottom_ratio_max"])*text_height)
    padding_left = int(rand_uniform(config["padding_left_ratio_min"], config["padding_left_ratio_max"])*text_width)
    padding_right = int(rand_uniform(config["padding_right_ratio_min"], config["padding_right_ratio_max"])*text_width)
    padding = [padding_top, padding_bottom, padding_left, padding_right]
    return generate_typed_text_line_image_from_params(text, fnt, bg_color, txt_color, config["color_mode"], padding)


def generate_typed_text_line_image_from_params(text, font, bg_color, txt_color, color_mode, padding):
    padding_top, padding_bottom, padding_left, padding_right = padding
    text_width, text_height = font.getsize(text)
    img_height = padding_top + padding_bottom + text_height
    img_width = padding_left + padding_right + text_width
    img = Image.new(color_mode, (img_width, img_height), color=bg_color)
    d = ImageDraw.Draw(img)
    d.text((padding_left, padding_bottom), text, font=font, fill=txt_color, spacing=0)
    return np.array(img)


def get_valid_fonts(alphabet=None):
    valid_fonts = list()
    for fold_detail in os.walk("../../../Fonts"):
        if fold_detail[2]:
            for font_name in fold_detail[2]:
                font_path = os.path.join(fold_detail[0], font_name)
                to_add = True
                if alphabet is not None:
                    for char in alphabet:
                        if not char_in_font(char, font_path):
                            to_add = False
                            break
                    if to_add:
                        valid_fonts.append(font_path)
                else:
                    valid_fonts.append(font_path)
    return valid_fonts


def char_in_font(unicode_char, font_path):
    with TTFont(font_path) as font:
        for cmap in font['cmap'].tables:
            if cmap.isUnicode():
                if ord(unicode_char) in cmap.cmap:
                    return True
    return False