from faster_dan.OCR.ocr_manager import OCRManager
import torch
from faster_dan.basic.utils import randint
import numpy as np


class Manager(OCRManager):

    def __init__(self, params):
        super(Manager, self).__init__(params)

    def load_save_info(self, info_dict):
        if "curriculum_config" in info_dict.keys():
            if self.dataset.train_dataset is not None:
                self.dataset.train_dataset.curriculum_config = info_dict["curriculum_config"]

    def add_save_info(self, info_dict):
        info_dict["curriculum_config"] = self.dataset.train_dataset.curriculum_config
        return info_dict

    def get_init_hidden(self, batch_size):
        num_layers = 1
        hidden_size = self.params["model_params"]["enc_dim"]
        return torch.zeros(num_layers, batch_size, hidden_size), torch.zeros(num_layers, batch_size, hidden_size)

    def ctc_remove_successives_identical_ind(self, ind, confidence=None):
        res = list()
        conf = list()
        for i, e in enumerate(ind):
            if res and res[-1] == e:
                continue
            res.append(e)
            if confidence is not None:
                conf.append(confidence[i])
        if confidence is not None:
            return res, conf
        return res

    def ctc_remove_blank(self, pred, confidence=None):
        if confidence is not None:
            pred, conf = self.remove_ind_from_pred(pred, [self.dataset.tokens["blank"]], confidence)
            return pred, conf
        return self.remove_ind_from_pred(pred, [self.dataset.tokens["blank"]])

    def remove_ind_from_pred(self, pred, ind, confidence=None):
        res = list()
        conf = list()
        for i, e in enumerate(pred):
            if e in ind:
                continue
            res.append(e)
            if confidence is not None:
                conf.append(confidence[i])
        if confidence is not None:
            return res, conf
        return res

    def remove_ind_from_pred_list(self, preds, inds, confidences=None):
        res_pred = list()
        if confidences is None:
            confidences = [None for _ in range(len(preds))]
            res_conf = None
        else:
            res_conf = list()
        for p, c in zip(preds, confidences):
            if c is not None:
                pred, conf = self.remove_ind_from_pred(p, inds, c)
                res_pred.append(pred)
                res_conf.append(conf)
            else:
                res_pred.append(self.remove_ind_from_pred(p, inds))
        if res_conf is None:
            return res_pred
        return res_pred, res_conf

    def ctc_decoding_list_tensors(self, tensors, confidences=None):
        res_pred = list()
        if confidences is None:
            confidences = [None for _ in range(len(tensors))]
            res_conf = None
        else:
            res_conf = list()
        for pred_tensor, pred_conf in zip(tensors, confidences):
            if pred_conf is not None:
                pred, conf = self.ctc_remove_successives_identical_ind(pred_tensor, pred_conf)
                pred, conf = self.ctc_remove_blank(pred, conf)
                res_pred.append(pred)
                res_conf.append(conf)
            else:
                pred = self.ctc_remove_successives_identical_ind(pred_tensor)
                pred = self.ctc_remove_blank(pred)
                res_pred.append(pred)
        if res_conf is None:
            return res_pred
        return res_pred, res_conf

    def pack_list_tensors_1d(self, tensors, padding_value):
        max_len = max([t.size(0) for t in tensors])
        num = len(tensors)
        res = torch.ones((num, max_len), dtype=tensors[0].dtype, device=tensors[0].device) * padding_value
        for k in range(num):
            res[k, :tensors[k].size(0)] = tensors[k]
        return res

    def add_error_in_gt(self, y, y_len, error_rate, preserve_layout=False):
        y_error = y.clone()
        for b in range(len(y_len)):
            for i in range(1, y_len[b]):
                if np.random.rand() < error_rate:
                    if preserve_layout:
                        if y[b][i] != self.dataset.charset.index("\n") and y[b][i] < len(self.dataset.char_only_set):
                            new_token = self.dataset.charset.index("\n")
                            while new_token == self.dataset.charset.index("\n"):
                                new_token = np.random.randint(0, len(self.dataset.char_only_set))
                        else:
                            new_token = y_error[b][i]
                    else:
                        new_token = np.random.randint(0, len(self.dataset.charset))
                    y_error[b][i] = new_token
        return y_error

    def add_sub_rem_from_gt(self, y, y_len, padding_value, add_rate=0.05, rem_rate=0.05, sub_rate=0.1):
        new_y_tensor = list()
        id_rate = 1-add_rate-rem_rate-sub_rate
        for b in range(len(y_len)):
            new_y = list()
            for i in range(y_len[b]):
                op = np.random.choice(4, p=[id_rate, add_rate, rem_rate, sub_rate])
                # Identity = no change
                if op == 0:
                    new_y.append(y[b, i])
                # Addition
                elif op == 1:
                    new_y.append(randint(0, len(self.dataset.charset)))
                    new_y.append(y[b, i])
                # Remove
                elif op == 2:
                    pass
                # Substitute
                elif op == 3:
                    new_y.append(randint(0, len(self.dataset.charset)))
            new_y_tensor.append(torch.tensor(new_y, dtype=y.dtype, device=y.device))
        new_y_len = [len(t) for t in new_y_tensor]
        new_y_tensor = self.pack_list_tensors_1d(new_y_tensor, padding_value)
        return new_y_tensor, new_y_len

    def is_end_layout_token(self, token):
        end_layout_tokens = [self.dataset.charset.index(t) for t in self.metric_manager["train"].matching_tokens.values()]
        return token in [self.dataset.charset.index("\n"), self.dataset.tokens["end"]] or token in end_layout_tokens

    def evaluate_encoder(self, x, batch_data):
        b = x.size(0)
        pos = batch_data["imgs_position"]
        if b > 1:
            features_list = list()
            for i in range(b):
                features_list.append(self.models["encoder"](x[i:i + 1, :, pos[i][0][0]:pos[i][0][1], pos[i][1][0]:pos[i][1][1]]))
            max_height = max([f.size(2) for f in features_list])
            max_width = max([f.size(3) for f in features_list])
            features = torch.zeros((b, features_list[0].size(1), max_height, max_width), device=self.device, dtype=features_list[0].dtype)
            for i in range(b):
                features[i, :, :features_list[i].size(2), :features_list[i].size(3)] = features_list[i]
        else:
            features = self.models["encoder"](x)
        return features

    def train_encoder(self, x, batch_data):
        return self.models["encoder"](x)