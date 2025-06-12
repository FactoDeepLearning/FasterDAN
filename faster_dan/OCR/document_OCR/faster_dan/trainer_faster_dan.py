#  Copyright Université de Rouen Normandie (1), INSA Rouen (2),
#  tutelles du laboratoire LITIS (1 et 2)
#  contributors :
#  - Denis Coquenet
#
#  This software is a computer program written in Python whose purpose is 
#  to recognize text and layout from full-page images with end-to-end deep neural networks.
#
#  This software is governed by the CeCILL-C license under French law and
#  abiding by the rules of distribution of free software.  You can  use,
#  modify and/ or redistribute the software under the terms of the CeCILL-C
#  license as circulated by CEA, CNRS and INRIA at the following URL
#  "http://www.cecill.info".
#
#  As a counterpart to the access to the source code and  rights to copy,
#  modify and redistribute granted by the license, users are provided only
#  with a limited warranty  and the software's author,  the holder of the
#  economic rights,  and the successive licensors  have only  limited
#  liability.
#
#  In this respect, the user's attention is drawn to the risks associated
#  with loading,  using,  modifying and/or developing or reproducing the
#  software by the user in light of its specific status of free software,
#  that may mean  that it is complicated to manipulate,  and  that  also
#  therefore means  that it is reserved for developers  and  experienced
#  professionals having in-depth computer knowledge. Users are therefore
#  encouraged to load and test the software's suitability as regards their
#  requirements in conditions enabling the security of their systems and/or
#  data to be ensured and,  more generally, to use and operate it in the
#  same conditions as regards security.
#
#  The fact that you are presently reading this means that you have had
#  knowledge of the CeCILL-C license and that you accept its terms.

from faster_dan.OCR.document_OCR.faster_dan.trainer_dan import Manager as DANManager
from torch.nn import CrossEntropyLoss
import torch
from faster_dan.OCR.ocr_utils import LM_ind_to_str
import numpy as np
from torch.cuda.amp import autocast
import time


class Manager(DANManager):

    def __init__(self, params):
        super(DANManager, self).__init__(params)

    def split_tokens_two_step(self, tokens):
        end_line_token = torch.tensor(self.dataset.tokens["end"], device=tokens.device, dtype=tokens.dtype)
        batch_token_list = list()
        for k in range(tokens.size(0)):
            first_pass_tokens = list()
            lines_tokens = list()
            line_tokens = list()
            for i in range(tokens.size(1)):
                if tokens[k, i] == self.dataset.tokens["pad"]:
                    continue
                if tokens[k, i] >= len(self.dataset.char_only_set) or (i > 0 and (tokens[k, i-1] >= len(self.dataset.char_only_set) or tokens[k, i-1] == self.dataset.charset.index("\n"))):
                    first_pass_tokens.append(tokens[k, i])
                    if len(line_tokens) > 0:
                        line_tokens.append(end_line_token)
                        lines_tokens.append(line_tokens)
                        line_tokens = list()
                elif tokens[k, i] == self.dataset.charset.index("\n"):
                    line_tokens.append(end_line_token)
                    lines_tokens.append(line_tokens)
                    line_tokens = list()
                    continue
                if tokens[k, i] < len(self.dataset.char_only_set):
                    line_tokens.append(tokens[k, i])
            if len(line_tokens) > 0:
                lines_tokens.append(line_tokens)
            first_pass = torch.stack(first_pass_tokens)
            lines_tokens = [torch.stack(l) for l in lines_tokens]
            lines_tokens.insert(0, first_pass)

            tf_tokens = torch.cat([l[:-1] for l in lines_tokens])
            gt_tokens = torch.cat([l[1:] for l in lines_tokens])
            line_indices = torch.cat([(torch.ones(lines_tokens[i].size(), device=tokens.device, dtype=tokens.dtype)*i)[:-1] for i in range(len(lines_tokens))])
            index_in_lines = torch.cat([torch.arange(0, lines_tokens[i].size(0), device=tokens.device, dtype=tokens.dtype)[:-1] for i in range(len(lines_tokens))])
            batch_token_list.append((tf_tokens, gt_tokens, line_indices, index_in_lines))
        max_len = max([b[0].size(0) for b in batch_token_list])
        padded_batch_tokens = list()
        for i in range(len(batch_token_list[0])):
            padded_tokens = torch.ones((tokens.size(0), max_len), device=tokens.device, dtype=tokens.dtype) * self.dataset.tokens["pad"]
            for k in range(tokens.size(0)):
                padded_tokens[k, :batch_token_list[k][i].size(0)] = batch_token_list[k][i]
            padded_batch_tokens.append(padded_tokens)
        return padded_batch_tokens

    def merge_tokens_two_step(self, tokens, line_indices, y):
        new_tokens = list()
        for b in range(len(tokens)):
            lines_tokens = list()
            num_first_pass = int(np.sum(line_indices[b] == 0))
            indices = np.argwhere(y[b, :num_first_pass] < len(self.dataset.char_only_set))[:, 0]
            for i in range(tokens[b].shape[0]):
                line_index = int(line_indices[b, i])
                if line_index == 0:
                    lines_tokens.append([tokens[b][i]])
                else:
                    lines_tokens[indices[line_index-1]].append(tokens[b][i])

            for i in range(len(lines_tokens)-1):
                if len(lines_tokens[i]) > 1 and len(lines_tokens[i+1]) > 1:
                    lines_tokens[i].append(self.dataset.charset.index("\n"))
            new_tokens.append(np.concatenate(lines_tokens, axis=0))
        return new_tokens

    def is_end_layout_token(self, token):
        end_layout_tokens = [self.dataset.charset.index(t) for t in self.metric_manager["train"].matching_tokens.values()]
        return token in [self.dataset.charset.index("\n"), self.dataset.tokens["end"]] or token in end_layout_tokens

    def train_batch(self, batch_data, metric_names):
        loss_ce = CrossEntropyLoss(ignore_index=self.dataset.tokens["pad"], reduction="sum")
        sum_loss = 0
        x = batch_data["imgs"].to(self.device)
        y = batch_data["fdan_labels"].to(self.device)
        line_indices = batch_data["fdan_line_indices"][:, :-1].to(self.device)
        index_in_lines = batch_data["fdan_index_in_lines"][:, :-1].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]
        str_y = batch_data["raw_labels"]

        # add errors in teacher forcing
        if "teacher_forcing_error_rate" in self.params["training_params"] and self.params["training_params"]["teacher_forcing_error_rate"] is not None:
            error_rate = self.params["training_params"]["teacher_forcing_error_rate"]
            simulated_y_pred = self.add_error_in_gt(y, y_len, error_rate, preserve_layout=True)
        elif "teacher_forcing_scheduler" in self.params["training_params"]:
            error_rate = self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"] + min(self.latest_step, self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]) * (self.params["training_params"]["teacher_forcing_scheduler"]["max_error_rate"]-self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"]) / self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]
            simulated_y_pred = self.add_error_in_gt(y, y_len, error_rate, preserve_layout=True)
        else:
            simulated_y_pred = y

        with autocast(enabled=self.params["training_params"]["use_amp"]):
            raw_features = self.train_encoder(x, batch_data)
            features_size = raw_features.size()
            b, c, h, w = features_size

            pos_features = self.models["decoder"].features_updater.get_pos_features(raw_features)
            pos_features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)
            y = y[:, 1:]
            simulated_y_pred = simulated_y_pred[:, :-1]
            y[simulated_y_pred == self.dataset.tokens["end"]] = self.dataset.tokens["pad"]
            token_len = [int(torch.where(y_i==self.dataset.tokens["pad"])[0][0]) if torch.where(y_i==self.dataset.tokens["pad"])[0].size(0) > 0 else y_i.size(0) for y_i in simulated_y_pred]
            output, pred, cache, weights = self.models["decoder"](pos_features, pos_features,
                                                                  simulated_y_pred,
                                                                  reduced_size,
                                                                  token_len,
                                                                  features_size,
                                                                  start=0,
                                                                  keep_all_weights=True,
                                                                  line_indices=line_indices,
                                                                  index_in_lines=index_in_lines,
                                                                  padding_value=self.dataset.tokens["pad"])
            first_pass_mask = torch.logical_and(line_indices == 0, y != self.dataset.tokens["pad"])
            second_pass_mask = torch.logical_and(line_indices != 0, y != self.dataset.tokens["pad"])
            sum_loss += (loss_ce(pred.permute(0, 2, 1)[first_pass_mask], y[first_pass_mask]) + loss_ce(pred.permute(0, 2, 1)[second_pass_mask], y[second_pass_mask])) / (torch.sum(second_pass_mask)+torch.sum(first_pass_mask))

            predicted_tokens = torch.argmax(pred, dim=1).detach().cpu().numpy()
            predicted_tokens = [predicted_tokens[i, :token_len[i]] for i in range(b)]
            predicted_tokens = self.merge_tokens_two_step(predicted_tokens, line_indices.detach().cpu().numpy(), y.detach().cpu().numpy())

            with autocast(enabled=False):
                self.backward_loss(sum_loss)
                self.step_optimizers()
                self.zero_optimizers()

            str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in predicted_tokens]

        values = {
            "nb_samples": b,
            "str_y": str_y,
            "str_x": str_x,
            "loss": sum_loss.item(),
            "syn_max_lines": self.dataset.train_dataset.get_syn_max_lines() if self.params["dataset_params"]["config"]["synthetic_data"] else 0,
        }

        return values

    def evaluate_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]

        start_time = time.time()
        with autocast(enabled=self.params["training_params"]["use_amp"]):
            b = x.size(0)
            features = self.evaluate_encoder(x, batch_data)
            features_size = features.size()
            pos_features = self.models["decoder"].features_updater.get_pos_features(features)
            pos_features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)

            predicted_tokens, prediction_len, confidence_scores, cache = self.eval_first_pass(pos_features, reduced_size, features_size)

            _, y, y_line_ind, _ = self.split_tokens_two_step(batch_data["labels"])

            str_y_first_pass = [y[i][torch.where(y_line_ind[i] == 0)] for i in range(y.size(0))]
            str_y_first_pass = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in str_y_first_pass]
            str_x_first_pass = [p[p != self.dataset.tokens["pad"]] for p in predicted_tokens]
            str_x_first_pass = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in str_x_first_pass]

            scores, predicted_tokens = self.eval_second_pass(pos_features, reduced_size, features_size,
                                                             predicted_tokens, prediction_len,
                                                             confidence_scores, cache)

            str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in predicted_tokens]
            str_y = batch_data["raw_labels"]

        process_time = time.time() - start_time
        values = {
            "nb_samples": b,
            "str_y": str_y,
            "str_x": str_x,
            "str_y_first_pass": str_y_first_pass,
            "str_x_first_pass": str_x_first_pass,
            "confidence_score": scores,
            "time": process_time,
            "names": batch_data["names"]
        }

        return values

    def eval_first_pass(self, pos_features, reduced_size, features_size, return_weights=False):
        max_line_pred = self.params["training_params"]["max_line_pred"]
        b = pos_features.size(1)
        cache = None
        whole_output = list()
        global_weights = list()
        confidence_scores = torch.zeros((b, 1), dtype=torch.float, device=self.device)
        line_indices = torch.zeros((b, 1), dtype=torch.long, device=self.device)
        index_in_lines = torch.zeros((b, 1), dtype=torch.long, device=self.device)
        predicted_tokens = torch.ones((b, 1), dtype=torch.long, device=self.device) * self.dataset.tokens["start"]
        prediction_len = torch.ones((b,), dtype=torch.int, device=self.device)
        reached_end = torch.zeros((b,), dtype=torch.bool, device=self.device)
        for i in range(1, max_line_pred+1):
            output, pred, cache, weights = self.models["decoder"](pos_features,
                                                                  pos_features,
                                                                  predicted_tokens,
                                                                  reduced_size,
                                                                  prediction_len,
                                                                  features_size,
                                                                  cache=cache,
                                                                  num_pred=1,
                                                                  line_indices=line_indices,
                                                                  index_in_lines=index_in_lines,
                                                                  padding_value=self.dataset.tokens["pad"])
            whole_output.append(output)
            if return_weights:
                global_weights.append(weights)
            confidence_scores = torch.cat([confidence_scores, torch.max(torch.softmax(pred[:, :], dim=1), dim=1).values], dim=1)
            new_predicted_tokens = torch.argmax(pred[:, :, -1], dim=1, keepdim=True)
            new_predicted_tokens[reached_end] = self.dataset.tokens["pad"]
            predicted_tokens = torch.cat([predicted_tokens, new_predicted_tokens], dim=1)

            line_indices = torch.zeros(predicted_tokens.size(), device=line_indices.device, dtype=line_indices.dtype)
            index_in_lines = torch.stack([torch.arange(0, predicted_tokens.size(1), device=index_in_lines.device, dtype=index_in_lines.dtype) for _ in range(predicted_tokens.size(0))])

            prediction_len[reached_end == False] = i + 1
            reached_end = torch.logical_or(reached_end, torch.eq(predicted_tokens[:, -1], self.dataset.tokens["end"]))
            if torch.all(reached_end):
                break
        prediction_len[torch.eq(reached_end, False)] = max_line_pred
        predicted_tokens = predicted_tokens[:, 1:-1]
        confidence_scores = confidence_scores[:, 1:-1]
        cache = cache[:, :-1]
        prediction_len -= 2
        if return_weights:
            return predicted_tokens, prediction_len, confidence_scores, cache, torch.cat(global_weights[:-1], dim=1)
        return predicted_tokens, prediction_len, confidence_scores, cache

    def eval_second_pass(self, pos_features, reduced_size, features_size, predicted_tokens, prediction_len, confidence_scores, cache, first_pass_weights=None):
        max_pred_per_line = self.params["training_params"]["max_pred_per_line"]
        b = pos_features.size(1)
        global_weights = torch.cat([first_pass_weights, first_pass_weights], dim=1) if first_pass_weights is not None else None
        line_indices = torch.zeros(predicted_tokens.size(), device=predicted_tokens.device, dtype=predicted_tokens.dtype)
        index_in_lines = torch.stack([torch.arange(0, predicted_tokens.size(1), device=predicted_tokens.device, dtype=predicted_tokens.dtype) for _ in range(b)], dim=0)
        line_indices2 = torch.where(predicted_tokens[:, :] < len(self.dataset.char_only_set), 1, 0)
        line_indices2 = torch.cumsum(line_indices2, dim=1)
        line_indices = torch.cat([line_indices, line_indices2], dim=1)
        index_in_lines = torch.cat([index_in_lines, torch.zeros(index_in_lines.size(), device=index_in_lines.device,
                                                                dtype=index_in_lines.dtype)], dim=1)
        num_lines = predicted_tokens.size(1)
        if num_lines == 0:
            return np.empty((b, 0)), np.empty((b, 0))
        predicted_tokens = torch.cat([predicted_tokens, predicted_tokens], dim=1)
        index_to_pad = torch.logical_and(predicted_tokens >= len(self.dataset.char_only_set), line_indices!=0)
        predicted_tokens[index_to_pad] = self.dataset.tokens["pad"]
        num_lines_per_sample = prediction_len
        prediction_len = prediction_len * 2
        confidence_scores = torch.cat([confidence_scores, confidence_scores], dim=1)
        prediction_len_line = torch.zeros((b, num_lines), dtype=torch.int, device=self.device)
        reached_end_line = torch.zeros((b, num_lines), dtype=torch.bool, device=self.device)
        for bi in range(b):
            for k in range(num_lines):
                if predicted_tokens[bi, k] >= len(self.dataset.char_only_set):
                    reached_end_line[bi, k] = True
                    prediction_len_line[bi, k] = 1

        for i in range(0, max_pred_per_line):
            num_pred = num_lines
            model = self.models["decoder"]
            output, pred, cache, weights = model(pos_features,
                                                 pos_features,
                                                 predicted_tokens,
                                                 reduced_size,
                                                 prediction_len,
                                                 features_size,
                                                 cache=cache,
                                                 num_pred=num_pred,
                                                 line_indices=line_indices,
                                                 index_in_lines=index_in_lines,
                                                 two_step_parallel=True,
                                                 padding_value=self.dataset.tokens["pad"]
                                                 )
            if first_pass_weights is not None:
                global_weights = torch.cat([global_weights, weights], dim=1)
            confidence_scores = torch.cat([confidence_scores, torch.max(torch.softmax(pred[:, :], dim=1), dim=1).values], dim=1)
            new_predicted_tokens = torch.argmax(pred[:, :, -num_lines:], dim=1, keepdim=False)
            new_predicted_tokens[reached_end_line] = self.dataset.tokens["pad"]
            predicted_tokens = torch.cat([predicted_tokens, new_predicted_tokens], dim=1)
            index_in_lines = torch.cat([index_in_lines, index_in_lines[:, -num_lines:] + 1], dim=1)
            line_indices = torch.cat([line_indices, line_indices[:, num_lines:2 * num_lines]], dim=1)
            prediction_len += num_lines
            prediction_len_line[reached_end_line == False] = i + 1
            reached_end_line = torch.logical_or(reached_end_line, torch.eq(predicted_tokens[:, -num_lines:], self.dataset.tokens["end"]))
            if torch.all(reached_end_line):
                break

        line_break_token = torch.ones((1,), device=predicted_tokens.device, dtype=predicted_tokens.dtype) * self.dataset.charset.index("\n")
        line_break_confidence = torch.ones((1,), device=confidence_scores.device, dtype=confidence_scores.dtype)
        prediction_len_line[torch.eq(reached_end_line, False)] = max_pred_per_line + 1
        batch_pred = list()
        for k in range(b):
            list_line_tokens = list()
            list_line_scores = list()
            list_line_weights = list()
            for l in range(num_lines_per_sample[k]):
                line_tokens = torch.cat([predicted_tokens[k, l:l+1], predicted_tokens[k, 2*num_lines + l::num_lines]], dim=0)
                line_tokens = line_tokens[:prediction_len_line[k, l]]
                list_line_tokens.append(line_tokens)
                line_scores = torch.cat([confidence_scores[k, l:l+1], confidence_scores[k, 2*num_lines + l::num_lines]], dim=0)
                line_scores = line_scores[:prediction_len_line[k, l]]
                list_line_scores.append(line_scores)
                if first_pass_weights is not None:
                    line_weights = global_weights[k, num_lines + l::num_lines]
                    line_weights = line_weights[:prediction_len_line[k, l]]
                    list_line_weights.append(line_weights)
            batch_pred.append([list_line_tokens, list_line_scores, list_line_weights])
        if first_pass_weights is None:
            tokens = list()
            scores = list()
            for k in range(b):
                list_line_tokens, list_line_scores, list_line_weights = batch_pred[k]
                for l in range(len(list_line_tokens) - 1):
                    if list_line_tokens[l][0] < len(self.dataset.char_only_set) and list_line_tokens[l + 1][0] < len(self.dataset.char_only_set):
                        list_line_tokens[l] = torch.cat([list_line_tokens[l], line_break_token], dim=0)
                        list_line_scores[l] = torch.cat([list_line_scores[l], line_break_confidence], dim=0)

                list_line_tokens = torch.cat(list_line_tokens, dim=0) if len(list_line_tokens) > 0 else torch.tensor([], device=self.device)
                tokens.append(list_line_tokens)
                list_line_scores = torch.cat(list_line_scores, dim=0) if len(list_line_scores) > 0 else torch.tensor([], device=self.device)
                if "blank" in self.dataset.tokens:
                    list_line_scores = list_line_scores[list_line_tokens != self.dataset.tokens["blank"]]
                scores.append(list_line_scores.cpu().detach().numpy())
            return scores, tokens
        line_pred = [a[0] for a in batch_pred]
        line_weights = [a[2] for a in batch_pred]
        return line_pred, line_weights
