from faster_dan.OCR.document_OCR.faster_dan.trainer_dan import Manager as DANManager
from torch.nn import CrossEntropyLoss
import torch
from faster_dan.OCR.ocr_utils import LM_ind_to_str
from torch.cuda.amp import autocast
import time


class Manager(DANManager):

    def __init__(self, params):
        super(DANManager, self).__init__(params)

    def get_line_indices_from_tokens(self, batch_tokens):
        batch_line_indices = list()
        batch_index_in_lines = list()
        for b in range(batch_tokens.size(0)):
            tokens = batch_tokens[b]
            new_line_token = self.dataset.charset.index("\n")
            line_indices = list()
            index_in_lines = list()
            last_token_char = True
            line_index = -1
            index_in_line = 0
            for t in tokens:
                if t == self.dataset.tokens["pad"]:
                    pass
                elif t != new_line_token and t <= len(self.dataset.char_only_set):
                    if not last_token_char:
                        line_index += 1
                        index_in_line = 0
                    else:
                        index_in_line += 1
                    last_token_char = True
                else:
                    last_token_char = False
                    index_in_line += 1

                index_in_lines.append(index_in_line)
                line_indices.append(line_index)
            batch_line_indices.append(torch.tensor(line_indices, dtype=batch_tokens.dtype, device=batch_tokens.device))
            batch_index_in_lines.append(torch.tensor(index_in_lines, dtype=batch_tokens.dtype, device=batch_tokens.device))
        return torch.stack(batch_line_indices, dim=0), torch.stack(batch_index_in_lines, dim=0)

    def train_batch(self, batch_data, metric_names):
        fn_loss_ce = CrossEntropyLoss(ignore_index=self.dataset.tokens["pad"], reduction="mean")
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]
        str_y = batch_data["raw_labels"]

        # add errors in teacher forcing
        if "teacher_forcing_error_rate" in self.params["training_params"] and self.params["training_params"]["teacher_forcing_error_rate"] is not None:
            error_rate = self.params["training_params"]["teacher_forcing_error_rate"]
            simulated_y_pred = self.add_error_in_gt(y, y_len, error_rate)
        elif "teacher_forcing_scheduler" in self.params["training_params"]:
            error_rate = self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"] + min(self.latest_step, self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]) * (self.params["training_params"]["teacher_forcing_scheduler"]["max_error_rate"]-self.params["training_params"]["teacher_forcing_scheduler"]["min_error_rate"]) / self.params["training_params"]["teacher_forcing_scheduler"]["total_num_steps"]
            simulated_y_pred = self.add_error_in_gt(y, y_len, error_rate)
        else:
            simulated_y_pred = y

        with autocast(enabled=self.params["training_params"]["use_amp"]):

            raw_features = self.train_encoder(x, batch_data)
            features_size = raw_features.size()
            b, c, h, w = features_size

            pos_features = self.models["decoder"].features_updater.get_pos_features(raw_features)
            pos_features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)

            simulated_y_pred = simulated_y_pred[:, :-1]
            y = y[:, 1:]
            line_indices = None
            index_in_lines = None
            if "use_line_indices" in self.params["model_params"] and self.params["model_params"]["use_line_indices"]:
                line_indices, index_in_lines = self.get_line_indices_from_tokens(simulated_y_pred)
            output, pred, cache, weights = self.models["decoder"](pos_features, pos_features,
                                                                           simulated_y_pred,
                                                                           reduced_size,
                                                                           y_len,
                                                                           features_size,
                                                                           start=0,
                                                                          line_indices=line_indices,
                                                                          index_in_lines=index_in_lines,
                                                                           keep_all_weights=True,
                                                                            padding_value=self.dataset.tokens["pad"])
            loss = fn_loss_ce(pred, y)
            predicted_tokens = torch.argmax(pred, dim=1).detach().cpu().numpy()
            predicted_tokens = [predicted_tokens[i, :y_len[i]] for i in range(b)]
            str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in predicted_tokens]

        self.backward_loss(loss)
        self.step_optimizers()
        self.zero_optimizers()

        values = {
            "nb_samples": b,
            "str_y": str_y,
            "str_x": str_x,
            "loss": loss.item(),
            "syn_max_lines": self.dataset.train_dataset.get_syn_max_lines() if self.params["dataset_params"]["config"]["synthetic_data"] else 0,
        }

        return values

    def evaluate_batch(self, batch_data, metric_names):
        x = batch_data["imgs"].to(self.device)
        reduced_size = [s[:2] for s in batch_data["imgs_reduced_shape"]]
        max_chars = self.params["training_params"]["max_char_prediction"]

        start_time = time.time()
        with autocast(enabled=self.params["training_params"]["use_amp"]):
            b = x.size(0)
            reached_end = torch.zeros((b,), dtype=torch.bool, device=self.device)
            predicted_tokens = torch.ones((b, 1), dtype=torch.long, device=self.device) * self.dataset.tokens["start"]
            prediction_len = torch.ones((b,), dtype=torch.int, device=self.device)

            whole_output = list()
            confidence_scores = list()
            cache = None

            features = self.evaluate_encoder(x, batch_data)
            features_size = features.size()
            pos_features = self.models["decoder"].features_updater.get_pos_features(features)
            pos_features = torch.flatten(pos_features, start_dim=2, end_dim=3).permute(2, 0, 1)

            for i in range(1, max_chars+1):
                line_indices = None
                index_in_lines = None
                if "use_line_indices" in self.params["model_params"] and self.params["model_params"]["use_line_indices"]:
                    line_indices, index_in_lines = self.get_line_indices_from_tokens(predicted_tokens)
                output, pred, cache, weights = self.models["decoder"](pos_features, pos_features, predicted_tokens,
                                                                      reduced_size,
                                                                      prediction_len, features_size, start=0,
                                                                      cache=cache, num_pred=1,
                                                                      line_indices=line_indices,
                                                                      index_in_lines=index_in_lines,
                                                                      padding_value=self.dataset.tokens["pad"])
                whole_output.append(output)
                confidence_scores.append(torch.max(torch.softmax(pred[:, :], dim=1), dim=1).values)
                predicted_tokens = torch.cat([predicted_tokens, torch.argmax(pred[:, :, -1], dim=1, keepdim=True)], dim=1)
                reached_end = torch.logical_or(reached_end, torch.eq(predicted_tokens[:, -1], self.dataset.tokens["end"]))

                prediction_len[torch.eq(reached_end, False)] = i + 1
                if torch.all(reached_end):
                    break

            prediction_len[torch.eq(reached_end, False)] = max_chars
            predicted_tokens = predicted_tokens[:, 1:]
            confidence_scores = torch.cat(confidence_scores, dim=1).cpu().detach().numpy()
            pred_tokens = [predicted_tokens[i, :prediction_len[i]] for i in range(b)]
            confidence_scores = [confidence_scores[i, :prediction_len[i]].tolist() for i in range(b)]

            ind_to_remove = [self.dataset.tokens["end"]]
            if "blank" in self.dataset.tokens:
                ind_to_remove.append(self.dataset.tokens["blank"])
            pred_tokens, confidence_scores = self.remove_ind_from_pred_list(pred_tokens, ind_to_remove, confidence_scores)
            str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in pred_tokens]

        process_time = time.time() - start_time

        values = {
            "nb_samples": b,
            "str_y": batch_data["raw_labels"],
            "str_x": str_x,
            "confidence_score": confidence_scores,
            "time": process_time,
            "names": batch_data["names"]
        }

        return values