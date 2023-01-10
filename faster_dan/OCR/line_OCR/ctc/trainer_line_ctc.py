from faster_dan.OCR.ocr_manager import OCRManager
from faster_dan.OCR.ocr_utils import LM_ind_to_str
import torch
from torch.cuda.amp import autocast
from torch.nn import CTCLoss
import re
import time


class TrainerLineCTC(OCRManager):

    def __init__(self, params):
        super(TrainerLineCTC, self).__init__(params)
        super(TrainerLineCTC, self).__init__(params)

    def ctc_remove_successives_identical_ind(self, ind):
        res = []
        for i in ind:
            if res and res[-1] == i:
                continue
            res.append(i)
        return res

    def train_batch(self, batch_data, metric_names):
        """
        Forward and backward pass for training
        """
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"])
        self.zero_optimizers()

        with autocast(enabled=self.params["training_params"]["use_amp"]):
            x = self.models["encoder"](x)
            global_pred = self.models["decoder"](x)

        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)

        self.backward_loss(loss)
        self.step_optimizers()
        pred = torch.argmax(global_pred, dim=1).cpu().numpy()

        values = {
            "nb_samples": len(batch_data["raw_labels"]),
            "loss": loss.item(),
            "str_x": self.pred_to_str(pred, x_reduced_len),
            "str_y": batch_data["raw_labels"]
        }

        return values

    def evaluate_batch(self, batch_data, metric_names):
        """
        Forward pass only for validation and test
        """
        x = batch_data["imgs"].to(self.device)
        y = batch_data["labels"].to(self.device)
        x_reduced_len = [s[1] for s in batch_data["imgs_reduced_shape"]]
        y_len = batch_data["labels_len"]

        loss_ctc = CTCLoss(blank=self.dataset.tokens["blank"])

        start_time = time.time()
        with autocast(enabled=self.params["training_params"]["use_amp"]):
            x = self.models["encoder"](x)
            global_pred = self.models["decoder"](x)

        loss = loss_ctc(global_pred.permute(2, 0, 1), y, x_reduced_len, y_len)
        pred = torch.argmax(global_pred, dim=1).cpu().numpy()
        str_x = self.pred_to_str(pred, x_reduced_len)

        process_time = time.time() - start_time

        values = {
            "nb_samples": len(batch_data["raw_labels"]),
            "loss_ctc": loss.item(),
            "str_x": str_x,
            "str_y": batch_data["raw_labels"],
            "time": process_time
        }
        return values

    def pred_to_str(self, pred, pred_len):
        """
        convert prediction tokens to string
        """
        ind_x = [pred[i][:pred_len[i]] for i in range(pred.shape[0])]
        ind_x = [self.ctc_remove_successives_identical_ind(t) for t in ind_x]
        str_x = [LM_ind_to_str(self.dataset.charset, t, oov_symbol="") for t in ind_x]
        str_x = [re.sub("( )+", ' ', t).strip(" ") for t in str_x]
        return str_x