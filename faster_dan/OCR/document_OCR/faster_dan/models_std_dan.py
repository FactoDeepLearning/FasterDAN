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

import torch
from torch import relu
from torch.nn import Conv1d, Dropout
from torch.nn import Embedding
from torch.nn import Module
from faster_dan.OCR.document_OCR.faster_dan.models_dan import FeaturesUpdater, GlobalAttDecoder, PositionalEncoding1D, PositionalEncoding1DOnTheFly


class GlobalHTADecoder(Module):
    """
    DAN decoder module
    """
    def __init__(self, params):
        super(GlobalHTADecoder, self).__init__()
        self.params = params
        self.enc_dim = params["enc_dim"]
        self.dec_l_max = params["l_max"]

        self.dropout = Dropout(params["dec_pred_dropout"])
        self.dec_att_win = params["attention_win"] if params["attention_win"] is not None else 1

        self.features_updater = FeaturesUpdater(params)
        self.att_decoder = GlobalAttDecoder(params)

        self.emb = Embedding(num_embeddings=params["vocab_size"]+3, embedding_dim=self.enc_dim)

        self.use_line_indices = params["use_line_indices"] if "use_line_indices" in params else False
        if self.use_line_indices:
            if params["two_step_pos_enc_mode"] == "add":
                self.pe_1d = PositionalEncoding1DOnTheFly(self.enc_dim, params["device"])
            else:
                self.pe_1d = PositionalEncoding1DOnTheFly(self.enc_dim // 2, params["device"])
        else:
            self.pe_1d = PositionalEncoding1D(self.enc_dim, self.dec_l_max, params["device"])

        vocab_size = params["vocab_size"] + 1
        self.end_conv = Conv1d(self.enc_dim, vocab_size, kernel_size=1)

    def forward(self, raw_features_1d, enhanced_features_1d, tokens, reduced_size, token_len, features_size, start=0, padding_value=None,
                cache=None, num_pred=None, keep_all_weights=False, line_indices=None, index_in_lines=None):
        device = raw_features_1d.device
        # Token to Embedding
        emb_tokens = self.emb(tokens).permute(0, 2, 1)

        if self.use_line_indices:
            if self.params["two_step_pos_enc_mode"] == "cat":
                pos_tokens = emb_tokens + torch.cat([self.pe_1d(line_indices), self.pe_1d(index_in_lines)], dim=1)
            else:
                pos_tokens = emb_tokens + self.pe_1d(line_indices) + self.pe_1d(index_in_lines)
        else:
            # Add 1D Positional Encoding
            pos_tokens = self.pe_1d(emb_tokens, start=start)

        pos_tokens = pos_tokens.permute(2, 0, 1)

        if num_pred is None:
            num_pred = tokens.size(1)

        # Use cache values to avoid useless computation at eval time
        if self.dec_att_win > 1 and cache is not None:
            cache = cache[:, -self.dec_att_win + 1:]
        else:
            cache = None
        num_tokens_to_keep = num_pred if self.dec_att_win is None else min([num_pred + self.dec_att_win - 1, pos_tokens.size(0), max(token_len)])

        pos_tokens = pos_tokens[-num_tokens_to_keep:]

        # Generate dynamic masks
        target_mask = self.generate_target_mask(tokens.size(1), device)  # Use only already predicted tokens (causal)
        memory_mask = None  # Use all feature position

        # Generate static masks
        key_target_mask = self.generate_token_mask(tokens, token_len, device, padding_value)  # Use all token except padding
        key_memory_mask = self.generate_enc_mask(reduced_size, features_size, device)  # Use all feature position except padding

        target_mask = target_mask[..., -num_pred:, -num_tokens_to_keep:]
        key_target_mask = key_target_mask[:, -num_tokens_to_keep:]

        output, weights, cache = self.att_decoder(pos_tokens, memory_key=enhanced_features_1d,
                                        memory_value=raw_features_1d,
                                        tgt_mask=target_mask,
                                        memory_mask=memory_mask,
                                        tgt_key_padding_mask=key_target_mask,
                                        memory_key_padding_mask=key_memory_mask,
                                        use_cache=True,
                                        cache=cache,
                                        predict_last_n_only=num_pred,
                                        keep_all_weights=keep_all_weights)

        dp_output = self.dropout(relu(output))
        preds = self.end_conv(dp_output.permute(1, 2, 0))

        if not keep_all_weights:
            weights = torch.sum(weights, dim=1, keepdim=True).reshape(-1, 1, features_size[2], features_size[3])
        return output, preds, cache, weights

    def generate_enc_mask(self, batch_reduced_size, total_size, device):
        """
        Generate mask for encoded features
        """
        batch_size, _, h_max, w_max = total_size
        mask = torch.ones((batch_size, h_max, w_max), dtype=torch.bool, device=device)
        for i, (h, w) in enumerate(batch_reduced_size):
            mask[i, :h, :w] = False
        return torch.flatten(mask, start_dim=1, end_dim=2)

    def generate_token_mask(self, tokens, token_len, device, padding_value):
        """
        Generate mask for tokens per sample
        """
        batch_size, len_max = tokens.size()
        mask = torch.zeros((batch_size, len_max), dtype=torch.bool, device=device)
        for i, len_ in enumerate(token_len):
            mask[i, :len_] = False
        mask[tokens == padding_value] = True
        return mask

    def generate_target_mask(self, target_len, device, correction_pass=False):
        """
        Generate mask for tokens per time step (teacher forcing)
        """
        if correction_pass:
            return torch.zeros((target_len, target_len), dtype=torch.bool, device=device)
        else:
            return torch.logical_not(
                torch.logical_and(torch.tril(torch.ones((target_len, target_len), dtype=torch.bool, device=device), diagonal=0),
                                  torch.triu(torch.ones((target_len, target_len), dtype=torch.bool, device=device), diagonal=-self.dec_att_win+1)))

