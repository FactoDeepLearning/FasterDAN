import torch
from torch import relu, softmax
from torch.nn import Dropout,  Linear, LayerNorm
from torch.nn import ModuleList, Module
from torch.nn.init import xavier_uniform_


class PositionalEncoding1D(Module):

    def __init__(self, dim, len_max, device):
        super(PositionalEncoding1D, self).__init__()
        self.len_max = len_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, len_max), device=device, requires_grad=False)

        div = torch.exp(-torch.arange(0., dim, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        l_pos = torch.arange(0., len_max)
        self.pe[:, ::2, :] = torch.sin(l_pos * div).unsqueeze(0)
        self.pe[:, 1::2, :] = torch.cos(l_pos * div).unsqueeze(0)

    def forward(self, x, start):
        """
        Add 1D positional encoding to x
        x: (B, C, L)
        start: index for x[:,:, 0]
        """
        if isinstance(start, int):
            return x + self.pe[:, :, start:start+x.size(2)].to(x.device)
        else:
            for i in range(x.size(0)):
                x[i] = x[i] + self.pe[0, :, start[i]:start[i]+x.size(2)]
            return x


class PositionalEncoding1DOnTheFly(Module):

    def __init__(self, dim, device):
        super(PositionalEncoding1DOnTheFly, self).__init__()
        self.dim = dim
        self.div = torch.exp(-torch.arange(0., dim, 2, device=device) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(0)

    def forward(self, indices):
        emb_indices = torch.zeros((indices.size(0), self.dim, indices.size(1)), device=indices.device)
        emb_indices[:, ::2, :] = torch.sin(indices.unsqueeze(2) * self.div).permute(0, 2, 1)
        emb_indices[:, 1::2, :] = torch.cos(indices.unsqueeze(2) * self.div).permute(0, 2, 1)
        return emb_indices


class PositionalEncoding2D(Module):

    def __init__(self, dim, h_max, w_max, device):
        super(PositionalEncoding2D, self).__init__()
        self.h_max = h_max
        self.max_w = w_max
        self.dim = dim
        self.pe = torch.zeros((1, dim, h_max, w_max), device=device, requires_grad=False)

        div = torch.exp(-torch.arange(0., dim // 2, 2) / dim * torch.log(torch.tensor(10000.0))).unsqueeze(1)
        w_pos = torch.arange(0., w_max)
        h_pos = torch.arange(0., h_max)
        self.pe[:, :dim // 2:2, :, :] = torch.sin(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        self.pe[:, 1:dim // 2:2, :, :] = torch.cos(h_pos * div).unsqueeze(0).unsqueeze(3).repeat(1, 1, 1, w_max)
        self.pe[:, dim // 2::2, :, :] = torch.sin(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)
        self.pe[:, dim // 2 + 1::2, :, :] = torch.cos(w_pos * div).unsqueeze(0).unsqueeze(2).repeat(1, 1, h_max, 1)

    def forward(self, x):
        """
        Add 2D positional encoding to x
        x: (B, C, H, W)
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

    def get_pe_by_size(self, h, w, device):
        return self.pe[:, :, :h, :w].to(device)


class CustomMultiHeadAttention(Module):
    """
    Re-implementation of Multi-head Attention
    """
    def __init__(self, embed_dim, num_heads, dropout=0, proj_value=True):
        super().__init__()

        self.proj_value = proj_value

        self.in_proj_q = Linear(embed_dim, embed_dim)
        self.in_proj_k = Linear(embed_dim, embed_dim)
        if self.proj_value:
            self.in_proj_v = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale_factor = float(self.head_dim) ** -0.5
        self.dropout = Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, output_weights=True):
        target_len, b, c = query.size()
        source_len = key.size(0)
        q = self.in_proj_q(query)
        k = self.in_proj_k(key)
        v = self.in_proj_v(value) if self.proj_value else value
        q = q * self.scale_factor

        q = torch.reshape(q, (target_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
        k = torch.reshape(k, (source_len, b*self.num_heads, self.head_dim)).transpose(0, 1)
        v = torch.reshape(v, (source_len, b*self.num_heads, self.head_dim)).transpose(0, 1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            if attn_mask.ndim == 2:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = torch.repeat_interleave(attn_mask.unsqueeze(1), self.num_heads, dim=1).reshape(b * self.num_heads, target_len, source_len)

            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(b, self.num_heads, target_len, source_len)

            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf"),
            )
            attn_output_weights = attn_output_weights.view(b * self.num_heads, target_len, source_len)

        attn_output_weights_raw = softmax(attn_output_weights, dim=-1)

        attn_output_weights = self.dropout(attn_output_weights_raw)

        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.transpose(0, 1).contiguous().view(target_len, b, c)
        attn_output = self.out_proj(attn_output)

        if output_weights:
            attn_output_weights_raw = attn_output_weights_raw.view(b, self.num_heads, target_len, source_len)
            return attn_output, attn_output_weights_raw.sum(dim=1) / self.num_heads
        return attn_output

    def init_weights(self):
        xavier_uniform_(self.in_proj_q.weight)
        xavier_uniform_(self.in_proj_k.weight)
        if self.proj_value:
            xavier_uniform_(self.in_proj_v.weight)


class GlobalDecoderLayer(Module):
    """
    Transformer Decoder Layer
    """

    def __init__(self, params):
        super(GlobalDecoderLayer, self).__init__()
        self.emb_dim = params["enc_dim"]
        self.dim_feedforward = params["dec_dim_feedforward"]

        self.self_att = CustomMultiHeadAttention(embed_dim=self.emb_dim,
                                                  num_heads=params["dec_num_heads"],
                                                  proj_value=True,
                                                  dropout=params["dec_att_dropout"])

        self.norm1 = LayerNorm(self.emb_dim)
        self.att = CustomMultiHeadAttention(embed_dim=self.emb_dim,
                                                  num_heads=params["dec_num_heads"],
                                                  proj_value=True,
                                                  dropout=params["dec_att_dropout"])

        self.linear1 = Linear(self.emb_dim, self.dim_feedforward)
        self.linear2 = Linear(self.dim_feedforward, self.emb_dim)

        self.dropout = Dropout(params["dec_res_dropout"])

        self.norm2 = LayerNorm(self.emb_dim)
        self.norm3 = LayerNorm(self.emb_dim)

    def forward(self, tgt, memory_key, memory_value=None, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, predict_last_n_only=None):

        if memory_value is None:
            memory_value = memory_key

        self_att_query = tgt[-predict_last_n_only:] if predict_last_n_only else tgt

        tgt2, weights_self = self.self_att(self_att_query, tgt, tgt, attn_mask=tgt_mask,
                                           key_padding_mask=tgt_key_padding_mask, output_weights=True)
        tgt = self_att_query + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        att_query = tgt

        tgt2, weights = self.att(att_query, memory_key, memory_value, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask, output_weights=True)

        tgt = att_query + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt, weights, weights_self


class GlobalAttDecoder(Module):
    """
    Stack of transformer decoder layers
    """

    def __init__(self, params):
        super(GlobalAttDecoder, self).__init__()

        self.decoder_layers = ModuleList([GlobalDecoderLayer(params) for _ in range(params["dec_num_layers"])])

    def forward(self, tgt, memory_key, memory_value, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask,
                use_cache=False, cache=None, predict_last_n_only=False, keep_all_weights=False):
        output = tgt
        cache_t = list()
        all_weights = {
            "self": list(),
            "mix": list()
        }

        for i, dec_layer in enumerate(self.decoder_layers):
            output, weights, weights_self = dec_layer(output, memory_key=memory_key,
                                        memory_value=memory_value,
                                        tgt_mask=tgt_mask,
                                        memory_mask=memory_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask,
                                        predict_last_n_only=predict_last_n_only)
            if use_cache:
                cache_t.append(output)
                if cache is not None:
                    output = torch.cat([cache[i], output], dim=0)
            if keep_all_weights:
                all_weights["self"].append(weights_self)
                all_weights["mix"].append(weights)
        if use_cache:
            cache = torch.cat([cache, torch.stack(cache_t, dim=0)], dim=1) if cache is not None else torch.stack(cache_t, dim=0)

        if predict_last_n_only:
            output = output[-predict_last_n_only:]

        if keep_all_weights:
            return output, all_weights, cache

        return output, weights, cache


class FeaturesUpdater(Module):
    """
    Module that handle 2D positional encoding
    """
    def __init__(self, params):
        super(FeaturesUpdater, self).__init__()
        self.enc_dim = params["enc_dim"]
        self.enc_h_max = params["pe_h_max"]
        self.enc_w_max = params["pe_w_max"]
        self.pe_2d = PositionalEncoding2D(self.enc_dim, self.enc_h_max, self.enc_w_max, params["device"])
        self.use_pe_2d = ("dec_use_pe_2d" not in params) or params["dec_use_pe_2d"]

    def get_pos_features(self, features):
        if self.use_pe_2d:
            return self.pe_2d(features)
        return features