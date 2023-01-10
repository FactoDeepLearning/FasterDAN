from torch.nn.functional import log_softmax
from torch.nn import AdaptiveMaxPool2d, Conv1d
from torch.nn import Module


class Decoder(Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.vocab_size = params["vocab_size"]

        self.ada_pool = AdaptiveMaxPool2d((1, None))
        self.end_conv = Conv1d(in_channels=params["enc_size"], out_channels=self.vocab_size+1, kernel_size=1)

    def forward(self, x):
        x = self.ada_pool(x).squeeze(2)
        x = self.end_conv(x)
        return log_softmax(x, dim=1)
