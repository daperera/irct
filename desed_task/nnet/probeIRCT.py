import torch
import torch.nn as nn
import einops
import copy
import numpy as np


class GRU(nn.GRU):
    def __init__(self, input_size, hidden_size, num_layers=2, bias=True, batch_first=True, dropout=0, bidirectional=True):
        super(GRU, self).__init__(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

    def forward(self, x):
        return super().forward(x)[0]

class GLU(nn.Module):
    def __init__(self, input_num):
        super(GLU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 3, 1))
        lin = lin.permute(0, 3, 1, 2)
        sig = self.sigmoid(x)
        res = lin * sig
        return res
    
class GLU1d(nn.Module):
    def __init__(self, input_num):
        super(GLU1d, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(input_num, input_num)

    def forward(self, x):
        lin = self.linear(x.permute(0, 2, 1))
        lin = lin.permute(0, 2, 1)
        sig = self.sigmoid(x)
        res = lin * sig
        return res

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cnn_kernel_size=3, stride=1, padding=1, eps=0.001, momentum=0.99, dropout=0.5, pooling_kernel_size=2):
        super(ConvBlock, self).__init__()
        self.tf = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, cnn_kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels, eps=eps, momentum=momentum),
            GLU(input_num=out_channels),
            nn.Dropout(dropout),
            nn.AvgPool2d(kernel_size=pooling_kernel_size)
        )

    def forward(self, x):
        return self.tf(x)
    
class InverseConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cnn_kernel_size=3, stride=1, padding=1, eps=0.001, momentum=0.99, dropout=0.5):
        super(InverseConvBlock, self).__init__()
        self.tf = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, cnn_kernel_size, stride, padding),
            nn.BatchNorm2d(num_features=out_channels, eps=eps, momentum=momentum),
            GLU(input_num=out_channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.tf(x)

class RnnBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2, bias=True, batch_first=True, rnn_dropout=0, bidirectional=True, dropout=0.5):
        super(RnnBlock, self).__init__()
        self.tf = nn.Sequential(
            GRU(in_channels, hidden_channels, num_layers, bias, batch_first, rnn_dropout, bidirectional),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.tf(x)

class AttentionHead(nn.Module):
    def __init__(self, in_channels=128):
        super(AttentionHead, self).__init__()
        self.strong_classifier = nn.Sequential(nn.Linear(in_channels*2, 10), nn.Sigmoid())
        self.weak_classifier = nn.Sequential(nn.Linear(in_channels*2, 10), nn.Softmax(dim=-1))

    def forward(self, x):
        # strong prediction
        strong = self.strong_classifier(x)

        # weak prediction
        weak = self.weak_classifier(x)
        weak = torch.clamp(weak, min=1e-7, max=1)
        weak = (strong * weak).sum(1) / weak.sum(1)

        # rearrange strong prediction
        strong = einops.rearrange(strong, "b t f -> b f t")

        return strong, weak

class Rearrange(nn.Module):
    def __init__(self, pattern, *args, **kwargs):
        super(Rearrange, self).__init__()
        self.pattern = pattern
        self.args = args
        self.kwargs = kwargs
    
    def forward(self, x):
        return einops.rearrange(x, self.pattern, *self.args, **self.kwargs)

class Repeat(nn.Module):
    def __init__(self, pattern, *args, **kwargs):
        super(Repeat, self).__init__()
        self.pattern = pattern
        self.args = args
        self.kwargs = kwargs
    
    def forward(self, x):
        return einops.repeat(x, self.pattern, *self.args, **self.kwargs)


class CRNN(nn.Module):
    def __init__(self,
        n_in_channel=1,
        nclass=10,
        attention=True,
        activation="glu",
        dropout=0.5,
        train_cnn=True,
        rnn_type="BGRU",
        n_RNN_cell=128,
        n_layers_RNN=2,
        dropout_recurrent=0,
        cnn_integration=False,
        freeze_bn=False,
        use_embeddings=False,
        embedding_size=527,
        embedding_type="global",
        frame_emb_enc_dim=512,
        aggregation_type="global",
        latent_index=8,
        crnn_shape=(1, 16, 32, 64, 128, 128, 128, 128, 128),
        head_shape=128,
        pretrained_crnn=None,
        **kwargs,):
        super(CRNN, self).__init__()

        # define encoder
        self.e = copy.deepcopy(pretrained_crnn[:latent_index])

        # estimate encoder output shape
        with torch.no_grad():
            x = torch.randn(1, 128, 626)
            z = self.e(x)
            channel_shape = z.shape[1]*z.shape[3] if latent_index <= 8 else z.shape[2]
            time_shape = z.shape[2] if latent_index <= 8 else z.shape[1]

        # define classifier
        if latent_index <= 8:
            self.c = nn.Sequential(
                Rearrange(pattern="b c t f -> b (c f) t "),
                nn.Linear(time_shape, 156),                                                                # linear
                Rearrange(pattern="b c t -> b t c "),
                RnnBlock(in_channels=channel_shape, hidden_channels=head_shape),                             # rnn
                AttentionHead(head_shape), 
            )
        else:
            self.c = nn.Sequential(
                Rearrange(pattern="b t c -> b c t "),
                nn.Linear(time_shape, 156),                                                                # linear
                Rearrange(pattern="b c t -> b t c "),
                RnnBlock(in_channels=channel_shape, hidden_channels=head_shape),                             # rnn
                AttentionHead(head_shape), 
            )

    def forward(self, x):    
        with torch.no_grad():
            z = self.e(x)
        strong, weak = self.c(z)
        return strong, weak


