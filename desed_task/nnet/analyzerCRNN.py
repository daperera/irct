import torch
import torch.nn as nn
import einops
import copy
from collections import defaultdict

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
    def __init__(self):
        super(AttentionHead, self).__init__()
        self.strong_classifier = nn.Sequential(nn.Linear(128*2, 10), nn.Sigmoid())
        self.weak_classifier = nn.Sequential(nn.Linear(128*2, 10), nn.Softmax(dim=-1))

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
        cost_i=1,
        cost_r=1,
        latent_index=6,
        crnn_shape=(1, 16, 32, 64, 128, 128, 128, 128, 128),
        **kwargs,):
        super(CRNN, self).__init__()

        # complete crnn
        self.crnn = nn.Sequential(
            Rearrange(pattern="b f t -> b 1 t f"),
            ConvBlock(in_channels=crnn_shape[0], out_channels=crnn_shape[1], pooling_kernel_size=2),           # layer 1
            ConvBlock(in_channels=crnn_shape[1], out_channels=crnn_shape[2], pooling_kernel_size=2),          # layer 2
            ConvBlock(in_channels=crnn_shape[2], out_channels=crnn_shape[3], pooling_kernel_size=(1,2)),      # layer 3
            ConvBlock(in_channels=crnn_shape[3], out_channels=crnn_shape[4], pooling_kernel_size=(1,2)),     # layer 4
            ConvBlock(in_channels=crnn_shape[4], out_channels=crnn_shape[5], pooling_kernel_size=(1,2)),    # layer 5
            ConvBlock(in_channels=crnn_shape[5], out_channels=crnn_shape[6], pooling_kernel_size=(1,2)),    # layer 6
            ConvBlock(in_channels=crnn_shape[6], out_channels=crnn_shape[7], pooling_kernel_size=(1,2)),    # layer 7
            Rearrange(pattern="b c t f -> b t (c f)"),
            RnnBlock(in_channels=crnn_shape[7], hidden_channels=crnn_shape[8]),                             # rnn
            AttentionHead(),                                                            # attention head                                        
        )

        # define invariance and reconstruction metrics
        self.metric_i = nn.MSELoss()


    def forward(self, x, augmentation_dict=None, label_partition_index=0):   
        
        # unpack input
        x_1, x_2, x_3 = x

        # iterate through the network
        loss_dict = defaultdict(float)
        z_1, z_2, z_3 = x_1, x_2, x_3
        for layer_index, layer in enumerate(self.crnn[:-1]):
            
            z_1, z_2, z_3 = layer(z_1), layer(z_2), layer(z_3)

            # compute invariance loss (equivariant mixup + time-shift)
            if augmentation_dict is not None:
                for partition_name in augmentation_dict.keys():
                    mixup_mask = augmentation_dict[partition_name]["mixup_mask"]
                    time_shift_mask = augmentation_dict[partition_name]["time_shift_mask"]
                    permutation_list = augmentation_dict[partition_name]["mixup"]
                    time_shift = augmentation_dict[partition_name]["time_shift"]
                    partition_ratio = mixup_mask.sum() / len(x_1)

                    # handle mixup
                    z_mixup = z_1[mixup_mask].clone()
                    for permutation in permutation_list:
                        z_mixup += z_1[mixup_mask][permutation]
                    loss_i = partition_ratio * self.metric_i(z_mixup, z_2[mixup_mask])
                    loss_dict[f"loss_i/{layer_index}/mixup"] += loss_i

                    # handle time-shift
                    if time_shift is not None:
                        # expecting (bctf) format for z
                        frame_split = int(z_1.shape[2] * time_shift)
                        if len(z_1.shape ) == 3:
                            z_time_shift = torch.cat([z_1[time_shift_mask,-frame_split:,:],
                                                    z_1[time_shift_mask,:-frame_split,:]], dim=1)
                        if len(z_1.shape ) == 4:
                            z_time_shift = torch.cat([z_1[time_shift_mask,:,-frame_split:,:],
                                                z_1[time_shift_mask,:, :-frame_split,:]], dim=2)

                        loss_i = partition_ratio * self.metric_i(z_time_shift, z_3[time_shift_mask])
                        loss_dict[f"loss_i/{layer_index}/time_shift"] += loss_i
                    else:
                        loss_i = partition_ratio * self.metric_i(z_1, z_3)
                        loss_dict[f"loss_i/{layer_index}/augmentation"] += loss_i

        return loss_dict


