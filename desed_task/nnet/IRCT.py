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
        latent_index=7,
        crnn_shape=(1, 16, 32, 64, 128, 128, 128, 128, 128),
        **kwargs,):
        super(CRNN, self).__init__()

        # complete crnn
        crnn = nn.Sequential(
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

        # complete decoder
        if latent_index == 8:
            decoder = nn.Sequential(
                InverseConvBlock(in_channels=256, out_channels=128),  # layer 6
                Repeat(pattern="b c t f -> b c t (f 2)"),
                InverseConvBlock(in_channels=128, out_channels=128),  # layer 5
                Repeat(pattern="b c t f -> b c t (f 2)"),
                InverseConvBlock(in_channels=128, out_channels=128),  # layer 5
                Repeat(pattern="b c t f -> b c t (f 2)"),
                InverseConvBlock(in_channels=128, out_channels=128),  # layer 4
                Repeat(pattern="b c t f -> b c t (f 2)"),
                InverseConvBlock(in_channels=128, out_channels=64),   # layer 3
                Repeat(pattern="b c t f -> b c t (f 2)"),
                InverseConvBlock(in_channels=64, out_channels=32),    # layer 2
                Repeat(pattern="b c t f -> b c (t 2) (f 2)"),
                InverseConvBlock(in_channels=32, out_channels=16),    # layer 1
                Repeat(pattern="b c t f -> b c (t 2) (f 2)"),
                InverseConvBlock(in_channels=16, out_channels=1, padding=(0,1)),     # layer 0
                Repeat(pattern="b 1 t f -> b f t"),
            )
        else:
            decoder = nn.Sequential(
                InverseConvBlock(in_channels=256, out_channels=128),  # layer 6
                Repeat(pattern="b c t f -> b c t (f 2)"),
                InverseConvBlock(in_channels=128, out_channels=128),  # layer 5
                Repeat(pattern="b c t f -> b c t (f 2)"),
                InverseConvBlock(in_channels=128, out_channels=128),  # layer 4
                Repeat(pattern="b c t f -> b c t (f 2)"),
                InverseConvBlock(in_channels=128, out_channels=64),   # layer 3
                Repeat(pattern="b c t f -> b c t (f 2)"),
                InverseConvBlock(in_channels=64, out_channels=32),    # layer 2
                Repeat(pattern="b c t f -> b c (t 2) (f 2)"),
                InverseConvBlock(in_channels=32, out_channels=16),    # layer 1
                Repeat(pattern="b c t f -> b c (t 2) (f 2)"),
                InverseConvBlock(in_channels=16, out_channels=1, padding=(0,1)),     # layer 0
                Repeat(pattern="b 1 t f -> b f t"),
            )

        # define invariant encoder, residual encoder, classifier & decoder
        self.e_i = copy.deepcopy(crnn[:latent_index])
        self.e_r = copy.deepcopy(crnn[:latent_index])
        self.c = copy.deepcopy(crnn[latent_index:])
        self.d = copy.deepcopy(decoder)

        # define invariance and reconstruction metrics
        self.metric_i = nn.MSELoss()
        self.metric_r = nn.MSELoss()
        self.cost_i = cost_i
        self.cost_r = cost_r

    def forward(self, x, augmentation_dict=None, label_partition_index=0):   
        # unpack input
        x_1, x_2, x_3 = x

        # encode 
        z_i_1, z_i_2, z_i_3 = self.e_i(x_1), self.e_i(x_2), self.e_i(x_3) # original, mixup, audio_augmentations
        z_r = self.e_r(x_3) if self.cost_r > 0 else None

        # classify (using the partition corresponding to the label)
        label_partition_dict = {0: z_i_1, 1: z_i_2, 2: z_i_3}
        if label_partition_index >= 0:
            z_c = label_partition_dict[label_partition_index]
            strong, weak = self.c(z_c)
        else:
            strong, weak = [], []
            for z_i in label_partition_dict.values():
                strong_i, weak_i = self.c(z_i)
                strong.append(strong_i)
                weak.append(weak_i)
            strong, weak = torch.cat(strong, dim=0), torch.cat(weak, dim=0)

        # decode
        x_d = self.d(torch.cat([z_i_3, z_r], dim=1)) if self.cost_r > 0 else None  
        
        # compute reconstruction loss
        loss_r = self.cost_r * self.metric_r(x_3, x_d) if self.cost_r > 0 else torch.tensor(0., device=x_1.device)

        # compute invariance loss (equivariant mixup + time-shift)
        if type(z_i_1) != tuple:
            z_i_1 = (z_i_1,)
            z_i_2 = (z_i_2,)
            z_i_3 = (z_i_3,)
        loss_i = torch.zeros_like(loss_r)
        if augmentation_dict is not None:
            for partition_name in augmentation_dict.keys():
                mixup_mask = augmentation_dict[partition_name]["mixup_mask"]
                augments_mask = augmentation_dict[partition_name]["augments_mask"]
                permutation_list = augmentation_dict[partition_name]["mixup"]
                time_shift = augmentation_dict[partition_name]["time_shift"]
                partition_ratio = mixup_mask.sum() / len(x_1)

                # handle mixup
                for z1, z2 in zip(z_i_1, z_i_2):
                    z_mixup = z1[mixup_mask].clone()
                    for permutation in permutation_list:
                        z_mixup += z1[mixup_mask][permutation]
                    loss_i += self.cost_i * partition_ratio * self.metric_i(z_mixup, z2[mixup_mask])
                
                # handle time-shift
                if time_shift is not None:
                    # only apply embedding with temporal dimension
                    z1, z3 = z_i_1[0], z_i_3[0]

                    # expecting (bctf) format for z
                    frame_split = int(z1.shape[2] * time_shift)
                    if len(z1.shape ) == 3:
                        z_time_shift = torch.cat([z1[augments_mask,-frame_split:,:],
                                                z1[augments_mask,:-frame_split,:]], dim=1)
                    if len(z1.shape ) == 4:
                        z_time_shift = torch.cat([z1[augments_mask,:,-frame_split:,:],
                                            z1[augments_mask,:, :-frame_split,:]], dim=2)

                    loss_i += self.cost_i * partition_ratio * self.metric_i(z_time_shift, z3[augments_mask])

                # handle other augmentations (invariance cost instead of equivariance)
                # else:
                #     for z1, z3 in zip(z_i_1, z_i_3):
                #         loss_i += self.cost_i * partition_ratio * self.metric_i(z1[augments_mask], z3[augments_mask])

        loss_dict = {"loss_i": loss_i, "loss_r": loss_r}

        return strong, weak, loss_dict


