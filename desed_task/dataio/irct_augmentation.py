import torch
import random
import numpy as np
from collections import defaultdict
from torchaudio.transforms import MelSpectrogram
from torchaudio.sox_effects import apply_effects_tensor
import torchaudio
import pandas as pd
import soundfile
import librosa
import scipy
import os

class RandAugment:
    def __init__(self,
                 hparams,
                 device=None,):
        self.feat_params = hparams["feats"]
        self.sample_rate = self.feat_params["sample_rate"]
        # GPU device
        self.device = device

        # Mixup methods
        self.mixup_func = self.mixup_hard if hparams["augs"]["mixup"] == "HARD" else None

        self.mixup_scale = hparams["augs"]["mixup_scale"]
        # Random augmentation methods
        augment_list = []
        if "Time shift" in hparams["augs"]["aug_methods"]:
            augment_list.append(self.time_shift)
            print("Using time shift")
        if "New time shift" in hparams["augs"]["aug_methods"]:
            augment_list.append(self.new_time_shift)
            print("Using new time shift")
        if "Pitch shift" in hparams["augs"]["aug_methods"]:
            augment_list.append(self.pitch_shift)
            print("Using pitch shift")
        if "Time mask" in hparams["augs"]["aug_methods"]:
            augment_list.append(self.time_mask)
            print("Using time mask")
        if "Frequency mask" in hparams["augs"]["aug_methods"]:
            augment_list.append(self.freq_mask)
            print("Using frequency mask")
        if "Filter" in hparams["augs"]["aug_methods"]:
            augment_list.append(self.filter_aug)
        if "Reverberation" in hparams["augs"]["aug_methods"]:
            rir_df_path = hparams["augs"]["rir_df_path"]
            self.rir_df =  pd.read_csv(os.path.expandvars(rir_df_path), header=0, sep="\t")
            augment_list.append(self.reverberation)
            print("Using reverberation")
        if "Compression" in hparams["augs"]["aug_methods"]:
            augment_list.append(self.compression)
            print("Using compression")
        if "Noise" in hparams["augs"]["aug_methods"]:
            augment_list.append(self.noise)
            print("Using noise")
        if "Delay" in hparams["augs"]["aug_methods"]:
            augment_list.append(self.delay)
            print("Using delay")
        if "Overdrive" in hparams["augs"]["aug_methods"]:
            augment_list.append(self.overdrive)
            print("Using overdrive")
        if augment_list:
            self.augment_selector = lambda: np.random.choice(augment_list)
        else:
            self.augment_selector = None
        self.augment_scale = hparams["augs"]["aug_scale"]
        print("RandAugment scale:", self.augment_scale)

        # Augmentation for unsupervised data
        self.aug_unsup = hparams["augs"]["unsup"]

        # Generate masks for both sync, weak and unsup data
        batch_num = sum(hparams["training"]["batch_size"])

        # handle the case where there is no weak data
        #indx_synth, indx_weak, indx_unlabelled = hparams["training"]["batch_size"]
        indx_synth = np.sum(hparams["training"]["batch_size"][0])
        indx_weak = np.sum(hparams["training"]["batch_size"][1]) if len(hparams["training"]["batch_size"]) > 1 else 0
        
        
        strong_mask = torch.zeros(batch_num).to(device)
        weak_mask = torch.zeros(batch_num).to(device)
        unsup_mask = torch.zeros(batch_num).to(device)
        strong_mask[:indx_synth] = 1
        weak_mask[indx_synth: indx_weak + indx_synth] = 1
        unsup_mask[indx_weak + indx_synth:] = 1
        self.weak_mask = weak_mask.bool()
        self.strong_mask = strong_mask.bool()
        self.unsup_mask = unsup_mask.bool()

        # Get mel spectrum extractor
        self.mel_spec = self.__get_mel_spec()

    def random_augment(self, batch: torch.FloatTensor, labels: torch.FloatTensor, epoch=None, equivariance=True):
        distortions = defaultdict()
        masks = defaultdict()
        if self.mixup_func is None and self.augment_selector is None:
            masks["weak"] = self.weak_mask
            masks["strong"] = self.strong_mask
            masks["unsup"] = self.unsup_mask
            return self.mel_spec(batch), labels, masks, distortions

        # Augmentation for weak data
        if self.weak_mask.any():
            aug_weak_data, aug_weak_labels, distortions["weak"] = self._augment(batch[self.weak_mask].clone(),
                                                                            labels[self.weak_mask].clone(), equivariance=equivariance)
        else:
            distortions["weak"] = {"mixup": None, "augments": None}

        # Augmentation for strong data
        aug_strong_data, aug_strong_labels, distortions["strong"] = self._augment(batch[self.strong_mask].clone(),
                                                                                  labels[self.strong_mask].clone(), equivariance=equivariance)
        
        # Update weak mask and strong mask
        if self.weak_mask.any():
            strong_mask = torch.cat([self.strong_mask,
                                 torch.ones(len(aug_strong_data)).to(self.device),
                                 torch.zeros(len(aug_weak_data)).to(self.device)], dim=0).bool()
        else:
            strong_mask = torch.cat([self.strong_mask,
                                 torch.ones(len(aug_strong_data)).to(self.device)], dim=0).bool()
        if self.weak_mask.any():
            weak_mask = torch.cat([self.weak_mask,
                               torch.zeros(len(aug_strong_labels)).to(self.device),
                               torch.ones(len(aug_weak_labels)).to(self.device)], dim=0).bool()
        else:
            weak_mask = torch.cat([self.weak_mask,
                               torch.zeros(len(aug_strong_labels)).to(self.device)], dim=0).bool()
        # Update the augmented data and labels to the batch and labels.
        if self.weak_mask.any():
            aug_batch = torch.cat([self.mel_spec(batch),
                               aug_strong_data,
                               aug_weak_data], dim=0)
        else:
            aug_batch = torch.cat([self.mel_spec(batch),
                               aug_strong_data], dim=0)
        if self.weak_mask.any():
            aug_labels = torch.cat([labels,
                                    aug_strong_labels,
                                    aug_weak_labels], dim=0)
        else:
            aug_labels = torch.cat([labels,
                                aug_strong_labels], dim=0)

        

        # Augmentation for unsup data
        if self.aug_unsup:
            aug_unsup_data, aug_unsup_labels, unsup_distortions = self._augment(batch[self.unsup_mask].clone(),
                                                                            labels[self.unsup_mask].clone(), equivariance=equivariance)
            aug_batch = torch.cat([aug_batch,
                                aug_unsup_data], dim=0)
            aug_labels = torch.cat([aug_labels,
                                aug_unsup_labels], dim=0)
            # Update the masks
            weak_mask = torch.cat([weak_mask,
                                torch.zeros(len(aug_unsup_data)).to(self.device)], dim=0).bool()
            strong_mask = torch.cat([strong_mask,
                                    torch.zeros(len(aug_unsup_data)).to(self.device)], dim=0).bool()
            unsup_mask = torch.cat([self.unsup_mask,
                                    torch.zeros(len(aug_strong_data)).to(self.device),
                                    torch.zeros(len(aug_weak_data)).to(self.device),
                                    torch.ones(len(aug_unsup_data)).to(self.device)], dim=0).bool()
        else:
            unsup_distortions = {"mixup": None, "augments": None}
            unsup_mask = torch.zeros_like(strong_mask).bool()

        distortions["unsup"] = unsup_distortions
        masks["unsup"] = unsup_mask
        masks["weak"] = weak_mask
        masks["strong"] = strong_mask
        return aug_batch, aug_labels, masks, distortions

    def _augment(self, data, label, equivariance=True):
        distortion_mixup = None
        # Mixup:
        if self.mixup_func is not None:
            data_mixup, label_mixup, distortion_mixup = self.mixup_hard(utt=data, labels=label)

        if self.augment_selector is None:
            return data_mixup, label_mixup, {"mixup": distortion_mixup, "augments": None}
        else:
            # Random augmentation
            randaug_func = self.augment_selector()
            data_randaug, label_randaug, distortion_randaug = randaug_func(utt=data, labels=label, equivariance=equivariance)

            if not equivariance:
                label_randaug = label

            if self.mixup_func is None:
                aug_data = data_randaug
                aug_label = label_randaug
            else:
                aug_data = torch.cat([data_mixup, data_randaug], dim=0)
                aug_label = torch.cat([label_mixup, label_randaug], dim=0)

            return aug_data, aug_label, {"mixup": distortion_mixup, "augments": distortion_randaug}

    def time_shift(self, utt: torch.FloatTensor, labels: torch.FloatTensor, equivariance=True):
        """Time shift utterance as well as labels in time domain.
        Input:
            Utterances              -   [bsz (12 for weak/strong), num_samples (16000Hz * 10s)]
            Labels                  -   [bsz (12 for weak/strong), num_class (10), num_clips (156)]
        Output:
            Augmented utterance     -   [bsz (12 for weak/strong), feat_size (128), num_frames (626)]
            Shifted labels          -   [bsz (12 for weak/strong), num_class (10), num_clips (156)]
            Distortion              -   ("Time shift", frame_split_point)
        """
        with torch.no_grad():
            scale = random.randint(1, self.augment_scale) 
            shift_scale = scale / 10 if equivariance else scale / 1000 # lower shift if invariance objective
            # Shifting data in time-domain
            utt_split = int(utt.shape[1] * shift_scale)
            utt_aug = torch.cat([utt[:, -utt_split:], utt[:, :-utt_split]], dim=-1)

            # Shifting labels in time-axis
            frame_split = int(labels.shape[2] * shift_scale)
            labels_aug = torch.cat([labels[:, :, -frame_split:], labels[:, :, :-frame_split]], dim=-1)
            # Transform time-domain -> freq-domain
            utt_spec = self.mel_spec(utt_aug)
        return utt_spec, labels_aug, ("Time shift", frame_split)

    def new_time_shift(self, utt: torch.FloatTensor, labels: torch.FloatTensor, equivariance=True):
        """Time shift utterance as well as labels in time domain.
        Input:
            Utterances              -   [bsz (12 for weak/strong), num_samples (16000Hz * 10s)]
            Labels                  -   [bsz (12 for weak/strong), num_class (10), num_clips (156)]
        Output:
            Augmented utterance     -   [bsz (12 for weak/strong), feat_size (128), num_frames (626)]
            Shifted labels          -   [bsz (12 for weak/strong), num_class (10), num_clips (156)]
            Distortion              -   ("Time shift", frame_split_point)
        """
        with torch.no_grad():
            scale = random.randint(1, self.augment_scale) 
            shift_scale = scale / 10 if equivariance else scale / 1000 # lower shift if invariance objective
            # Shifting data in time-domain
            utt_split = int(utt.shape[1] * shift_scale)
            utt_aug = torch.cat([utt[:, -utt_split:], utt[:, :-utt_split]], dim=-1)

            # Shifting labels in time-axis
            frame_split = int(labels.shape[2] * shift_scale)
            labels_aug = torch.cat([labels[:, :, -frame_split:], labels[:, :, :-frame_split]], dim=-1)
            # Transform time-domain -> freq-domain
            utt_spec = self.mel_spec(utt_aug)
        return utt_spec, labels_aug, ("Time shift", shift_scale)

    def pitch_shift(self, utt: torch.FloatTensor, labels: torch.FloatTensor, equivariance=True):
        """Pitch shifting by vocoder method inside sox.
        Input:
            Utterances              -   [bsz (12 for weak/strong), num_samples (16000Hz * 10s)]
            Labels                  -   [bsz (12 for weak/strong), num_class (10), num_clips (156)]
        Output:
            Augmented utterances    -   [bsz (12 for weak/strong), feat_size (128), num_frames (626)]
            Labels (not changed)    -   [bsz (12 for weak/strong), num_class (10), num_clips (156)]
            Distortion              -   ("Pitch shift", None)
        """
        with torch.no_grad():
            scale = random.randint(1, self.augment_scale)
            level = scale * 50 * self.rand_sign()
            effect = [
                ["pitch", str(level)],
                ["rate", str(self.sample_rate)]
            ]
            utt_aug, _ = apply_effects_tensor(utt.clone().cpu(), self.sample_rate, effect, channels_first=True)
        return self.mel_spec(utt_aug.to(self.device)), labels, ("Pitch shift", None)

    def time_mask(self, utt: torch.FloatTensor, labels: torch.FloatTensor, equivariance=True):
        """Time mask in time domain, randomly mask some clips.
        Input:
            Utterances              -   [bsz (12 for weak/strong), num_samples (16000Hz * 10s)]
            Labels                  -   [bsz (12 for weak/strong), num_class (10), num_clips (156)]
        Output:
            Augmented utterances    -   [bsz (12 for weak/strong), feat_size (128), num_frames (626)]
            Labels (not changed)    -   [bsz (12 for weak/strong), num_class (10), num_clips (156)]
            Distortion              -   ("Time_mask", None)
        """
        with torch.no_grad():
            bsz = utt.shape[0]
            mask_interval = int(0.01 * utt.shape[1])
            scale = random.randint(1, self.augment_scale)
            # scale = self.augment_scale  # NOT USING RAND SCALE !
            random_starts = [
                [random.randint(0, utt.shape[1] - mask_interval) for _ in range(scale * 5)]
                for _ in range(bsz)]
            mask_mat = torch.ones_like(utt).to(self.device)
            for i, starts in zip(range(bsz), random_starts):
                for start in starts:
                    mask_mat[i, start: start + mask_interval] = 0

        return self.mel_spec(utt * mask_mat), labels, ("Time mask", None)

    def freq_mask(self, utt: torch.FloatTensor, labels: torch.FloatTensor, equivariance=True):
        """Frequency mask in T-F domain, randomly mask some narrow bands.
        Input:
            Utterances              -   [bsz (12 for weak/strong), num_samples (16000Hz * 10s)]
            Labels                  -   [bsz (12 for weak/strong), num_class (10), num_clips (156)]
        Output:
            Augmented utterances    -   [bsz (12 for weak/strong), feat_size (128), num_frames (626)]
            Labels (not changed)    -   [bsz (12 for weak/strong), num_class (10), num_clips (156)]
            Distortion              -   ("Frequency_mask", None)
        """
        with torch.no_grad():
            bsz = utt.shape[0]
            mel_spectrogram = self.mel_spec(utt)
            mask_bands = torch.ones_like(mel_spectrogram).to(self.device)
            scale = random.randint(1, self.augment_scale) * 2   # Maximal mask 20 dims in LogMel
            mask_band_start = np.random.choice(np.arange(mel_spectrogram.shape[1] - scale), size=bsz, replace=False)
            for i, start in zip(range(bsz), mask_band_start):
                mask_bands[i, start: start+scale, :] = 0
        return mel_spectrogram * mask_bands, labels, ("Freq mask", None)

    def filter_aug(self, utt: torch.FloatTensor, labels: torch.FloatTensor, equivariance=True):
        """Filter augmentation referred to: https://github.com/frednam93/FilterAugSED.
        Input:
            Utterances              -   [bsz (12 for weak/strong), num_samples (16000Hz * 10s)]
            Labels                  -   [bsz (12 for weak/strong), num_class (10), num_clips (156)]
        Output:
            Augmented utterance     -   [bsz (12 for weak/strong), feat_size (128), num_frames (626)]
            Labels (not changed)    -   [bsz (12 for weak/strong), num_class (10), num_clips (156)]
            Distortion              -   ("Filter", None)
        """
        with torch.no_grad():
            db_range = (-7.5, 6)
            spec = self.mel_spec(utt)
            batch_size, n_freq_bin, _ = spec.shape
            n_freq_band = torch.randint(low=2, high=5, size=(1,)).item()  # [low, high)
            band_bndry_freqs = torch.cat((torch.tensor([0]),
                                          torch.sort(torch.randint(1, n_freq_bin - 1, (n_freq_band - 1,)))[0],
                                          torch.tensor([n_freq_bin])))
            band_factors = torch.rand((batch_size, n_freq_band)).to(spec) * (db_range[1] - db_range[0]) + \
                           db_range[0]

            band_factors = 10 ** (band_factors / 20)

            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(spec)
            for i in range(n_freq_band):
                freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = band_factors[:, i].unsqueeze(
                    -1).unsqueeze(-1)
            return spec * freq_filt.to(spec), labels, ("Filter", None)

    def mixup_hard(self, utt: torch.FloatTensor, labels: torch.FloatTensor):
        """Hard mixup: mix two samples up by addition of data as well as labels.
        Input:
            Utterances              -   [bsz (12 for weak/strong), num_samples (16000Hz * 10s)]
            Labels                  -   [bsz (12 for weak/strong), num_class (10), num_clips (156)]
        Output:
            Augmented utterances    -   [bsz (12 for weak/strong), feat_size (128), num_frames (626)]
            Mixed labels            -   [bsz (12 for weak/strong), num_class (10), num_clips (156)]
            Distortion              -   ("Hard mixup", perm_list)
        """
        with torch.no_grad():
            perm_list = []
            mixed_data = utt.clone()
            mixed_target = labels.clone()
            batch_size = utt.shape[0]
            duplicate = np.random.choice(self.mixup_scale)
            for _ in range(duplicate):
                perm = torch.randperm(batch_size)
                mixed_data += utt[perm]
                mixed_target = torch.clamp(mixed_target + labels[perm], min=0, max=1)
                perm_list.append(perm)
            return self.mel_spec(mixed_data), mixed_target, ("Hard mixup", perm_list)

    def reverberation(self, utt: torch.FloatTensor, labels: torch.FloatTensor, equivariance=True):
        # transform hyperparameters
        target_sr = 16000
        
        with torch.no_grad():
            scale = None # No scale for reverberation

            # select random index
            index = np.random.randint(len(self.rir_df))
            
            # load rir
            filepath = self.rir_df.iloc[index]["filepath"]
            rir, sr = soundfile.read(filepath)

            # reduce channels
            if rir.ndim > 1:
                rir = np.mean(rir, axis=1)
            
            # resample
            if sr != target_sr:
                rir = librosa.resample(rir, orig_sr=sr, target_sr=target_sr)
            
            # apply reverb (on cpu)
            device = utt.device
            utt = torch.tensor(scipy.signal.fftconvolve(utt.to("cpu"), rir[None,:], mode="same")).float().to(device)

            spec = self.mel_spec(utt)

        return spec, labels, ("Reverberation", None)

    def compression(self, utt: torch.FloatTensor, labels: torch.FloatTensor, equivariance=True):
        with torch.no_grad():
            scale = self.augment_scale
            enhancement_amount =  np.random.randint(low=0, high=scale*2)
            utt = torchaudio.functional.contrast(utt, enhancement_amount)
            spec = self.mel_spec(utt)
        return spec, labels, ("Compression", enhancement_amount)

    def noise(self, utt: torch.FloatTensor, labels: torch.FloatTensor, equivariance=True):
        # transform hyperparameters
        sr = 16000
        snr_min=15
        snr_max=50 
        
        with torch.no_grad():
            scale = self.augment_scale

            # generate noise
            noise = torch.normal(torch.zeros(sr), torch.ones(sr)).to(utt.device)
            
            # generate psd profile
            f_decay = np.random.choice([0, 1, 2, -1, -2])
            spec = torch.fft.rfft(noise)
            mask = torch.pow(torch.linspace(1, (sr / 2) ** 0.5, spec.shape[0]).to(utt.device), -f_decay)
            
            # apply psd
            spec *= mask
            #noise = self.rms_normalize(torch.fft.irfft(spec).reshape(1, -1)).squeeze()
            noise = torch.fft.irfft(spec).reshape(1, -1).squeeze()
            noise /= torch.sqrt(torch.mean(torch.square(noise)))
            
            # get correct shape
            noise = torch.cat([noise] * int(np.ceil(utt.shape[1]/noise.shape[0])), axis=0)[:utt.shape[1]]
            
            # scale noise
            snr = np.random.uniform(snr_min, snr_min + (snr_max - snr_min) / 5 * scale)
            gain = torch.sqrt(torch.square(utt).sum() / (10 ** (snr / 10) * torch.square(noise).sum()))
            noise *= gain

            # add noise
            utt += noise 

            spec = self.mel_spec(utt)
        
        return spec, labels, ("Noise", snr)

    def delay(self, utt: torch.FloatTensor, labels: torch.FloatTensor, equivariance=True):
        # transform hyperparameters
        min_delay=30
        max_delay=100
        min_gain=0.3
        max_gain=1.0
        sr = 16000

        with torch.no_grad():
            scale = self.augment_scale

            # generate offset & gain
            offset = int(np.random.uniform(low=min_delay, high=min_delay + (max_delay - min_delay) / 5 * scale) * sr / 1000)
            gain = np.random.uniform(low=min_gain, high=min_gain + (max_gain - min_gain) / 5 * scale)
            
            # shift signal
            start = torch.zeros(utt.shape[0], offset).to(utt.device)
            end = utt[:,:-offset]
            utt_shifted = torch.cat((start, end), dim=1)

            # apply delay
            utt = utt + utt_shifted * gain

            spec = self.mel_spec(utt)

        return spec, labels, ("Delay", (offset, gain))

    def overdrive(self, utt: torch.FloatTensor, labels: torch.FloatTensor, equivariance=True):
        # transform hyperparameters
        min_gain=0.3
        max_gain=1.0
        min_colour=0.3
        max_colour=1.0
        
        with torch.no_grad():
            scale = self.augment_scale

            gain = np.random.uniform(low=min_gain, high=min_gain + (max_gain - min_gain) / 5 * scale)
            colour = np.random.uniform(low=min_colour, high=min_colour + (max_colour - min_colour)/5*scale)
            utt = torchaudio.functional.overdrive(utt, gain=gain, colour=colour)

            spec = self.mel_spec(utt)

        return spec, labels, ("Overdrive", (gain, colour))


    @staticmethod
    def rand_sign():
        return 1 if torch.randn(1) > 0.5 else -1

    def __get_mel_spec(self):
        return MelSpectrogram(
            sample_rate=self.feat_params["sample_rate"],
            n_fft=self.feat_params["n_window"],
            win_length=self.feat_params["n_window"],
            hop_length=self.feat_params["hop_length"],
            f_min=self.feat_params["f_min"],
            f_max=self.feat_params["f_max"],
            n_mels=self.feat_params["n_mels"],
            window_fn=torch.hamming_window,
            wkwargs={"periodic": False},
            power=1,
        ).to(self.device)