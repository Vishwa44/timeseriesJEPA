import numpy as np
from multiprocessing import Value
import torch

class TimeSeriesMaskCollator(object):
    def __init__(
        self,
        seq_len,
        patch_size=16,
        stride=8,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        nenc=1,
        npred=1,
        min_keep=4,
        allow_overlap=False,
    ):
        super(TimeSeriesMaskCollator, self).__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = ((seq_len - patch_size) // stride) + 1
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes
    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_mask_size(self, scale):
        min_s, max_s = scale
        mask_scale = min_s + torch.rand(1).item() * (max_s - min_s)
        mask_size = int(self.num_patches * mask_scale)
        return mask_size

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        mask_length = b_size

        def constrain_mask(mask, tries=0):
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            start = torch.randint(0, self.num_patches - mask_length + 1, (1,))
            mask = torch.zeros(self.num_patches, dtype=torch.int32)
            mask[start:start+mask_length] = 1

            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask).squeeze()

            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    print(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')

        mask_complement = torch.ones(self.num_patches, dtype=torch.int32)
        mask_complement[start:start+mask_length] = 0

        return mask, mask_complement

    def __call__(self, batch):
        seq_x, label, inputmask = zip(*batch)
        seq_x = torch.from_numpy(np.stack(seq_x)).float().unsqueeze(2)
        label = torch.from_numpy(np.stack(label)).float().unsqueeze(2)
        inputmask = torch.from_numpy(np.stack(inputmask)).float().unsqueeze(2)

        B, _, N = seq_x.shape

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        
        pred_mask_size = self._sample_mask_size(self.pred_mask_scale)
        enc_mask_size = self._sample_mask_size(self.enc_mask_scale)

        collated_masks_pred, collated_masks_enc = [], []
        # max_keep_pred, max_keep_enc = 0, 0
        min_keep_pred = self.num_patches
        min_keep_enc = self.num_patches
        for b in range(B):
            collated_masks_pred_n, collated_masks_enc_n = [], []
            for n in range(N):
                masks_p, masks_C = [], []
                for _ in range(self.npred):
                    mask, mask_C = self._sample_block_mask(pred_mask_size)
                    masks_p.append(mask)
                    masks_C.append(mask_C)
                    min_keep_pred = min(min_keep_pred, len(mask))
                collated_masks_pred_n.append(masks_p)

                acceptable_regions = masks_C if not self.allow_overlap else None
                
                masks_e = []
                for _ in range(self.nenc):
                    mask, _ = self._sample_block_mask(enc_mask_size, acceptable_regions)
                    masks_e.append(mask)
                    min_keep_enc = min(min_keep_enc, len(mask))
                collated_masks_enc_n.append(masks_e)
            collated_masks_pred.append(collated_masks_pred_n)
            collated_masks_enc.append(collated_masks_enc_n)
        collated_masks_pred = [[[c[:min_keep_pred] for c in cm] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = [
            torch.stack([
                torch.stack([sample[n_var][n_mask] for n_var in range(N)])
                for sample in collated_masks_pred
            ])
            for n_mask in range(self.npred)
        ]
        collated_masks_enc = [[[c[:min_keep_enc] for c in cm] for cm in cm_list] for cm_list in collated_masks_enc]
        # collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_enc = [
            torch.stack([
                torch.stack([sample[n_var][n_mask] for n_var in range(N)])
                for sample in collated_masks_enc
            ])
            for n_mask in range(self.nenc)
        ]
        return seq_x, label, inputmask, collated_masks_enc, collated_masks_pred
