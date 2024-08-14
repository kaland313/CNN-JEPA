# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Attribution Information:
# This script is based on the implementation by Meta Platforms, Inc. and affiliates.
# The original implementation can be found at 
# https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py
# The original implementation is licensed under the CC BY-NC 4.0 license found at 
# https://github.com/facebookresearch/ijepa/blob/main/LICENSE


import math

from multiprocessing import Value

import torch

class MultiBlockMask(object):

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=32,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        num_enc_blocks=1,
        num_pred_blocks=4,
        min_keep=4,
        allow_overlap=False
    ):
        super(MultiBlockMask, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = num_enc_blocks
        self.npred = num_pred_blocks
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks        
        # self._itr_counter = Value('i', -1)  # collator is shared across worker processes
        self.__itr_counter = 0

    def step(self):
        # i = self._itr_counter
        # with i.get_lock():
        #     i.value += 1
        #     v = i.value
        self.__itr_counter += 1
        return self.__itr_counter


    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h > self.height:
            h -= 1
        while w > self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            if self.height - h > 0 and self.width - w > 0:
                # -- Sample block top-left corner
                top = torch.randint(0, self.height - h, (1,))
                left = torch.randint(0, self.width - w, (1,))
            else:
                top = 0
                left = 0
                                
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1

            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch_size):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(batch_size):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                print(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        # This is only needed for ViTs
        # collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        # collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]

        # merge blocks to mask
        collated_masks_pred = [torch.cat(cm_list).unique() for cm_list in collated_masks_pred]
        collated_masks_enc = [torch.cat(cm_list).unique() for cm_list in collated_masks_enc]

        # Convert to width x height binary mask from index
        collated_masks_pred = [self.indeces_to_hw_mask(m) for m in collated_masks_pred]
        collated_masks_enc = [self.indeces_to_hw_mask(m) for m in collated_masks_enc]

        # Stack masks
        collated_masks_enc = torch.stack(collated_masks_enc)
        collated_masks_pred = torch.stack(collated_masks_pred)

        return collated_masks_enc, collated_masks_pred
    
    def indeces_to_hw_mask(self, indeces):
        mask = torch.zeros((self.height*self.width), dtype=torch.int32)
        mask[indeces] = 1
        mask = mask.view(self.height, self.width)
        return mask