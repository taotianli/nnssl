"""3-D nnSSL loader that carries anatomy support through spatial transforms.

The standard loader intentionally ignores the optional anatomy mask. Prior
experiments need it aligned with the sampled image crop, so this isolated
loader returns it as ``seg`` without changing any existing trainer.
"""

from __future__ import annotations

import numpy as np

from nnssl.ssl_data.dataloading.base_data_loader import nnsslDataLoaderBase


class nnsslPriorDataLoader3D(nnsslDataLoaderBase):
    """Return image crops plus an aligned optional binary anatomy mask."""

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        anat_all = np.zeros(self.data_shape, dtype=np.int16)
        availability = np.zeros((len(selected_keys),), dtype=np.uint8)
        case_properties = []

        for batch_index, image_identifier in enumerate(selected_keys):
            data, _anon, anat, properties = self._data[image_identifier]
            if anat is not None:
                availability[batch_index] = 1
            case_properties.append(properties)

            shape = data.shape[1:]
            bbox_lbs, bbox_ubs = self.get_bbox(shape)
            valid_lbs = [max(0, lower) for lower in bbox_lbs]
            valid_ubs = [min(size, upper) for size, upper in zip(shape, bbox_ubs)]
            spatial_crop = tuple(slice(i, j) for i, j in zip(valid_lbs, valid_ubs))
            crop = (slice(0, data.shape[0]), *spatial_crop)
            padding = [
                (-min(0, lower), max(upper - size, 0))
                for lower, upper, size in zip(bbox_lbs, bbox_ubs, shape)
            ]
            # Blosc2 arrays are memory mapped. Slice before np.asarray so each
            # augmentation worker only decompresses its sampled crop.
            data_crop = np.asarray(data[crop], dtype=np.float32)
            if anat is None:
                anat_crop = np.zeros(data_crop.shape, dtype=np.int16)
            elif len(anat.shape) == len(data.shape) - 1:
                anat_crop = np.asarray(anat[spatial_crop], dtype=np.int16)[None]
            else:
                anat_crop = np.asarray(
                    anat[(slice(0, anat.shape[0]), *spatial_crop)], dtype=np.int16
                )
            if anat_crop.shape[0] != data_crop.shape[0]:
                # Primus pretraining is single-channel; retain one support
                # channel rather than concatenating unrelated mask channels.
                anat_crop = anat_crop[:1]
            data_all[batch_index] = np.pad(
                data_crop, ((0, 0), *padding), "constant", constant_values=0
            )
            anat_all[batch_index] = np.pad(
                anat_crop, ((0, 0), *padding), "constant", constant_values=0
            )

        return {
            "data": data_all,
            "seg": anat_all,
            "prior_mask_available": availability,
            "properties": case_properties,
            "keys": selected_keys,
        }
