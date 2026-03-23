from deprecated.micromap_webdataset import MicromapDataset


class MicromapsDatasetLDM(MicromapDataset):
    def __init__(
        self,
        dset,
        split="train",
        mode="LDM-train-mask",
        custom_transform=None,
        output_tuple=("npy",),
        **kwargs,
    ):
        super().__init__(
            dset,
            split=split,
            mode=mode,
            custom_transform=custom_transform,
            scaler=None,
            output_tuple=output_tuple,
            **kwargs,
        )
