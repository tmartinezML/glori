from dataclasses import dataclass


@dataclass
class SWIITSamplerConfig:
    denoiser: str | None = None
    denoiser_ckpt: str = "last-best"
    vae: str = "VQ-VAE-256-DR3opt-FT"
    vae_ckpt: str = "best"
    uncond_denoiser: str | None = None
    uncond_denoiser_ckpt: str = "best"
    image_size: int = 512
    f_vae: int = 4
    latent_dim: int = 3
    div_stride: float = 1
    mask_coverage: float = 0.5
    f_ext_ctxt: int = 2
    device: str = "cuda:0"
    resample_first: bool = False
    do_inpainting: bool = True
    decoding_stride: int | None = None
    scaler: str = "LOFAR_scaler_II"

    @property
    def latent_size(self) -> int:
        return self.image_size // self.f_vae

    @property
    def ext_ctxt_size(self) -> int:
        return self.latent_size * self.f_ext_ctxt

    @property
    def mask_overlap(self) -> int:
        return int(self.mask_coverage * self.latent_size)

    @property
    def decode_overlap(self) -> int:
        return self.image_size - self.img_stride

    @property
    def stride(self) -> int:
        return (self.latent_size - self.mask_overlap) // self.div_stride

    @property
    def img_stride(self) -> int:
        return self.decoding_stride * self.f_vae

    def __post_init__(self):
        # Validate that the image size is divisible by f_vae
        if self.image_size % self.f_vae != 0:
            raise ValueError(
                f"Image size {self.image_size} must be divisible by f_vae {self.f_vae}"
            )

        # Validate that the mask coverage is between 0 and 1
        if not (0 <= self.mask_coverage <= 1):
            raise ValueError(
                f"Mask coverage {self.mask_coverage} must be between 0 and 1"
            )

        # Validate that 1 <= div_stride <= latent_size
        if self.div_stride < 1:
            raise ValueError(f"div_stride {self.div_stride} must be larger than 1.")

        if self.decoding_stride is None:
            self.decoding_stride = self.stride * self.div_stride

        elif self.decoding_stride < 1:
            raise ValueError(
                f"decoding_stride {self.decoding_stride} must be larger than 1."
            )
