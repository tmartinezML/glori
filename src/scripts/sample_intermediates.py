from models.diffusion.diffusion import get_sampling_noise_levels
import models.utils as mutil
from utils.devices import distribute_model
from tqdm import tqdm
import torch

batch_size = 16

# Generate time steps (= noise levels).
timesteps = 25
sigma_min = 2e-3
sigma_max = 80
rho = 7
sigma_steps = get_sampling_noise_levels(
    timesteps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho
)
# sigma_steps = sigma_steps.to('cuda:1')

# Prepare sampling loop.
# latents = torch.tensor(z, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
latents = torch.randn([batch_size, 1, 100, 100], dtype=torch.float32)
imgs = []
denoiser_outputs = []
x_next = latents * sigma_steps[0]  # Generate initial sample at t_0
imgs.append(x_next.cpu().numpy())

model = mutil.load_model("Prototypes_Model")
# model, _ = distribute_model(model, n_devices=2, device_ids=[1, 2])
model = model.eval().to("cuda:1")

# Sampling loop:
for i, (sigma_cur, sigma_next) in tqdm(
    enumerate(zip(sigma_steps[:-1], sigma_steps[1:])),
    desc="Sampling...",
    total=timesteps,
):
    # Update current image (= output from previous iteration)
    x_cur = x_next

    # Calculate denoised image with forward model pass
    # print(x_cur.shape, sigma_cur.expand(batch_size, 1).shape)
    x_cur = x_cur.to("cuda:1")
    sigma_cur = sigma_cur.to("cuda:1")

    denoised = model(x_cur, sigma_cur.expand(batch_size))
    denoiser_outputs.append(denoised.cpu().detach().numpy())

    # Release GPU memory
    x_cur = x_cur.cpu()
    sigma_cur = sigma_cur.cpu()
    denoised = denoised.cpu()

    # Score estimate
    d_cur = (x_cur - denoised) / sigma_cur

    # Euler step
    x_next = x_cur + d_cur * (sigma_next - sigma_cur)

    # Apply 2nd order correction
    if i < timesteps - 1:

        x_next = x_next.to("cuda:1")
        sigma_next = sigma_next.to("cuda:1")
        # Denoised image for next step
        denoised = model(x_next, sigma_next)

        # Release GPU memory
        x_next = x_next.cpu()
        sigma_next = sigma_next.cpu()
        denoised = denoised.cpu()

        # Score estimate for next step
        d_next = (x_next - denoised) / sigma_next

        # 2nd order correction by applying trapezoidal rule
        x_next = x_cur + (sigma_next - sigma_cur) * (0.5 * d_cur + 0.5 * d_next)

    # Append to list
    imgs.append(x_next.cpu().detach().numpy())
