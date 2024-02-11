import math
import torch
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from contextlib import nullcontext
from moviepy.editor import ImageSequenceClip
from torch import autocast
from torchvision.transforms import ToTensor, CenterCrop, Resize, Compose, Normalize
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import create_carvekit_interface, load_and_preprocess, instantiate_from_config


def preprocess_image(models, input_im, preprocess):
    '''
    :param input_im (PIL Image).
    :return input_im (H, W, 3) array in [0, 1].
    '''

    if preprocess:
        input_im = load_and_preprocess(models['carvekit'], input_im)
        input_im = (input_im / 255.0).astype(np.float32)
    else:
        input_im = input_im.resize([256, 256], Image.Resampling.LANCZOS)
        input_im = np.asarray(input_im, dtype=np.float32) / 255.0

        alpha = input_im[:, :, 3:4]
        white_im = np.ones_like(input_im)
        input_im = alpha * input_im + (1.0 - alpha) * white_im

        input_im = input_im[:, :, 0:3]

    return input_im


def load_model_from_config(config, ckpt, device, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd:
        print(f'Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_model(
    input_im, model, sampler, precision, h, w, ddim_steps, n_samples, scale, ddim_eta,
    x, y, z):
    precision_scope = autocast if precision == 'autocast' else nullcontext
    with precision_scope('cuda'):
        with model.ema_scope():
            c = model.get_learned_conditioning(input_im).tile(n_samples, 1, 1)
            T = torch.tensor([
                math.radians(x),
                math.sin(math.radians(y)),
                math.cos(math.radians(y)),
                z]).to(c)
            T = T[None, None, :].repeat(n_samples, 1, 1).to(c.device)
            c = torch.cat([c, T], dim=-1)
            c = model.cc_projection(c)
            cond = {}
            cond['c_crossattn'] = [c]
            cond['c_concat'] = [model.encode_first_stage((input_im.to(c.device))).mode().detach()
                                .repeat(n_samples, 1, 1, 1)]
            if scale != 1.0:
                uc = {}
                uc['c_concat'] = [torch.zeros(n_samples, 4, h // 8, w // 8).to(c.device)]
                uc['c_crossattn'] = [torch.zeros_like(c).to(c.device)]
            else:
                uc = None

            shape = [4, h // 8, w // 8]
            samples_ddim, _ = sampler.sample(
                S=ddim_steps, conditioning=cond,
                batch_size=n_samples, shape=shape, verbose=False,
                unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                eta=ddim_eta, x_T=None)
            x_samples_ddim=model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()


def get_rotating_angles(
        n_steps=120,
        n_rot=1,
        elev_low=-math.pi/4,
        elev_high=math.pi/4):
    """Return the elevation and azimus angle of 360 rotation.
    """
    half_steps = n_steps // 2
    rot_steps = n_steps // n_rot
    elevs = np.linspace(elev_low, elev_high, half_steps)
    elevs = np.concatenate([elevs, elevs[::-1]])
    azims = np.linspace(0, 2 * np.pi, rot_steps)
    azims = np.concatenate([azims, azims[::-1]] * n_rot)
    return azims, elevs


def write_video(output_path, frames, fps=24):
    """Convert a list of frames to an MP4 video using MoviePy.
    Args:
        output_path: Path to the output video file.
        frames: List of image frames (PIL images or NumPy arrays).
        fps: Frames per second for the output video.
    """
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_path,
        codec='libx264', fps=fps,
        preset='ultrafast', threads=1)


ddim_steps = 50
n_samples = 1

expr_dir = '../../../expr/zero123'
ckpt = '../../../pretrained/zero123-xl.ckpt'
config = 'configs/sd-objaverse-finetune-c_concat-256.yaml'
device = f'cuda:0'
precision = 'autocast'
SIZE = 256
proc_fn = Compose([
    Resize([SIZE, SIZE]),
    ToTensor(),
    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

config = OmegaConf.load(config)

# Instantiate all models beforehand for efficiency.
models = dict()
models['turncam'] = load_model_from_config(config, ckpt, device=device)
models['carvekit'] = create_carvekit_interface()

sampler = DDIMSampler(models['turncam'])
h, w = 256, 256
scale = 3.0
ddim_eta = 1.0

delta_azims, delta_elevs = get_rotating_angles()
delta_azims = np.rad2deg(delta_azims) # delta azimuth
delta_elevs = np.rad2deg(delta_elevs) # delta elevation
delta_radius = np.zeros_like(delta_azims) # delta radius

for image_name in ['anya_rgba.png', 'face_centered.jpg', 'face_uncentered.png']:
    name = image_name.split('.')[0]
    image_fpath = f'../data/{image_name}'
    raw_im = Image.open(image_fpath)
    input_im = preprocess_image(models, raw_im, preprocess=True)
    input_im = Image.fromarray((input_im * 255.0).astype(np.uint8))
    input_im.save(f'{expr_dir}/{name}_input.jpg')
    input_im = proc_fn(input_im)[None].to(device)
    frames = []
    for x, y, z in zip(delta_elevs, delta_azims, delta_radius):
        x_samples_ddim = sample_model(
            input_im, models['turncam'], sampler, precision, h, w,
            ddim_steps, n_samples, scale, ddim_eta,
            x, y, z)
        image = 255.0 * x_samples_ddim[0].cpu().numpy()
        frames.append(rearrange(image, 'c h w -> h w c').astype('uint8'))

    write_video(f'{expr_dir}/{name}_render_rotate.mp4', frames, fps=24)