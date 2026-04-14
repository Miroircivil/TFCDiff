import numpy as np
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from functools import partial
from tqdm import tqdm
from inspect import isfunction
import torch_dct as dct


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class DDPM(nn.Module):
    def __init__(self, base_model, config, device, conditional=True):
        super().__init__()
        self.device = device
        self.model = base_model
        self.config = config
        self.device = device
        self.conditional = conditional
        
        self.loss_func = nn.L1Loss(reduction='sum').to(device)
        
        config_diff = config["diffusion"]
        
        self.num_steps = config_diff["num_steps"]
        
        self.set_new_noise_schedule(config_diff, device)
    
    # # We've tried the hybrid loss, yet it degraded the performance
    # def charbonnier_loss(self, x, eps=1e-4):
    #     return torch.sqrt(x ** 2 + eps ** 2)
        
    # def hybrid_loss(self, noise, x_recon, alpha=1, eps=1e-4):
    #     err = noise - x_recon
    #     freq_loss = self.charbonnier_loss(err, eps).sum()

    #     # idct
    #     err_temp = dct.idct(F.pad(err, (0, 2600), mode='constant', value=0), norm='ortho')

    #     # Loss in temporal domain
    #     temp_loss = self.charbonnier_loss(err_temp, eps).sum()

    #     total_loss = freq_loss + alpha * temp_loss
    #     return total_loss
    
    # ultilized by WaveGrad for rapid generation
    def generate_fibonacci_with_averages(self, n):
        fibonacci = [1, 1]
        while len(fibonacci) < n:
            fibonacci.append(fibonacci[-1] + fibonacci[-2])
        result = []
        for i in range(len(fibonacci) - 1):
            result.append(fibonacci[i])
            average = (fibonacci[i] + fibonacci[i + 1]) / 2
            result.append(average)
        result.append(fibonacci[-1])
        return torch.tensor(result) 
    
    def make_beta_schedule(self, schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        elif schedule == "quartic":
            betas = torch.linspace(start ** 0.25, end ** 0.25, n_timesteps) ** 4
        elif schedule == "fibonacci":
            n_timesteps = 50
            start = 4e-6
            betas = self.generate_fibonacci_with_averages(26)
            betas = betas * start
        return betas
    
    
    def set_new_noise_schedule(self, config_diff, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        
        betas = self.make_beta_schedule(schedule=config_diff["schedule"], n_timesteps=config_diff["num_steps"],
                                            start=config_diff["beta_start"], end=config_diff["beta_end"])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        SNR = alphas_cumprod / (1 - alphas_cumprod)
        # SNR scaling factor
        factor = 150
        alphas_cumprod = (factor * SNR) / ((factor * SNR) + 1)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))
        alphas = np.zeros_like(alphas_cumprod)
        # iteratively solve alpha_t and beta_t
        alphas[0] = alphas_cumprod[0]  
        alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]  
        betas = 1 - alphas
        
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        
        
    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped
    
    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.model(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.model(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=False, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_steps//10))
        if not self.conditional:
            shape = x_in
            cur_x = torch.randn(shape, device=device)
            ret_x = cur_x
            for i in reversed(range(0, self.num_steps)):
                cur_x = self.p_sample(cur_x, i)
                if i % sample_inter == 0:
                    ret_x = torch.cat([ret_x, cur_x], dim=0)
        else:
            # Classifier-free Guidance for future work, we set guidance = 1 to disable it
            zero_condition = torch.zeros_like(x_in).to(device)
            guidance_scale = 1

            x = x_in
            shape = x.shape
            cur_x = torch.randn(shape, device=device)
            ret_x = [cur_x]

            for i in reversed(range(0, self.num_steps)):
                # obtain conditional inference
                model_mean_cond, model_log_variance_cond = self.p_mean_variance(
                    x=cur_x, t=i, clip_denoised=False, condition_x=x
                )

                # obtain unconditional inference
                model_mean_uncond, model_log_variance_uncond = self.p_mean_variance(
                    x=cur_x, t=i, clip_denoised=False, condition_x=zero_condition
                )

                # Classifier-Free Guidance: when guidance = 1, it degrades into normal conditional diffusion
                model_mean = model_mean_uncond + guidance_scale * (model_mean_cond - model_mean_uncond)
                model_log_variance = model_log_variance_cond  

                # next timestep
                noise = torch.randn_like(cur_x) if i > 0 else torch.zeros_like(cur_x)
                cur_x = model_mean + noise * (0.5 * model_log_variance).exp()

                if i % sample_inter == 0:
                    ret_x.append(cur_x)

        if continous:
            return ret_x
        else:
            return ret_x[-1]  
    
    @torch.no_grad()
    def sample(self, batch_size=1, shape=[1, 1000], continous=False):
        return self.p_sample_loop((batch_size, shape[0], shape[1]), continous)
    
    @torch.no_grad()
    def denoising(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)
    
    def q_sample_loop(self, x_start, continous=False):
        sample_inter = (1 | (self.num_steps//10))
        ret_x = [x_start]
        cur_x = x_start
        for t in range(1, self.num_steps+1):
            B,C,L = cur_x.shape
            continuous_sqrt_alpha_cumprod = torch.FloatTensor(
                np.random.uniform(
                    self.sqrt_alphas_cumprod_prev[t-1],
                    self.sqrt_alphas_cumprod_prev[t],
                    size=B
                )
            ).to(cur_x.device)
            continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
                B, -1)

            noise = torch.randn_like(cur_x)
            cur_x = self.q_sample(
                x_start=cur_x, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1), noise=noise)
            if t % sample_inter == 0:
                ret_x.append(cur_x)
        if continous:
            return ret_x
        else:
            return ret_x[-1]
    
    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )
    
    def p_losses(self, x_in, y_in, noise=None, epoch_no=None):
        # x_in: clean signal
        # y_in: noisy signal as condition
        eta = 3
        x_start = dct.dct(x_in, norm='ortho') / eta
        y_in = dct.dct(y_in, norm='ortho') / eta
        x_start = x_start[:,: ,:1000]
        y_in = y_in[:,: ,:1000]
        B, C, L = x_start.shape

        # # this is the zero mask for CFG
        # zero_y = torch.zeros_like(y_in).to(y_in.device)
        # drop_mask = torch.rand(B, device=x_start.device) < 0
        # y_in = torch.where(drop_mask.view(B, 1, 1), zero_y, y_in)

        # hierarchical uniform sampling
        t = np.random.randint(1, self.num_steps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=B
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(B, -1)

        # diffusion forward process
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, 
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1), 
            noise=noise
        )

        if not self.conditional:
            x_recon = self.model(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.model(torch.cat([y_in, x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        loss = self.loss_func(noise, x_recon)
        return loss
    
    def forward(self, x, y, *args, **kwargs):
        return self.p_losses(x, y, *args, **kwargs)
    
    