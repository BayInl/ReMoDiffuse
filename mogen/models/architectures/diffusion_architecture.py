import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_architecture import BaseArchitecture
from ..builder import (
    ARCHITECTURES,
    build_architecture,
    build_submodule,
    build_loss
)
from ..utils.gaussian_diffusion import (
    GaussianDiffusion, get_named_beta_schedule, create_named_schedule_sampler,
    ModelMeanType, ModelVarType, LossType, space_timesteps, SpacedDiffusion
)
from diffusers import DDPMScheduler


def build_diffusion(cfg):
    beta_scheduler = cfg['beta_scheduler']
    diffusion_steps = cfg['diffusion_steps']

    betas = get_named_beta_schedule(beta_scheduler, diffusion_steps)
    model_mean_type = {
        'start_x': ModelMeanType.START_X,
        'previous_x': ModelMeanType.PREVIOUS_X,
        'epsilon': ModelMeanType.EPSILON
    }[cfg['model_mean_type']]
    model_var_type = {
        'learned': ModelVarType.LEARNED,
        'fixed_small': ModelVarType.FIXED_SMALL,
        'fixed_large': ModelVarType.FIXED_LARGE,
        'learned_range': ModelVarType.LEARNED_RANGE
    }[cfg['model_var_type']]
    if cfg.get('respace', None) is not None:
        diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(diffusion_steps, cfg['respace']),
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=LossType.MSE
        )
    else:
        diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=LossType.MSE)
    return diffusion


@ARCHITECTURES.register_module()
class MotionDiffusion(BaseArchitecture):

    def __init__(self,
                 model=None,
                 loss_recon=None,
                 diffusion_train=None,
                 diffusion_test=None,
                 init_cfg=None,
                 inference_type='ddpm',
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.model = build_submodule(model)
        self.loss_recon = build_loss(loss_recon)
        self.diffusion_train = build_diffusion(diffusion_train)
        self.diffusion_test = build_diffusion(diffusion_test)
        self.diffusion_train_cfg = diffusion_train
        self.diffusion_test_cfg = diffusion_test
        self.sampler = create_named_schedule_sampler(
            'uniform', self.diffusion_train)
        self.inference_type = inference_type
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=diffusion_train['diffusion_steps'],
                                             beta_schedule=diffusion_train['beta_scheduler'],
                                             variance_type=diffusion_train['model_var_type'],
                                             prediction_type='sample' if diffusion_train[
                                                 'model_mean_type'] == 'start_x' else 'epsilon',
                                             clip_sample=False)

    def forward(self, **kwargs):
        motion, motion_mask = kwargs['motion'].float(
        ), kwargs['motion_mask'].float()
        sample_idx = kwargs.get('sample_idx', None)
        clip_feat = kwargs.get('clip_feat', None)
        B, T = motion.shape[:2]
        shape = (B, T, kwargs['motion'].shape[-1])
        text = []
        for i in range(B):
            text.append(kwargs['motion_metas'][i]['text'])

        if self.training:
            t, _ = self.sampler.sample(B, motion.device)
            real_noise = torch.randn_like(motion)
            x_t = self.noise_scheduler.add_noise(motion, real_noise, t)
            output = self.diffusion_train.training_losses(
                model=self.model,
                x_start=motion,
                t=t,
                model_kwargs={
                    'motion_mask': motion_mask,
                    'motion_length': kwargs['motion_length'],
                    'text': text,
                    'clip_feat': clip_feat,
                    'sample_idx': sample_idx},
                x_t=x_t
            )
            pred, target = output['pred'], output['target']
            recon_loss = self.loss_recon(
                pred, target, reduction_override='none')
            recon_loss = (recon_loss.mean(dim=-1) *
                          motion_mask).sum() / motion_mask.sum()
            loss = {'recon_loss': recon_loss}
            return loss
        else:
            model_kwargs = self.model.get_precompute_condition(
                device=motion.device, text=text, **kwargs)
            model_kwargs['motion_mask'] = motion_mask
            model_kwargs['sample_idx'] = sample_idx
            inference_kwargs = kwargs.get('inference_kwargs', {})

            def sample(scheduler, model, motion, shape, inference_steps):
                scheduler.set_timesteps(inference_steps, device=motion.device)
                timesteps = [torch.tensor(
                    [t] * shape[0], device=motion.device).long() for t in scheduler.timesteps]
                sample = torch.randn(
                    shape, device=motion.device, dtype=motion.dtype)
                for i, t in enumerate(timesteps):
                    with torch.no_grad():
                        model_output = model(sample, t, **model_kwargs)
                    sample = scheduler.step(
                        model_output, t[0], sample).prev_sample
                return sample
            if self.inference_type == 'ddpm':
                output = self.diffusion_test.p_sample_loop(
                    self.model,
                    shape,
                    clip_denoised=False,
                    progress=False,
                    model_kwargs=model_kwargs,
                    **inference_kwargs
                )
            elif self.inference_type == 'ddim':
                output = self.diffusion_test.ddim_sample_loop(
                    self.model,
                    shape,
                    clip_denoised=False,
                    progress=False,
                    model_kwargs=model_kwargs,
                    eta=0,
                    **inference_kwargs
                )
            elif self.inference_type == 'sde-dpmsolver++_2order':
                from diffusers import DPMSolverMultistepScheduler
                dpm_scheduler = DPMSolverMultistepScheduler(
                    num_train_timesteps=1000,
                    beta_schedule='linear',
                    variance_type='fixed_large',
                    prediction_type='sample',
                    algorithm_type='sde-dpmsolver++',
                    use_karras_sigmas=True,
                    solver_order=2,
                )
                output = sample(dpm_scheduler, self.model, motion=motion, shape=shape,
                                inference_steps=self.diffusion_test_cfg['inference_steps'])
            elif self.inference_type == 'dpmsolver++_2order':
                from diffusers import DPMSolverMultistepScheduler
                dpm_scheduler = DPMSolverMultistepScheduler(
                    num_train_timesteps=1000,
                    beta_schedule='linear',
                    variance_type='fixed_large',
                    prediction_type='sample',
                    algorithm_type='dpmsolver++',
                    use_karras_sigmas=True,
                    solver_order=2,
                )
                output = sample(dpm_scheduler, self.model, motion=motion, shape=shape,
                                inference_steps=self.diffusion_test_cfg['inference_steps'])
            elif self.inference_type == 'sde-dpmsolver++_3order':
                from diffusers import DPMSolverMultistepScheduler
                dpm_scheduler = DPMSolverMultistepScheduler(
                    num_train_timesteps=1000,
                    beta_schedule='linear',
                    variance_type='fixed_large',
                    prediction_type='sample',
                    algorithm_type='sde-dpmsolver++',
                    use_karras_sigmas=True,
                    solver_order=3,
                )
                output = sample(dpm_scheduler, self.model, motion=motion, shape=shape,
                                inference_steps=self.diffusion_test_cfg['inference_steps'])
            elif self.inference_type == 'dpmsolver++_3order':
                from diffusers import DPMSolverMultistepScheduler
                dpm_scheduler = DPMSolverMultistepScheduler(
                    num_train_timesteps=1000,
                    beta_schedule='linear',
                    variance_type='fixed_large',
                    prediction_type='sample',
                    algorithm_type='dpmsolver++',
                    use_karras_sigmas=True,
                    solver_order=3,
                )
                output = sample(dpm_scheduler, self.model, motion=motion, shape=shape,
                                inference_steps=self.diffusion_test_cfg['inference_steps'])
            else:
                pass

            if getattr(self.model, "post_process") is not None:
                output = self.model.post_process(output)
            results = kwargs
            results['pred_motion'] = output
            results = self.split_results(results)
            return results

