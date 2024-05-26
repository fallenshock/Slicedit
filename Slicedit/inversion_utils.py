import torch
import os
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from Slicedit.video_utils import image_grid
import torch.nn as nn
from torch.cuda.amp import autocast 
import numpy as np
from Slicedit.slicedit_attention_utils import register_n_keyframes, register_extended_attention, register_conv_injection, register_time, register_denoiser_xy
from Slicedit.video_utils import create_video_from_frames, save_video
from PIL import Image
from math import ceil as ceil
import torch.nn.functional as F
from einops import rearrange


def sample_xts_from_x0_and_eps(model, x0, num_inference_steps=50, eps = None):
	"""
	Samples from P(x_1:T|x_0)
	"""
	# torch.manual_seed(43256465436)

	alpha_bar = model.scheduler.alphas_cumprod
	sqrt_one_minus_alpha_bar = (1-alpha_bar) ** 0.5
	variance_noise_shape = (
			num_inference_steps,
			model.unet.in_channels, 
			model.unet.sample_size,
			model.unet.sample_size)
	
	timesteps = model.scheduler.timesteps.to(model.device)
	t_to_idx = {int(v):k for k,v in enumerate(timesteps)}
 
	xts = torch.zeros(variance_noise_shape).to(x0.device)
	for t in reversed(timesteps):
		idx = t_to_idx[int(t)]
		if eps is None:
			xts[idx] = x0 * (alpha_bar[t] ** 0.5) + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
		else:
			xts[idx] = x0 * (alpha_bar[t] ** 0.5) + eps[idx] * sqrt_one_minus_alpha_bar[t]
	xts = torch.cat([xts, x0 ],dim = 0)

	
	return xts


def encode_text(model, prompts):
	text_input = model.tokenizer(
		prompts,
		padding="max_length",
		max_length=model.tokenizer.model_max_length, 
		truncation=True,
		return_tensors="pt",
	)
	with torch.no_grad():
		text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
	return text_encoding


def get_variance(model, timestep):
	prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
	alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
	alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
	beta_prod_t = 1 - alpha_prod_t
	beta_prod_t_prev = 1 - alpha_prod_t_prev
	variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
	return variance


def calc_mu(model, model_output, timestep, sample, eta = 0):
	# get previous step value (=t-1)
	prev_timestep = timestep - model.scheduler.config.num_train_timesteps // model.scheduler.num_inference_steps
	# compute alphas, betas
	alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
	alpha_prod_t_prev = model.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else model.scheduler.final_alpha_cumprod
	beta_prod_t = 1 - alpha_prod_t
	# compute predicted original sample from predicted noise also called
	# "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
	pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
	# compute variance: "sigma_t(η)" -> see formula (16)
	# σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)    
	# variance = self.scheduler._get_variance(timestep, prev_timestep)
	variance = get_variance(model, timestep) #, prev_timestep)
	# std_dev_t = eta * variance ** (0.5)
	# Take care of asymetric reverse process (asyrp)
	model_output_direction = model_output
	# compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
	# pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction

	pred_sample_direction = (1 - alpha_prod_t_prev - eta*variance) ** (0.5) * model_output_direction
	# compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
	mu = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
	# add noise if eta > 0

	return mu,variance


def slicedit_ddpm_inversion_loop(model, x0_video,
							etas = None, 
							prog_bar = False,
							src_prompt = "",
							tar_prompt = "",
							prompt_time = "",
							cfg_scale = 3.5,
							alpha=0.5,
							num_inference_steps=50,
							eps = None, 
							n_frames = 64, 
							skip=0, 
							cfg_scale_tar=1, 
							save_path="", 
							orig_frame_size=(512,512), 
							qk_injection_t=-1, 
							conv_injection_t=-1, 
							negative_time_prompt="", 
							cfg_scale_time=1,
							frame_batch_size=6,
							use_negative_tar_prompt=False,
							x_and_y=False):
	
	# xts for source and target frames for all timesteps shape:[T_diff, src/tar, n_frames, 4, 64, 64]
	xts = torch.zeros((num_inference_steps+1,2,n_frames,4,64,64),device=model.device)
	# noisy volume for source and target frames for current timestep shape:[src/tar, n_frames, 4, 64, 64]
	volume_t = torch.zeros(2,n_frames, 4, 64, 64).to(model.device)
	# noise predictions for source and target frames for current timestep shape:[src/tar, n_frames, 4, 64, 64]
	noise_pred_xy = torch.zeros(2, n_frames, 4, 64, 64).to(model.device)

	# embed source and target text prompts
	if not src_prompt=="":
		text_embeddings = encode_text(model, src_prompt)
		text_embeddings_time = encode_text(model, prompt_time)
	if not tar_prompt=="":
		tar_emb = encode_text(model, tar_prompt)

	## negative target prompt
	if use_negative_tar_prompt:
		negative_tar_prompt = "ugly, blurry, low res"
	else:
		negative_tar_prompt = ""
	neg_target_embedding = encode_text(model, negative_tar_prompt)

	
	negative_time_embedding = encode_text(model, negative_time_prompt)
	uncond_embedding = encode_text(model, "")

	# get timesteps
	timesteps = model.scheduler.timesteps.to(model.device)

	if type(etas) in [int, float]: etas = [etas]*(model.scheduler.num_inference_steps) 
	## we can specify a value of epsilon in order to be able to use a 
	## fixed epsilon for all the frames of a video
	for i in range(n_frames):
		xts[:,0,i,:,:,:] = sample_xts_from_x0_and_eps(model, x0_video[i], num_inference_steps=num_inference_steps, eps = eps)
		# initialize edit xts with source xts
		xts[:,1,i,:,:,:] = xts[:,0,i,:,:,:]


	if qk_injection_t == -1:
		qk_injection_t = num_inference_steps-skip
	else:
		qk_injection_t = (num_inference_steps-skip) * qk_injection_t // 100
	if conv_injection_t == -1:
		conv_injection_t = num_inference_steps-skip
	else:
		conv_injection_t = (num_inference_steps-skip) * conv_injection_t // 100
	qk_injection_timesteps = timesteps[skip:skip+qk_injection_t] if qk_injection_t >= 0 else []
	conv_injection_timesteps = timesteps[skip:skip+conv_injection_t] if conv_injection_t >= 0 else []

	register_extended_attention(model, qk_injection_timesteps)
	register_conv_injection(model, conv_injection_timesteps)


	n_keyframes = 3
	register_n_keyframes(model, n_keyframes)

	t_to_idx = {int(v):k for k,v in enumerate(timesteps[-(num_inference_steps-skip):])}
	op = tqdm(timesteps[-(num_inference_steps-skip):]) if prog_bar else timesteps[-(num_inference_steps-skip):]

	# initialize volume_t with source xts at timestep T-skip
	volume_t[1, :, :,:,:] = xts[skip,0,:,:,:,:] # .permute(1,2,0)

	for k,t in enumerate(op):

		idx = t_to_idx[int(t)]

		# noise_pred_xy holds both noise predictions for source and target
		noise_pred_xy = pred_noise_xy_func(xts, t, idx, uncond_embedding, neg_target_embedding, text_embeddings, model, cfg_scale, frame_batch_size, tar_emb, cfg_scale_tar, skip)
		volume_t[0, :, :,:,:] = xts[skip+idx,0,:,:,:,:] # .permute(1,2,0)
		# spatiotemporal denoise
		if alpha > 0:
			st_batch_size = 8 # reduce if having memory issues
			noise_pred_t = temporal_volume_denoise(model, n_frames, volume_t, t, negative_time_embedding, text_embeddings_time, st_batch_size, cfg_scale_time, x_and_y=x_and_y)
			# combine noise predictions
			noise_pred_src = (np.sqrt(alpha))*noise_pred_t[0] + (np.sqrt(1-alpha))*noise_pred_xy[0]
			noise_pred_tar = (np.sqrt(alpha))*noise_pred_t[1] + (np.sqrt(1-alpha))*noise_pred_xy[1]                   
		else:
			noise_pred_src = noise_pred_xy[0]
			noise_pred_tar = noise_pred_xy[1]
		
		## Edit Friendly DDPM inversion step https://arxiv.org/abs/2304.06140
		xtm1_src = xts[skip+idx+1,0,:,:,:,:]
		xt_src = xts[skip+idx,0,:,:,:,:]

		mu_xt_src,variance = calc_mu(model, noise_pred_src, t, xt_src,eta=etas[idx])
		z_src = (xtm1_src - mu_xt_src ) / ( etas[idx] * variance ** 0.5 )

		if k==num_inference_steps-1:
			z_src = torch.zeros_like(z_src)

		mu_xt_tar,variance = calc_mu(model, noise_pred_tar, t,volume_t[1],eta=etas[idx])
		sigma_z =  etas[idx] * variance ** (0.5) * z_src
		xt = mu_xt_tar + sigma_z
		volume_t[1] = xt
		xts[skip+k+1,1,:,:,:,:] = xt

		# error accumulation fix (as proposed in Edit Friendly DDPM Inversion)
		xtm1 = mu_xt_src + ( etas[idx] * variance ** 0.5 ) * z_src
		xts[skip+idx+1, 0, :, :,:,:] = xtm1 

	## end of diffusion loop.

	decoded_frames = []
	for n in range(n_frames):
		w0 = volume_t[1,n,:,:,:].expand(1, -1, -1, -1)
		# vae decode image
		with torch.autocast("cuda"), torch.inference_mode():
			x0_dec = model.vae.decode(1 / 0.18215 * w0).sample
		decoded_frames.append(x0_dec)
		img = image_grid(x0_dec)

		# resize images to original size and save them to disk as png files
		img = img.resize((orig_frame_size[1], orig_frame_size[0]), resample=Image.LANCZOS)
		image_name_png = f'frames/cfg_d_{[cfg_scale_tar][0]}_' + f'skip_{skip}_' + str(n) + ".png"

		os.makedirs(save_path+'/frames' , exist_ok=True)
		save_full_path = os.path.join(save_path, image_name_png)
		img.save(save_full_path)

	create_video_from_frames(save_path+'/frames')
	## Uncomment the following lines to generate a compressed video without ffmpeg
	decoded_frames = torch.stack(decoded_frames)
	save_video(raw_frames=decoded_frames, save_path=os.path.join(save_path, 'slicedit_video_compressed.mp4'), fps=25, orig_size=orig_frame_size[:-1])


def pred_noise_xy_func(xts, 
					   t, 
					   idx, 
					   uncond_embedding,
					   neg_target_embedding, 
					   text_embeddings, 
					   model, 
					   cfg_scale, 
					   frame_batch_size, 
					   tar_emb=None, 
					   cfg_scale_tar=1, 
					   skip=0,
					   n_keyframes=3):

	# process both xt src and xt tar
	noise_pred_xy = torch.zeros(xts.shape[1], xts.shape[2], 4, 64, 64).to(model.device)
	n_frames = xts.shape[2]

	for n in range(ceil(n_frames/frame_batch_size)):
		batch_upper_idx = min((n+1)*frame_batch_size, n_frames)
		cur_batch_size = batch_upper_idx - n*frame_batch_size
		if n_keyframes == 3:
			# global frames
			idx1 = n_frames*3//6

			# add local frame indices
			if cur_batch_size == frame_batch_size:
				ref_frame_indices = [idx1, n*frame_batch_size + 1, n*frame_batch_size + 4]
			# last batch in video smaller than frame_batch_size
			else:
				ref_frame_indices = [idx1, n*frame_batch_size, n*frame_batch_size]
		else:
			raise NotImplementedError

		# xts contain both original and edited frames
		xt_src = xts[skip+idx,0, n*frame_batch_size:batch_upper_idx,:,:,:]
		xt_tar = xts[skip+idx,1, n*frame_batch_size:batch_upper_idx,:,:,:]
		
		# compute text embeddings
		text_embed_input = torch.cat([uncond_embedding.repeat(cur_batch_size+n_keyframes, 1, 1),
								text_embeddings.repeat(cur_batch_size+n_keyframes, 1, 1),
								neg_target_embedding.repeat(cur_batch_size+n_keyframes, 1, 1),
								tar_emb.repeat(cur_batch_size+n_keyframes, 1, 1)])
		
		with torch.no_grad():
			ref_frames_src = torch.cat([xts[skip+idx,0, ref_frame_indices[i],:,:,:].unsqueeze(0) for i in range(n_keyframes)], dim=0)
			ref_frames_tar = torch.cat([xts[skip+idx,1, ref_frame_indices[i],:,:,:].unsqueeze(0) for i in range(n_keyframes)], dim=0)
			
			# register time for the injection schedule
			register_time(model, t.item())
			register_denoiser_xy(model, is_xy=True)
			# feed denoiser with a batch of current and reference frames for EA processing x4 (src_unc, src_cond, tar_unc, tar_cond)
			noise_pred = model.unet.forward(torch.cat(([xt_src]+[ref_frames_src]) * 2 + ([xt_tar]+[ref_frames_tar]) * 2), timestep =  t, encoder_hidden_states = text_embed_input)
			# set denoiser to normal mode
			register_denoiser_xy(model, is_xy=False)

			# perform guidance
			noise_pred_src_uncond, noise_pred_src_cond, noise_pred_tar_uncond, noise_pred_tar_cond = noise_pred['sample'].chunk(4)
			noise_pred_src = noise_pred_src_uncond + cfg_scale * (noise_pred_src_cond - noise_pred_src_uncond)
			noise_pred_tar = noise_pred_tar_uncond + cfg_scale_tar * (noise_pred_tar_cond - noise_pred_tar_uncond)
		
		# store source and target noise predictions
		noise_pred_xy[0, n*frame_batch_size:batch_upper_idx, :,:,:] = noise_pred_src[:cur_batch_size]
		noise_pred_xy[1, n*frame_batch_size:batch_upper_idx, :,:,:] = noise_pred_tar[:cur_batch_size]

	return noise_pred_xy
	

def temporal_volume_denoise(model, n_frames, volume_t, t, negative_time_embedding, text_embeddings_time, st_batch_size, cfg_scale_time, x_and_y=False):
	noise_pred_t = torch.zeros(2, n_frames, 4, 64, 64).to(model.device)
	# calculate number of size 64 temporal windows in the video s.t. there is atleast 1 overlapping frame between them
	n_slices = ceil((n_frames-1) / 63)
	# calculate the overlaps between windows
	if n_slices > 1:
		# overlap = (n_slices*64-n_frames)//n_slices #  
		overlap = (n_slices*64-n_frames)//(n_slices-1)
		# overlap_last = overlap+(n_slices*64-n_frames)%n_slices #
		overlap_last = overlap+(n_slices*64-n_frames)%(n_slices-1)
		overlap_list = [overlap]*(n_slices-2) + [overlap_last]
		# calculate the start indices of each window
		cumsum_overlap = np.cumsum(overlap_list)
		# sl_idxs = [0]+[i*64-i*overlap_list[i] for i in range(0, n_slices-1)] #
		sl_idxs = [0]+[(i+1)*64-cumsum_overlap[i] for i in range(0, n_slices-1)]

	else:
		overlap = 0
		overlap_last = 0
		overlap_list = [0]
		sl_idxs = [0]

	for idx, sl_i in enumerate(sl_idxs):
		for i in range(64//st_batch_size): 
			
			xt = rearrange(volume_t[0,sl_i:sl_i+64, :, :, i*st_batch_size:(i+1)*st_batch_size], 'n c x y -> y c n x')
			noise_pred = temporal_denoise(model, xt, t, negative_time_embedding, text_embeddings_time, st_batch_size, cfg_scale_time)
			noise_pred_t[0,sl_i:sl_i+64,:,:,i*st_batch_size:(i+1)*st_batch_size] += rearrange(noise_pred, 'y c n x -> n c x y')
			
			xt = rearrange(volume_t[1,sl_i:sl_i+64, :, :, i*st_batch_size:(i+1)*st_batch_size], 'n c x y -> y c n x')
			noise_pred = temporal_denoise(model, xt, t, negative_time_embedding, text_embeddings_time, st_batch_size, cfg_scale_time)
			noise_pred_t[1,sl_i:sl_i+64,:,:,i*st_batch_size:(i+1)*st_batch_size] += rearrange(noise_pred, 'y c n x -> n c x y')

		# normalize noise levels in overlapping frames
		if sl_i > 0:
			noise_pred_t[:,sl_i:sl_i+overlap_list[idx-1],:,:,:] = (noise_pred_t[:,sl_i:sl_i+overlap_list[idx-1],:,:,:])*np.sqrt(1/2)
		
	if x_and_y: # add the x_t_slice noise prediction to the existing x_y_slice noise prediction
		volume_t_p = rearrange(volume_t, 's n c x y -> s n c y x').permute(0,1,3,2,4)
		noise_pred_t_x = torch.zeros_like(volume_t_p)

		for idx, sl_i in enumerate(sl_idxs):
			for i in range(64//st_batch_size):
				xt = rearrange(volume_t_p[0,sl_i:sl_i+64, :, :, i*st_batch_size:(i+1)*st_batch_size], 'n c x y -> y c n x')
				noise_pred = temporal_denoise(model, xt, t, negative_time_embedding, text_embeddings_time, st_batch_size, cfg_scale_time)
				noise_pred_t_x[0,sl_i:sl_i+64,:,:,i*st_batch_size:(i+1)*st_batch_size] += rearrange(noise_pred, 'y c n x -> n c x y')

				
				xt = rearrange(volume_t_p[1,sl_i:sl_i+64, :, :, i*st_batch_size:(i+1)*st_batch_size], 'n c x y -> y c n x')
				noise_pred = temporal_denoise(model, xt, t, negative_time_embedding, text_embeddings_time, st_batch_size, cfg_scale_time)
				noise_pred_t_x[1,sl_i:sl_i+64,:,:,i*st_batch_size:(i+1)*st_batch_size] += rearrange(noise_pred, 'y c n x -> n c x y')
			# normalize noise levels in overlapping frames
			if sl_i > 0:
				noise_pred_t_x[:,sl_i:sl_i+overlap_list[idx-1],:,:,:] = (noise_pred_t_x[:,sl_i:sl_i+overlap_list[idx-1],:,:,:])*np.sqrt(1/2)

		noise_pred_t_x = rearrange(noise_pred_t_x, 's n c y x -> s n c x y')

		noise_pred_t = noise_pred_t + noise_pred_t_x # add the x_t_slice noise prediction to the existing x_y_slice noise prediction
		noise_pred_t = noise_pred_t * np.sqrt(1/2) # normalize noise levels between x and y slices
		
	return noise_pred_t
	

def temporal_denoise(model, xt, t, neg_embedding, text_embeddings_time, st_batch_size, cfg_scale):

	if cfg_scale == 1:
		with torch.no_grad():
			cond_out = model.unet.forward(xt, timestep =  t, encoder_hidden_states = text_embeddings_time.repeat(st_batch_size, 1, 1))
			return cond_out.sample

	## Unconditional embedding
	with torch.no_grad():
		uncond_out = model.unet.forward(xt, timestep =  t, encoder_hidden_states = neg_embedding.repeat(st_batch_size, 1, 1))

	## Conditional embedding  
	with torch.no_grad():
		cond_out = model.unet.forward(xt, timestep =  t, encoder_hidden_states = text_embeddings_time.repeat(st_batch_size, 1, 1))
	## classifier free guidance
	noise_pred = uncond_out.sample + cfg_scale * (cond_out.sample - uncond_out.sample)
	
	return noise_pred
