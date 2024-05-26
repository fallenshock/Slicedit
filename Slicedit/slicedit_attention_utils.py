## This file contains functions which were based upon the code from TokenFlow: https://github.com/omerbt/TokenFlow ##

from typing import Type
import torch

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True

    return False

# This function puts the denoiser into Extended Attention mode (for x_y slice denoising)
def register_denoiser_xy(diffusion_model, is_xy):
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = diffusion_model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'is_xy', is_xy)

            module = diffusion_model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'is_xy', is_xy)

    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = diffusion_model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'is_xy', is_xy)

            module = diffusion_model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'is_xy', is_xy)

    module = diffusion_model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 'is_xy', is_xy)

    module = diffusion_model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 'is_xy', is_xy)


def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)

    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)

            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)

    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)

            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)

    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)

    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 't', t)


def register_n_keyframes(model, n_keyframes):   

    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'n_keyframes', n_keyframes)

            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'n_keyframes', n_keyframes)

    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 'n_keyframes', n_keyframes)

            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 'n_keyframes', n_keyframes)

    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 'n_keyframes', n_keyframes)

    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 'n_keyframes', n_keyframes)


def prepare_extended_QKV(self, token_mat, n_keyframes, n_orig_frames, sequence_length, h, dim, is_SA=False):
    if not is_SA:
        token_mat = token_mat.reshape(1, n_keyframes * sequence_length, -1).repeat(n_orig_frames, 1, 1)

    token_mat = self.head_to_batch_dim(token_mat)
    token_mat = token_mat.view(n_orig_frames, h, sequence_length * n_keyframes, dim // h)
    return token_mat
    

def register_extended_attention(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            if self.is_xy == True:
                n_frames = batch_size // 4
                is_cross = encoder_hidden_states is not None

                encoder_hidden_states = encoder_hidden_states if is_cross else x
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                v = self.to_v(encoder_hidden_states)

                if self.injection_schedule is not None and (self.t in self.injection_schedule):
                    # inject unconditional
                    q[2*n_frames:3 * n_frames] = q[:n_frames]
                    k[2*n_frames:3 * n_frames] = k[:n_frames]
                    # inject conditional
                    q[3 * n_frames:] = q[n_frames:2*n_frames]
                    k[3 * n_frames:] = k[n_frames:2*n_frames]

                n_keyframes = self.n_keyframes # 1g2l
                idx_keyframe = n_frames-n_keyframes
                n_orig_frames = n_frames-n_keyframes                

                # keyframe indices
                fr_idx_low_src_unc = idx_keyframe
                fr_idx_high_src_unc = idx_keyframe+n_keyframes
                fr_idx_low_src_cnd = n_frames+idx_keyframe
                fr_idx_high_src_cnd = n_frames+idx_keyframe+n_keyframes
                fr_idx_low_unc = 2*n_frames+idx_keyframe
                fr_idx_high_unc =2*n_frames+idx_keyframe+n_keyframes
                fr_idx_low_cnd = 3*n_frames+idx_keyframe
                fr_idx_high_cnd = 3*n_frames+idx_keyframe+n_keyframes

                # K
                k_src_unc = prepare_extended_QKV(self, k[fr_idx_low_src_unc:fr_idx_high_src_unc], n_keyframes, n_orig_frames, sequence_length, h, dim)
                k_src_cnd = prepare_extended_QKV(self, k[fr_idx_low_src_cnd:fr_idx_high_src_cnd], n_keyframes, n_orig_frames, sequence_length, h, dim)
                k_uncond = prepare_extended_QKV(self, k[fr_idx_low_unc:fr_idx_high_unc], n_keyframes, n_orig_frames, sequence_length, h, dim)
                k_cond = prepare_extended_QKV(self, k[fr_idx_low_cnd:fr_idx_high_cnd], n_keyframes, n_orig_frames, sequence_length, h, dim)
                # V
                v_src_unc = prepare_extended_QKV(self, v[fr_idx_low_src_unc:fr_idx_high_src_unc], n_keyframes, n_orig_frames, sequence_length, h, dim)
                v_src_cnd = prepare_extended_QKV(self, v[fr_idx_low_src_cnd:fr_idx_high_src_cnd], n_keyframes, n_orig_frames, sequence_length, h, dim)
                v_uncond = prepare_extended_QKV(self, v[fr_idx_low_unc:fr_idx_high_unc], n_keyframes, n_orig_frames, sequence_length, h, dim)
                v_cond = prepare_extended_QKV(self, v[fr_idx_low_cnd:fr_idx_high_cnd], n_keyframes, n_orig_frames, sequence_length, h, dim)

                # Q
                q_src_unc = prepare_extended_QKV(self, q[:n_orig_frames], 1, n_orig_frames, sequence_length, h, dim, is_SA=True)
                q_src_cnd = prepare_extended_QKV(self, q[n_frames: n_frames+n_orig_frames], 1, n_orig_frames, sequence_length, h, dim, is_SA=True)
                q_uncond = prepare_extended_QKV(self, q[2*n_frames: 2*n_frames+n_orig_frames], 1, n_orig_frames, sequence_length, h, dim, is_SA=True)
                q_cond = prepare_extended_QKV(self, q[3*n_frames: 3*n_frames+n_orig_frames], 1, n_orig_frames, sequence_length, h, dim, is_SA=True)
                
                out_source_unc_all = []
                out_source_cnd_all = []
                out_uncond_all = []
                out_cond_all = []

                single_batch = True # n_frames <= 12
                b = n_orig_frames if single_batch else 1


                for frame in range(0, n_orig_frames, b):
                    out_source_unc = []
                    out_source_cnd = []
                    out_uncond = []
                    out_cond = []
                    for j in range(h):
                        sim_source_unc_b = torch.bmm(q_src_unc[frame: frame + b, j],
                                                k_src_unc[frame: frame + b, j].transpose(-1, -2)) * self.scale
                        sim_source_cnd_b = torch.bmm(q_src_cnd[frame: frame + b, j],
                                                k_src_cnd[frame: frame + b, j].transpose(-1, -2)) * self.scale
                        sim_uncond_b = torch.bmm(q_uncond[frame: frame + b, j],
                                                k_uncond[frame: frame + b, j].transpose(-1, -2)) * self.scale
                        sim_cond = torch.bmm(q_cond[frame: frame + b, j],
                                            k_cond[frame: frame + b, j].transpose(-1, -2)) * self.scale

                        out_source_unc.append(torch.bmm(sim_source_unc_b.softmax(dim=-1), v_src_unc[frame: frame + b, j]))
                        out_source_cnd.append(torch.bmm(sim_source_cnd_b.softmax(dim=-1), v_src_cnd[frame: frame + b, j]))
                        out_uncond.append(torch.bmm(sim_uncond_b.softmax(dim=-1), v_uncond[frame: frame + b, j]))
                        out_cond.append(torch.bmm(sim_cond.softmax(dim=-1), v_cond[frame: frame + b, j]))

                    out_source_unc = torch.cat(out_source_unc, dim=0)
                    out_source_cnd = torch.cat(out_source_cnd, dim=0)
                    out_uncond = torch.cat(out_uncond, dim=0)
                    out_cond = torch.cat(out_cond, dim=0)
                    if single_batch:
                        out_source_unc = out_source_unc.view(h, n_orig_frames, sequence_length, dim // h).permute(1, 0, 2, 3).reshape(
                            h * n_orig_frames, sequence_length, -1)
                        out_source_cnd = out_source_cnd.view(h, n_orig_frames, sequence_length, dim // h).permute(1, 0, 2, 3).reshape(
                            h * n_orig_frames, sequence_length, -1)
                        out_uncond = out_uncond.view(h, n_orig_frames, sequence_length, dim // h).permute(1, 0, 2, 3).reshape(
                            h * n_orig_frames, sequence_length, -1)
                        out_cond = out_cond.view(h, n_orig_frames, sequence_length, dim // h).permute(1, 0, 2, 3).reshape(
                            h * n_orig_frames, sequence_length, -1)
                    out_source_unc_all.append(out_source_unc)
                    out_source_cnd_all.append(out_source_cnd)
                    out_uncond_all.append(out_uncond)
                    out_cond_all.append(out_cond)

                out_source_unc = torch.cat(out_source_unc_all, dim=0)
                out_source_cnd = torch.cat(out_source_cnd_all, dim=0)
                out_uncond = torch.cat(out_uncond_all, dim=0)
                out_cond = torch.cat(out_cond_all, dim=0)

                # self attention only for the keyframes
                # Q
                q_kf_src_unc = prepare_extended_QKV(self, q[fr_idx_low_src_unc:fr_idx_high_src_unc], 1, n_keyframes, sequence_length, h, dim, is_SA=True)
                q_kf_src_cnd = prepare_extended_QKV(self, q[fr_idx_low_src_cnd:fr_idx_high_src_cnd], 1, n_keyframes, sequence_length, h, dim, is_SA=True)
                q_kf_unc = prepare_extended_QKV(self, q[fr_idx_low_unc:fr_idx_high_unc], 1, n_keyframes, sequence_length, h, dim, is_SA=True)
                q_kf_cnd = prepare_extended_QKV(self, q[fr_idx_low_cnd:fr_idx_high_cnd], 1, n_keyframes, sequence_length, h, dim, is_SA=True)
                # K
                k_kf_src_unc = prepare_extended_QKV(self, k[fr_idx_low_src_unc:fr_idx_high_src_unc], 1, n_keyframes, sequence_length, h, dim, is_SA=True)
                k_kf_src_cnd = prepare_extended_QKV(self, k[fr_idx_low_src_cnd:fr_idx_high_src_cnd], 1, n_keyframes, sequence_length, h, dim, is_SA=True)
                k_kf_unc = prepare_extended_QKV(self, k[fr_idx_low_unc:fr_idx_high_unc], 1, n_keyframes, sequence_length, h, dim, is_SA=True)
                k_kf_cnd = prepare_extended_QKV(self, k[fr_idx_low_cnd:fr_idx_high_cnd], 1, n_keyframes, sequence_length, h, dim, is_SA=True)
                # V
                v_kf_src_unc = prepare_extended_QKV(self, v[fr_idx_low_src_unc:fr_idx_high_src_unc], 1, n_keyframes, sequence_length, h, dim, is_SA=True)
                v_kf_src_cnd = prepare_extended_QKV(self, v[fr_idx_low_src_cnd:fr_idx_high_src_cnd], 1, n_keyframes, sequence_length, h, dim, is_SA=True)
                v_kf_unc = prepare_extended_QKV(self, v[fr_idx_low_unc:fr_idx_high_unc], 1, n_keyframes, sequence_length, h, dim, is_SA=True)
                v_kf_cnd = prepare_extended_QKV(self, v[fr_idx_low_cnd:fr_idx_high_cnd], 1, n_keyframes, sequence_length, h, dim, is_SA=True)

                out_kf_src_unc_all = []
                out_kf_src_cnd_all = []
                out_kf_unc_all = []
                out_kf_cnd_all = []

                b = n_keyframes if single_batch else 1
                for frame in range(0, n_keyframes, b):
                    
                    out_kf_src_unc = []
                    out_kf_src_cnd = []
                    out_kf_unc = []
                    out_kf_cnd = []
                    for j in range(h):
                        sim_kf_src_unc = torch.bmm(q_kf_src_unc[frame: frame + b, j],
                                                k_kf_src_unc[frame: frame + b, j].transpose(-1, -2)) * self.scale
                        sim_kf_src_cnd = torch.bmm(q_kf_src_cnd[frame: frame + b, j],
                                                k_kf_src_cnd[frame: frame + b, j].transpose(-1, -2)) * self.scale
                        sim_kf_unc = torch.bmm(q_kf_unc[frame: frame + b, j],
                                                k_kf_unc[frame: frame + b, j].transpose(-1, -2)) * self.scale
                        sim_kf_cnd = torch.bmm(q_kf_cnd[frame: frame + b, j],
                                                k_kf_cnd[frame: frame + b, j].transpose(-1, -2)) * self.scale

                        out_kf_src_unc.append(torch.bmm(sim_kf_src_unc.softmax(dim=-1), v_kf_src_unc[frame: frame + b, j]))
                        out_kf_src_cnd.append(torch.bmm(sim_kf_src_cnd.softmax(dim=-1), v_kf_src_cnd[frame: frame + b, j]))
                        out_kf_unc.append(torch.bmm(sim_kf_unc.softmax(dim=-1), v_kf_unc[frame: frame + b, j]))
                        out_kf_cnd.append(torch.bmm(sim_kf_cnd.softmax(dim=-1), v_kf_cnd[frame: frame + b, j]))

                    out_kf_src_unc = torch.cat(out_kf_src_unc, dim=0)
                    out_kf_src_cnd = torch.cat(out_kf_src_cnd, dim=0)
                    out_kf_unc = torch.cat(out_kf_unc, dim=0)
                    out_kf_cnd = torch.cat(out_kf_cnd, dim=0)
                    if single_batch:
                        out_kf_src_unc = out_kf_src_unc.view(h, n_keyframes, sequence_length, dim // h).permute(1, 0, 2, 3).reshape(
                            h * n_keyframes, sequence_length, -1)
                        out_kf_src_cnd = out_kf_src_cnd.view(h, n_keyframes, sequence_length, dim // h).permute(1, 0, 2, 3).reshape(
                            h * n_keyframes, sequence_length, -1)
                        out_kf_unc = out_kf_unc.view(h, n_keyframes, sequence_length, dim // h).permute(1, 0, 2, 3).reshape(
                            h * n_keyframes, sequence_length, -1)
                        out_kf_cnd = out_kf_cnd.view(h, n_keyframes, sequence_length, dim // h).permute(1, 0, 2, 3).reshape(
                            h * n_keyframes, sequence_length, -1)
                    out_kf_src_unc_all.append(out_kf_src_unc)
                    out_kf_src_cnd_all.append(out_kf_src_cnd)
                    out_kf_unc_all.append(out_kf_unc)
                    out_kf_cnd_all.append(out_kf_cnd)

                out_kf_src_unc = torch.cat(out_kf_src_unc_all, dim=0)
                out_kf_src_cnd = torch.cat(out_kf_src_cnd_all, dim=0)
                out_kf_unc = torch.cat(out_kf_unc_all, dim=0)
                out_kf_cnd = torch.cat(out_kf_cnd_all, dim=0)

                out = torch.cat([out_source_unc, out_kf_src_unc, out_source_cnd, out_kf_src_cnd, out_uncond, out_kf_unc, out_cond, out_kf_cnd], dim=0)
                out = self.batch_to_head_dim(out)

                return to_out(out)
            
            # temporal slices
            else:
                n_frames = batch_size
                is_cross = encoder_hidden_states is not None
                encoder_hidden_states = encoder_hidden_states if is_cross else x
                q = self.to_q(x)
                k = self.to_k(encoder_hidden_states)
                v = self.to_v(encoder_hidden_states)

                # self attention only for temporal frames

                q = self.head_to_batch_dim(q)
                k = self.head_to_batch_dim(k)
                v = self.head_to_batch_dim(v)

                q = q.view(n_frames, h, sequence_length, dim // h)
                k = k.view(n_frames, h, sequence_length, dim // h)
                v = v.view(n_frames, h, sequence_length, dim // h)

                out_all = []
                single_batch = True # n_frames <= 12
                b = n_frames if single_batch else 1
                for frame in range(0, n_frames, b):
                    
                    out_ = []
                    for j in range(h):
                        sim = torch.bmm(q[frame: frame + b, j],
                                                k[frame: frame + b, j].transpose(-1, -2)) * self.scale
                       
                        out_.append(torch.bmm(sim.softmax(dim=-1), v[frame: frame + b, j]))

                    out_ = torch.cat(out_, dim=0)

                    if single_batch:
                        out_ = out_.view(h, n_frames, sequence_length, dim // h).permute(1, 0, 2, 3).reshape(
                            h * n_frames, sequence_length, -1)
                       
                    out_all.append(out_)

                out_= torch.cat(out_all, dim=0)

                out = out_
                out = self.batch_to_head_dim(out)

                return to_out(out)

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.forward = sa_forward(module.attn1)
            setattr(module.attn1, 'injection_schedule', [])


    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)


def register_conv_injection(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb, scale):
            scale = None
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 4)
                # inject unconditional src
                hidden_states[2*source_batch_size:3 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[3 * source_batch_size:] = hidden_states[source_batch_size:2*source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1] # layer 4 only?
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)
