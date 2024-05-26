import argparse
import sys
import os 
import glob
import torch

from diffusers import StableDiffusionPipeline, DDIMScheduler

from Slicedit.inversion_utils import slicedit_ddpm_inversion_loop
from Slicedit.video_utils import load_512, extract_images, set_seed, dataset_from_yaml


def print_details(args, skip, cfg_dec, prompt_enc, prompt_tar, qk_inj, alpha, nbr_frames):
    print("---------")
    print("Working on:")
    print("video name: " + args.video_name)
    print("video path: " + args.video_path)
    print("exp_name: " + args.exp_name)
    print("skip: " + str(skip))
    print("decoder: " + str(cfg_dec))
    print("num_diffusion_steps: " + str(args.num_diffusion_steps))
    print("prompt encoder: " + prompt_enc)
    print("prompt decoder: " + prompt_tar)
    print("alpha: " + str(alpha))
    print("qk_inj: " + str(qk_inj))
    print("nbr_frames: " + str(nbr_frames))
    print("---------")

## main function prepares the data and runs slicedit for all videos and prompts in the dataset yaml file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## path to the yaml file with the dataset parameters. If not provided, argparse values are used!
    parser.add_argument("--dataset_yaml",  type=str, default="./yaml_files/dataset_configs/parkour.yaml")
    ## path to the yaml file with the experiment parameters. If not provided, argparse values are used!
    parser.add_argument("--exp_config",  type=str, default="./yaml_files/exp_configs/default_exp_params.yaml") #
    ## argpase parameters (overwritten by yaml files if provided)
    # general parameters
    parser.add_argument("--device_num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name",  default="slicedit_test_")
    parser.add_argument("--skip", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--use_negative_tar_prompt", action='store_true')
    parser.add_argument("--x_and_y", action='store_true')
    # prompt parameters
    parser.add_argument("--prompt_enc",  default="")
    parser.add_argument("--prompt_dec",  default="")
    parser.add_argument("--cfg_enc", type=float, default=3.5)
    parser.add_argument("--cfg_dec", type=int, default=10)
    # percentage of attention and conv feature injection, similarly to https://arxiv.org/abs/2211.12572
    parser.add_argument("--qk_inj", type=int, default=85) # 100% means injection in all steps (num_diffusion_steps-skip)
    parser.add_argument("--conv_inj", type=int, default=0) # 100% means injection in all steps (num_diffusion_steps-skip)
    # diffusion parameters
    parser.add_argument("--model_id", type=str, default ="stabilityai/stable-diffusion-2-1-base") 
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=1)
    # video parameters (default taken from dataset yaml)
    parser.add_argument("--video_path", type=str, default ="")
    parser.add_argument("--video_name", type=str, default ="")
    # negative time prompt for the ST slices (off by default cfg_time=1)
    parser.add_argument("--cfg_time", type=float, default=1)
    parser.add_argument("--prompt_time",  default="")
    parser.add_argument("--negative_time_prompt", type=str, default="jittery")  
    # enable xformers for (slightly faster) inference (if supported)
    parser.add_argument("--use_xformers", action='store_true')

    args = parser.parse_args()

    ## parse the yaml files

    # overwrite argparse with experiment yaml
    if args.exp_config != "":
        # iterate over all the options from config yaml and change args accordingly
        config_yaml_dict = dataset_from_yaml(args.exp_config)
        for key in config_yaml_dict.keys():
            setattr(args, key, config_yaml_dict[key])

    # overwrite argparse with dataset yaml
    if args.dataset_yaml != "":
        full_data = dataset_from_yaml(args.dataset_yaml)
    else:
        full_data = [{'video_name': args.video_name,'source_prompt': args.prompt_enc,'target_prompts': [args.prompt_dec]}]
    # set seed
    seed = args.seed
    set_seed(seed)

    assert args.eta > 0, "eta must be greater than 0 for DDPM"
    
    device = f"cuda:{args.device_num}"
    
    # load/reload model: "stabilityai/stable-diffusion-2-1-base"
    ldm_stable = StableDiffusionPipeline.from_pretrained(args.model_id).to(device)
    if args.use_xformers:
        ldm_stable.enable_xformers_memory_efficient_attention()

    ldm_stable.scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    ldm_stable.scheduler.set_timesteps(args.num_diffusion_steps)

    prompt_time = args.prompt_time 

    print(full_data)

    # run slicedit for all videos in the provided dataset yaml
    for i in range(len(full_data)):
        current_video_data = full_data[i]
        args.video_name = current_video_data['video_name']
        print(args.video_name)

        args.video_path =  './Videos/' + args.video_name + '.mp4'

        print(args.video_path)
        print(args.exp_name)
        prompt_enc = current_video_data.get('source_prompt', "") # default empty string
        prompt_tar_list = current_video_data['target_prompts']

        # extract frames from the video and resize them
        frame_number, orig_frame_size = extract_images(args.video_path, f'{args.video_name}', resize=True)
        torch.save(torch.tensor(orig_frame_size), f'data/{args.video_name}/orig_frame_size.pt')

        # verify there is enough frames
        if frame_number < 64:
            print('Error : not enough frames')
            sys.exit()
        if frame_number > 210:
            print('clipping video to 210 frames')
            frame_number = 210

        print(full_data[i])
                

        # run slicedit for all target prompts
        for k in range(len(prompt_tar_list)):
            prompt_tar = prompt_tar_list[k]
            print_details(args,args.skip,args.cfg_dec,prompt_enc,prompt_tar,args.qk_inj,args.alpha, frame_number)

            eps = [torch.randn(1, 4, 64, 64, device=device) for _ in range(args.num_diffusion_steps)]
            # encode video with VAE
            ws_video = []
            for j in range(frame_number):
                image_path = f'data/{args.video_name}/frame{j}.png'
                offsets=(0,0,0,0)
                x0 = load_512(image_path, *offsets, device)
                # vae encode image
                with torch.autocast("cuda"), torch.inference_mode():
                    w0 = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).float()
                ws_video.append(w0)

            save_path = f'output_data/{args.video_name}_{args.exp_name}/{prompt_tar}/d_stps_{args.num_diffusion_steps}_alpha_{args.alpha}_cfg_e_{args.cfg_enc}_cfg_d_{args.cfg_dec}_skip_{args.skip}_seed_{seed}_qk_inj_{args.qk_inj}'
            
            try:
                os.makedirs(save_path, exist_ok=False)
                print(save_path)
            except:

                if len(glob.glob(save_path + 'frames/*.png') ) > 0:
                    print(save_path, "already exists")
                    continue
                else:
                    print(save_path, "path exists but no frames found, continuing...")

            # save all args to txt file in save_path
            try:
                with open(save_path + '/args.txt', 'w') as f:
                    for arg in vars(args):
                        print(f"{arg}: {getattr(args, arg)}", file=f)
                        print("", file=f)
            except:
                print("Failed to save args.txt")

            slicedit_ddpm_inversion_loop(ldm_stable, ws_video,
                        etas = args.eta, 
                        prog_bar = True,
                        src_prompt = prompt_enc,
                        tar_prompt = prompt_tar,
                        prompt_time = args.prompt_time,
                        cfg_scale = args.cfg_enc,
                        alpha=args.alpha,
                        num_inference_steps=args.num_diffusion_steps, 
                        eps = eps, 
                        n_frames=frame_number,
                        skip=args.skip, 
                        cfg_scale_tar=args.cfg_dec, 
                        save_path=save_path, 
                        orig_frame_size=orig_frame_size, 
                        qk_injection_t=args.qk_inj, 
                        conv_injection_t=args.conv_inj, 
                        negative_time_prompt=args.negative_time_prompt, 
                        cfg_scale_time=args.cfg_time,
                        use_negative_tar_prompt=args.use_negative_tar_prompt,
                        x_and_y=args.x_and_y)
        
