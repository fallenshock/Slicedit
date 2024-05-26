import os
import glob
import cv2
import torch 
import numpy as np 
import random
import yaml

import os
import cv2


from PIL import Image

from PIL import Image, ImageDraw ,ImageFont
import torchvision.transforms as T

from torchvision.io import write_video

def create_video_from_frames(frame_directory):

	video_directory = '/'.join(frame_directory.split('/')[:-1])
	video_name = os.path.join(video_directory, "slicedit_out.mp4")

	# Check if video already exists
	# if so, skip this directory
	if os.path.exists(video_name):
		print(f"Video already exists for {video_name} !")
		return

	images = [img for img in sorted(os.listdir(frame_directory), key=lambda x: (len(x), x))
				if img.endswith(".jpg") or
				img.endswith(".jpeg") or
				img.endswith("png")]

	if len(images) > 0:
		height, width, _ = cv2.imread(os.path.join(frame_directory, images[0])).shape
		
		video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), 25, (width, height), True)
		
		print("Generating Video...")
		for image in images:
			curr_frame = cv2.imread(os.path.join(frame_directory, image))
			
			video.write(curr_frame)

		cv2.destroyAllWindows()
		video.release()

		compressed_video_name = os.path.join(video_directory, "Slicedit_out_compressed.mp4")

		print(compressed_video_name)
		# Convert video to higher MPEG-4 compression
		os.system(f'ffmpeg -i "{video_name}" -c:v libx264 -crf 23 -preset medium -y "{compressed_video_name}"')

		print(f"Video created for {frame_directory}")


# use this if you are having problems with ffmpeg
def save_video(raw_frames, save_path, fps=25, orig_size=(512, 512)):
	video_codec = "libx264"
	video_options = {
		"crf": "23",  
		"preset": "medium", 
	}
	# squeeze and reshape frames to orig size using LANCOZ interpolation
	frames = torch.nn.functional.interpolate(raw_frames.squeeze(), orig_size, mode="area")
	
	frames = (((frames+1)/2).clamp(0, 1) * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
	write_video(save_path, frames, fps=fps, video_codec=video_codec, options=video_options)


def add_margin(pil_img, top = 0, right = 0, bottom = 0, 
					left = 0, color = (255,255,255)):
	width, height = pil_img.size
	new_width = width + right + left
	new_height = height + top + bottom
	result = Image.new(pil_img.mode, (new_width, new_height), color)
	
	result.paste(pil_img, (left, top))
	return result


def tensor_to_pil(tensor_imgs):
	if type(tensor_imgs) == list:
		tensor_imgs = torch.cat(tensor_imgs)
	tensor_imgs = (tensor_imgs / 2 + 0.5).clamp(0, 1)
	to_pil = T.ToPILImage()
	pil_imgs = [to_pil(img) for img in tensor_imgs]    
	return pil_imgs


def image_grid(imgs, rows = 1, cols = None, 
					size = None,
				   titles = None, text_pos = (0, 0)):
	if type(imgs) == list and type(imgs[0]) == torch.Tensor:
		imgs = torch.cat(imgs)
	if type(imgs) == torch.Tensor:
		imgs = tensor_to_pil(imgs)
		
	if not size is None:
		imgs = [img.resize((size,size)) for img in imgs]
	if cols is None:
		cols = len(imgs)
	assert len(imgs) >= rows*cols
	
	top=20
	w, h = imgs[0].size
	delta = 0
	if len(imgs)> 1 and not imgs[1].size[1] == h:
		delta = top
		h = imgs[1].size[1]
	if not titles is  None:
		font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 
									size = 20, encoding="unic")
		h = top + h 
	grid = Image.new('RGB', size=(cols*w, rows*h+delta))    
	for i, img in enumerate(imgs):
		
		if not titles is  None:
			img = add_margin(img, top = top, bottom = 0,left=0)
			draw = ImageDraw.Draw(img)
			draw.text(text_pos, titles[i],(0,0,0), 
			font = font)
		if not delta == 0 and i > 0:
			grid.paste(img, box=(i%cols*w, i//cols*h+delta))
		else:
			grid.paste(img, box=(i%cols*w, i//cols*h))
		
	return grid    


def load_512(image_path, left=0, right=0, top=0, bottom=0, device=None):
	if type(image_path) is str:
		image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
	else:
		image = image_path
	h, w, c = image.shape
	left = min(left, w-1)
	right = min(right, w - left - 1)
	top = min(top, h - left - 1)
	bottom = min(bottom, h - top - 1)
	image = image[top:h-bottom, left:w-right]
	h, w, c = image.shape
	if h < w:
		offset = (w - h) // 2
		image = image[:, offset:offset + h]
	elif w < h:
		offset = (h - w) // 2
		image = image[offset:offset + w]
	image = np.array(Image.fromarray(image).resize((512, 512)))
	image = torch.from_numpy(image).float() / 127.5 - 1
	image = image.permute(2, 0, 1).unsqueeze(0).to(device)

	return image


def set_seed(seed=42):
	random.seed(seed)	
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def extract_images(path, name, resize):
	print(f"Extracting frames from video...")
	# read the video from specified path
	cam = cv2.VideoCapture(path)
	if os.path.exists(f'data/{name}') and len(glob.glob(os.path.join(f'data/{name}/', '*.png')))>0:
		ret, frame = cam.read()
		return len(glob.glob(os.path.join(f'data/{name}/', '*.png'))), frame.shape
	try:
		# creating a folder named data
		if not os.path.exists(f'data/{name}'):
			os.makedirs(f'data/{name}')
	# if not created then raise error
	except OSError:
		print('Error: Creating directory of data')

	# frame
	current_frame = 0

	while(True):
		# reading frames from video 
		ret, frame = cam.read()
		if ret:
			shape = frame.shape
			path_name = f'data/{name}/frame' + str(current_frame) + '.png'
			print ('Creating... ' + path_name)
			# saving the extracted frames and resize if needed 
			if resize: 
				resize_frame = cv2.resize(frame, dsize=(512,512), interpolation=cv2.INTER_CUBIC)
				cv2.imwrite(path_name, resize_frame)
			else:
				cv2.imwrite(path_name, frame)

			current_frame += 1
			last_frame = frame
		else:
			break
	if current_frame == 63 or current_frame == 62: 
		while (current_frame != 64):
			path_name = f'data/{name}/frame' + str(current_frame) + '.png'
			print ('Copying... ' + path_name)

			if resize: 
				resize_frame = cv2.resize(last_frame, dsize=(512,512), interpolation=cv2.INTER_CUBIC)
				cv2.imwrite(path_name, resize_frame)
			else:
				cv2.imwrite(path_name, last_frame)
			current_frame += 1
	if current_frame == 40: 
		while (current_frame != 64):
			path_name = f'data/{name}/frame' + str(current_frame) + '.png'
			print ('Copying... ' + path_name)

			if resize: 
				resize_frame = cv2.resize(last_frame, dsize=(512,512), interpolation=cv2.INTER_CUBIC)
				cv2.imwrite(path_name, resize_frame)
			else:
				cv2.imwrite(path_name, last_frame)
			current_frame += 1
		
	# release all space and windows once done
	cam.release()
	cv2.destroyAllWindows()
	return current_frame, shape


def resize_image_to_original(path, save_path, size):
	img = cv2.imread(path)
	res = cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)
	cv2.imwrite(save_path, res)
  

def dataset_from_yaml(yaml_location):
	with open(yaml_location, 'r') as stream:
		data_loaded = yaml.safe_load(stream)

	return data_loaded
