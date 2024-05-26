[![Zero-Shot Video Editing](https://img.shields.io/badge/zero%20shot-video%20editing-Green)](https://github.com/topics/video-editing)
[![Python](https://img.shields.io/badge/python-3.8+-blue?python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/release/python-38/)
![PyTorch](https://img.shields.io/badge/torch-2.0.0-red?PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

# Slicedit

[Project](https://matankleiner.github.io/slicedit/) | [Arxiv](https://arxiv.org/abs/2405.12211) 
### [ICML 2024] Official pytorch implementation of the paper: "Slicedit: Zero-Shot Video Editing With Text-to-Image Diffusion Models Using Spatio-Temporal Slices"

https://github.com/fallenshock/Slicedit/assets/63591190/bef5e826-726a-41fe-9160-2e9e27f7423b

## Installation
1. Clone the repository

2. Install the required dependencies: `pip install -r requirements.txt`
	* Tested with CUDA version 12.0 and diffusers 0.21.2
## Usage
1. Place desired input videos into `Videos` folder

2. Place desired dataset config yaml file into `yaml_files/dataset_configs`

3. Change experiment config .yaml if desired in `yaml_files/exp_configs`

    Note: The dataset config specifies the video name, source prompt and target prompt(s).
     Experiment configs specify the hyperparameters for the run. Use the provided default yamls as reference.

4. Run `python main.py --dataset_yaml <path to ds yaml>`
	* Optional: passing `--use_negative_tar_prompt` improves sharpness.

## License
This project is licensed under the [MIT License](LICENSE).


### Citation
If you use this code for your research, please cite our paper:

```
@article{
	cohen2024slicedit,
	title={Slicedit: Zero-Shot Video Editing With Text-to-Image Diffusion Models Using Spatio-Temporal Slices},
	author={Cohen, Nathaniel and Kulikov, Vladimir and Kleiner, Matan and Huberman-Spiegelglas, Inbar and Michaeli, Tomer},
	journal={arXiv preprint arXiv:2405.12211},
	year={2024},
}
```
