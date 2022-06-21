# How to use dockerized denoiser on your dataset (fb-demucs)

## 0. System Requirements
- _MANDATORY:_ Docker installed, up and running
- _RECOMMENDED:_ 8GB+ free RAM space

## 1. Installation
fb-demucs is just [Facebook Open Source Denoiser](https://github.com/facebookresearch/denoiser) wrapped into easy to use at any platform Docker image with all pretrained models included.
<br>Though all code is open-source anyway, image is [published to DockerHub](https://hub.docker.com/r/ruk0sh/fb-demucs/tags) (this link may not work on your computer due to proxy settings), so anyone anywhere may download it to host system by command:
```bash
[sudo] docker pull ruk0sh/fb-demucs:0.4-with-models
```
- _NOTE:_ Current recommended tag is `0.4-with-models`

## 2. Features
- All pretrained models included, you will need only to download image itself (which may take 30+ min to honest).
- All denoiser command line arguments bypassed to running container for ease and flexibility of usage.
- This is just another Docker image, so all freedom Docker grants belongs to you.

## 3. Usage
* Show help and all possible denoiser command line arguments:
```bash
docker run --rm ruk0sh/fb-demucs:0.4-with-models --help
```
`--rm` (remove) flag is important because new container created on each run and stopped after execution and without this flag there will be evergroing number of inactive containers on your system.
Use `docker ps -a` to show stopped containers and remove them by hand if you don't need them.
_NOTE_: everything after image tag is just command line arguments of denoiser inself bypassed to _`python -m denoiser.enhance`_ inside running container.
* Denoise your data:
```bash
sudo docker run --rm --shm-size 8G \
  -v /path/to/dataset/on/your/computer:/app/data \
  ruk0sh/fb-demucs:0.4-with-models \
  --master64 --noisy_dir=data/audio --out_dir=data/audio-denoised
```
Some more important `docker run` flags here:

`--shm-size 8G` sets size of shared memory allocated for runnning container to 8 GB. You may try other values.
<br>By default it's 64 MB only, which is way to low for purpose of solid ML model inference.
<br>⚠️ **If you will omit this flag, you may experience strange bus-related errors or even worse get your data partially processed**

`-v /path/on/your/system:/path/inside/running/container` mounts _/path/on/your/system_ to _/path/inside/running/container_
<br>_NOTE:_ Remember not only your input folder should be in mounted filesystem subtree, but output folder too, or data will be created succesfully, but then destroyed after container stops.
### 3.1. Added denoiser flags
⚠️ This features were added to denoiser after cloning the original repo and were not tested very thorougly.
**Use them at your own risk.**

`-e wav mp3 m4a etc` or `--valid_extensions wav ...`
<br> Pass space separated list of file extensions which denoiser would consider audio files then scanning directory.
<br> Checkout [list of supported audio formats](https://pytorch.org/audio/stable/backend.html#load), additional M4A support was hardcoded for my own purpose.

`--sr INTEGER`
<br> Provide sample rate if you wish to automatically resample denoised audio files.
<br> By default denoiser resamples any input to 16000 Hz and produces 16000 Hz outputs. So remember you may experience quality loss when upscaling.

`--keep_noisy`
<br> If this provided, original audios will be saved to output folder with `_noisy` postfix.
<br> NOTE: This overrides behavior of original denoiser, which kept original audio files in output by default.
