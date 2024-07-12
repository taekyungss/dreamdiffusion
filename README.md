# DreamDiffusion: Generating High-Quality Images from Brain EEG Signals
<p align="center">
<img src=assets/eeg_teaser.png />
</p>


### stage1 pretrain code

```sh
python -m torch.distributed.launch --nproc_per_node=4 code/stageA1_eeg_pretrain.py
```

### stage2 code

```sh
CUDA_VISIBLE_DEVICES=6,7 python code/eeg_ldm.py
```