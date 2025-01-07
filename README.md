# Trajectory Prediction

## Highlight

- Simple and Lightweight: Requires no additional CUDA libraries, ensuring simplicity and ease of use.
- Efficient on Embedded Devices: Designed to be friendly and effective on embedded devices.

<p align="center">
    <img src="https://raw.githubusercontent.com/KexianShen/trajectory-prediction/media/mrm.png?raw=true"><br/>
    <em>Masked road model</em>
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/KexianShen/trajectory-prediction/media/mtm.png?raw=true"><br/>
    <em>Masked trajectory model</em>
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/KexianShen/trajectory-prediction/media/all.png?raw=true"><br/>
    <em>Trajectory prediction</em>
</p>

## Getting Started

- [Environment](#environment)
- [Preprocess](#preprocess)
- [Train](#train)
- [Eval](#eval)

## Environment

```bash
pip install -r requirements.txt
```

## Preprocess

```bash
# Prepare data
# Manually download from: https://www.argoverse.org/av2.html#download-link
data_root
├── train
│   ├── 0000b0f9-99f9-4a1f-a231-5be9e4c523f7
│   ├── 0000b6ab-e100-4f6b-aee8-b520b57c0530
│   ├── ...
├── val
│   ├── 00010486-9a07-48ae-b493-cf4545855937
│   ├── 00062a32-8d6d-4449-9948-6fedac67bfcd
│   ├── ...
├── test
│   ├── 0000b329-f890-4c2b-93f2-7e2413d4ca5b
│   ├── 0008c251-e9b0-4708-b762-b15cb6effc27
│   ├── ...
```

> [!TIP]
>
> Download data with `s5cmd`.
> ```bash
> conda install s5cmd -c conda-forge
>
> s5cmd --no-sign-request cp "s3://argoverse/datasets/av2/motion-forecasting/*" <data_root>
> ```

```bash
# Set data_root in conf/config.yaml
python preprocess.py
```

## Train

```bash
# Train mrm and mtm models
python train.py model=model_mrm epochs=20

python train.py model=model_mtm epochs=20

# Set mrm_checkpoint, mtm_checkpoint in conf/config.yaml for combined training
python train.py monitor=val_AvgMinFDE
```

## Eval

```bash
# Set checkpoint in conf/config.yaml for evaluation
python eval.py test=true
```

## Credits
[forecast-mae](https://github.com/jchengai/forecast-mae), [sept](https://arxiv.org/abs/2309.15289)
