# Liver Vessel Segmentation with a 3D U-Net (PyTorch + MONAI)

Binary segmentation of liver vessels (hepatic and portal veins) on abdominal CT, trained on the public Medical Segmentation Decathlon dataset. This project was built to learn the full deep-learning pipeline for 3D medical imaging: data preparation, patch-based training, evaluation and error analysis.

## Results

Evaluation on **60 held-out validation cases** (split by patient, seed 0), Dice score on the vessel class, best checkpoint (epoch 60):

| Metric | Value |
|---|---|
| Mean Dice | **0.550** |
| Standard deviation | 0.103 |
| Median Dice | 0.550 |
| Min / Max | 0.319 / 0.778 |

Per-case values are in `final_results.json`.

![Training curves](images/curves.png)

![Predictions](images/predictions.png)

*Red: ground truth. Blue: prediction.*

### What works and what does not
- The model finds the main vessel branches in many cases (example: case 0).
- It misses many **small branches**, and in some cases it **under-segments large vessels** close to the heart level (example: case 1, where only small spots are predicted for a large structure). These failure cases are kept on purpose in the figure above.
- The validation Dice (measured on 15 cases during training) rose from 0.41 at epoch 5 to about 0.54 at epoch 40, then stayed almost flat until epoch 60, while the training loss was still decreasing slowly.

### Limits of this evaluation
- There is **no separate test set**. The best checkpoint was selected using the Dice on 15 of the 60 validation cases, so the reported value may be slightly optimistic.
- Only the Dice score is reported (no surface or topology metrics such as clDice), and the training was run once (no cross-validation, one seed).

## Data

[Medical Segmentation Decathlon](https://arxiv.org/abs/1902.09063), Task08 HepaticVessel: 303 labelled CT volumes. Labels: 0 background, 1 vessel, 2 tumour. In this project the task is **vessel vs. everything else** (the tumour label is merged with the background). Typical volume: 512 x 512 x 49 voxels with a 0.92 x 0.92 x 5.0 mm spacing. Vessels represent on average about 0.12 % of the voxels (20-case estimate), which is why a Dice-based loss is used.

Please check the dataset licence and terms on the Medical Segmentation Decathlon website before reusing the data. The data is not included in this repository.

## Method

**Preprocessing**
- RAS orientation, resampling to 1.5 x 1.5 x 2.0 mm
- CT window [-100, 250] HU scaled to [0, 1]
- 243 training / 60 validation cases, split by patient

**Training patches**
- 3D patches of 96 x 96 x 96 voxels, 2 per case, with a 1:1 ratio of patches centred on vessels and on background
- Augmentation: random flips (2 axes), 90-degree rotations, small intensity shift

**Model**
- MONAI 3D `UNet`, channels (16, 32, 64, 128, 256), 2 residual units per level, instance normalisation, 4.81 M parameters, 2 output channels

**Loss and optimisation**
- Dice + cross-entropy loss (`DiceCELoss`, background excluded from the Dice term)
- AdamW, learning rate 1e-3, weight decay 1e-5, cosine annealing, 60 epochs
- 2 cases per step (4 patches), mixed precision, one T4 GPU (Google Colab)

**Inference**
- Sliding window (96 x 96 x 96, overlap 0.25) on the whole volume
- Metric: Dice on the vessel class, computed per case

## Reproduce

Open `Liver_Vessel_UNet_Full_Pipeline.ipynb` on Google Colab (GPU runtime) and run the cells from top to bottom. The notebook follows four stages:

1. **Setup and data exploration**: environment, dataset download, first look at the data
2. **Data pipeline**: patient-level split, preprocessing, patch sampling
3. **Model and short smoke test**: 3D U-Net, Dice + cross-entropy loss, a first short training to check that the loss decreases
4. **Full training and evaluation**: overfit sanity check on 2 cases, full resumable training (60 epochs), evaluation on the 60 validation cases

The dataset is downloaded automatically by MONAI into your Google Drive. The full training takes several hours on a T4 GPU, so it is written to be stopped and resumed from a checkpoint.

Versions used: MONAI 1.6.1, PyTorch 2.14.1 (CUDA 13).

## Ideas to improve

- Longer training and a learning-rate schedule with restarts
- Compare with nnU-Net as a strong baseline
- Add a topology-aware loss (for example clDice) and report topology metrics, since thin branches matter more than their voxel count suggests
- Test-time augmentation and 5-fold cross-validation with a proper held-out test set
- Use the model as a starting point for low-resolution to high-resolution vessel enhancement

## Acknowledgements

Built with [MONAI](https://monai.io) and PyTorch. Dataset: Medical Segmentation Decathlon (Simpson et al., arXiv:1902.09063).
