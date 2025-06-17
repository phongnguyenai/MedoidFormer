# MedoidFormer

## 1. Installation

We recommend creating a conda environment (CUDA version 11.8 is required).
Simply run:

```
sh ./env.sh
```

## 2. Dataset

Sample data is provided in the `datasets` folder. These samples are sufficient for running inference.
The full dataset will be made publicly available after the paper is published.

## 3. Model Training and Inference

**Training:**

```
python train.py --num-lmk 68 --model-name MedoidFormer --dataset-path datasets/Facescape
```

**Inference:**

```
python inference.py --num-lmk 68 --weights-path pretrained/MedoidFormer/best.pt --dataset-path datasets/Facescape
```

For pretrained models or further information, please contact the corresponding author at:
📧 [ttpn997@uowmail.edu.au](mailto:ttpn997@uowmail.edu.au)
