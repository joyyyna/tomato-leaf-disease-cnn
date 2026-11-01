# Tomato Leaf Disease Classification with a CNN

Computer vision / image processing course project (USTO-MB, M1 AI & Applications, 2025/2026).

A convolutional neural network that classifies tomato leaf images into 6 classes (5 diseases + healthy), plus a study of whether an image-enhancement preprocessing step improves the results.

## Pipeline

1. **Dataset**: [Tomato Leaf Disease Dataset (6 classes)](https://www.kaggle.com/datasets/syedhashirali260/tomato-leaf-disease-dataset-6-classes), class counts and sample images
2. **Preprocessing**: BGR → LAB, bilateral filter on the L channel, then CLAHE; quality measured with SSIM and PSNR
3. **Training**: 70 / 20 / 10 stratified train / validation / test split, images resized to 128×128 and scaled to [0, 1]
4. **Model**: 3 × (Conv2D → BatchNorm → MaxPool → Dropout), then Flatten → Dense(256) → BatchNorm → Dropout → Dense(6, softmax); `EarlyStopping` and `ReduceLROnPlateau` callbacks (both on `val_loss`)
5. **Evaluation**: classification report and confusion matrix on the test set
6. **Ablation**: the same model, same hyperparameters and seed, trained on the original images without preprocessing, then compared on the same test set (accuracy, macro F1, per-class F1)

## Repository content

| Path | Description |
|------|-------------|
| `tomato_leaf_disease_cnn.ipynb` | The full notebook |
| `requirements.txt` | Python dependencies |

The dataset is **not** included in this repository. Download it from Kaggle (link above).

## Run it

**On Kaggle:** https://www.kaggle.com/code/narimened/cnn-tomato-leaf-disease-classification

**Locally:**

1. Download the dataset and extract it so that the `Tomato_Leaf_Dataset/` folder sits next to the notebook.
2. Install the dependencies and open the notebook:

```bash
pip install -r requirements.txt
jupyter notebook tomato_leaf_disease_cnn.ipynb
```

The notebook uses the Kaggle input folder when it exists, and the local `Tomato_Leaf_Dataset/` folder otherwise. Plots and trained models are saved in `/kaggle/working` on Kaggle, or in `outputs/` locally.

Training is seeded (`SEED = 42`) and uses the same split, initialization and callbacks for both experiments, but results can still vary slightly across hardware and TensorFlow versions. On Kaggle, training takes much less time with a GPU accelerator turned on (notebook settings). The notebook also prints a warning if a model ends up near chance-level accuracy, which means the training failed and the cell should be re-run.
