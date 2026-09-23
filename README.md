# Mars Landmark Classification

## Project Question

Can image classification models identify Mars surface landmarks from satellite
image patches?

## Purpose

The project classifies Mars satellite image patches into landmark categories
from the data available in the notebook. It supports model comparison and
computer vision learning. It does not provide a validated planetary science
classifier or production decision system.

## Results Summary

The strongest saved notebook result came from the Vision Transformer. It
reported 97.07% test accuracy, 0.9708 weighted precision, 0.9707 weighted
recall, 0.9707 weighted F1 score, and 0.998 micro-average AUC. In practical
terms, it classified about 97 out of every 100 held-out Mars image patches
correctly in this run.

The VGG16 transfer learning model also learned useful image patterns, but its
saved weighted scores were lower. It reported 0.8350 weighted precision, 0.8338
weighted recall, and 0.8342 weighted F1 score.

These results support model comparison and early research. They do not support
mission-critical science, automated geologic labeling, or operational deployment.

## Dataset

Source: [HiRISE Mars Landmark Dataset](https://www.kaggle.com/datasets/glorianikitchuk/hirise-mars-landmark)

The notebook downloads the dataset with `opendatasets` when the local
`hirise-mars-landmark` folder does not exist.

It reads labels from:

`hirise-mars-landmark/labels-map-proj-v3.txt`

### Inputs

- Images: Mars satellite image patches loaded from the dataset image folder.
- Image size: each image is resized to `128 x 128 x 3`.
- Modeled image data shape: `(16270, 128, 128, 3)`.

### Target

The target variable is the landmark class label. The notebook models six labels:

- `other`
- `crater`
- `dark_dune`
- `slope_streak`
- `bright_dune`
- `swiss_cheese`

### Size and Filtering

The original label file contains 73,031 records and eight labels.

Original class counts:

| Class | Count |
| --- | ---: |
| `other` | 61,054 |
| `crater` | 4,900 |
| `dark_dune` | 1,141 |
| `slope_streak` | 2,331 |
| `bright_dune` | 1,750 |
| `impact_ejecta` | 231 |
| `swiss_cheese` | 1,148 |
| `spider` | 476 |

The notebook reduces the `other` class to 5,000 samples. It removes
`impact_ejecta` and `spider`. The final modeled dataset contains 16,270 images
across six labels.

Modeled class counts after preprocessing:

| Class | Count |
| --- | ---: |
| `other` | 5,000 |
| `crater` | 4,900 |
| `dark_dune` | 1,141 |
| `slope_streak` | 2,331 |
| `bright_dune` | 1,750 |
| `swiss_cheese` | 1,148 |

The notebook uses a 75 percent training split and a 25 percent test split. The
resulting split contains 12,202 training images and 4,068 test images.

The Vision Transformer section prints an extra balancing step that selects 2,000
training samples each from `other` and `crater`. The following DataLoader cell
uses the regular training arrays. Review that section before changing the ViT
experiment.

### Dataset Limits

- The original dataset has strong class imbalance.
- The modeled dataset removes `impact_ejecta` and `spider`.
- The six modeled classes still have uneven sample counts.
- Resizing images to `128 x 128` can reduce surface detail.
- The notebook does not document a separate image-quality filtering step.
- Some filenames indicate rotated, flipped, or brightness-adjusted variants.
- The results may not transfer to new Mars image sources without more testing.

## Metric Definitions

The notebook reports classification metrics.

- Accuracy: measures the share of test images classified correctly.
- Precision: measures how often a predicted class is correct.
- Recall: measures how many real examples of a class the model finds.
- F1 score: combines precision and recall into one score.
- AUC: measures how well the model ranks the correct class above incorrect
  classes.
- Confusion matrix: shows correct predictions on the diagonal and mistakes
  outside the diagonal.

A 97.07% test accuracy means the model classified about 97 out of 100 test
images correctly. It does not mean the model will reach the same result on new
datasets.

## Models

All models in the notebook classify Mars landmark images. They are classification
models, not regression models. The notebook evaluates them with accuracy,
precision, recall, F1 score, AUC, and confusion matrices.

### Custom CNN

The custom CNN uses TensorFlow and Keras. It takes resized Mars images as input.
The model includes four convolution blocks with batch normalization, max pooling,
dropout, and dense classification layers. It outputs one of six landmark
classes.

The notebook includes code to evaluate the custom CNN with accuracy, precision,
recall, F1 score, a classification report, a confusion matrix, and ROC curves.
The checked-in notebook does not store the final custom CNN metric output.

### VGG16 Transfer Learning

The VGG16 model uses ImageNet pretrained weights through TensorFlow and Keras.
The notebook loads VGG16 without its top layer, freezes the copied VGG16 layers,
and adds a six-class softmax classification head.

Saved VGG16 results:

| Metric | Value |
| --- | ---: |
| Weighted precision | 0.8350 |
| Weighted recall | 0.8338 |
| Weighted F1 score | 0.8342 |
| Highest saved validation accuracy in log | 0.9068 |

The saved VGG16 metrics trail the Vision Transformer metrics.

### Vision Transformer

The Vision Transformer uses Hugging Face `google/vit-base-patch16-224` with a
new six-label classifier head. The notebook trains it with PyTorch.

Saved Vision Transformer results:

| Metric | Value |
| --- | ---: |
| Test loss | 0.0919 |
| Test accuracy | 97.07% |
| Weighted precision | 0.9708 |
| Weighted recall | 0.9707 |
| Weighted F1 score | 0.9707 |
| Micro-average AUC | 0.998 |

This model has the strongest saved result in the notebook.

The notebook plots confusion matrices and ROC curves. Running the final Vision
Transformer ROC cell can save `transformer_roc_curves.png`. Running the Vision
Transformer training section can save `best_model.pth`.

## Detailed Results

The notebook reports classification metrics. Higher accuracy, precision, recall,
F1 score, and AUC indicate stronger saved performance.

| Model | Reported Split | Saved Result |
| --- | ---: | --- |
| Custom CNN | Test | Evaluation code only. Final metric output is not saved. |
| VGG16 transfer learning | Test and validation log | Weighted precision `0.8350`, weighted recall `0.8338`, weighted F1 `0.8342`, highest saved validation accuracy `0.9068` |
| Vision Transformer | Test | Test accuracy `97.07%`, weighted precision `0.9708`, weighted recall `0.9707`, weighted F1 `0.9707`, micro-average AUC `0.998` |

The Vision Transformer has the strongest saved result. Its class-level results
stay high across all six modeled classes.

| Class | Precision | Recall | F1 score | Test images |
| --- | ---: | ---: | ---: | ---: |
| `other` | 0.9512 | 0.9543 | 0.9528 | 1,247 |
| `crater` | 0.9711 | 0.9742 | 0.9726 | 1,241 |
| `dark_dune` | 0.9789 | 1.0000 | 0.9893 | 278 |
| `slope_streak` | 0.9822 | 0.9534 | 0.9676 | 580 |
| `bright_dune` | 1.0000 | 0.9953 | 0.9977 | 427 |
| `swiss_cheese` | 0.9800 | 0.9966 | 0.9882 | 295 |

The VGG16 transfer-learning model provides a useful comparison, but its saved
weighted scores are lower than the Vision Transformer results.

The custom CNN cannot be ranked from the checked-in notebook alone. The notebook
contains the evaluation code, but it does not store the final CNN accuracy,
precision, recall, or F1 output.

The notebook does not report RMSE, dollar errors, MAE, R-squared, confidence
intervals, or error by price band. Those metrics apply to regression projects,
not this Mars landmark classification project.

## Limitations

Use this project for learning, model comparison, and portfolio demonstration
only. Do not use it as a validated planetary science classifier or operational
image-labeling system.

The model depends on one processed dataset. It removes two original labels and
uses one train/test split. It may not generalize to new Mars imagery, new camera
conditions, or different preprocessing choices.

## Reproducibility

1. Open `MarsLandmark.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Install the packages imported by the notebook if needed: `opendatasets`,
   `pandas`, `numpy`, `tensorflow`, `opencv-python`, `scikit-learn`, `seaborn`,
   `matplotlib`, `transformers`, `torch`, and `pillow`.
3. Configure Kaggle access for `opendatasets`, or download the Kaggle dataset
   manually.
4. Place the dataset at `./hirise-mars-landmark/` if you download it manually.
5. Run the notebook from top to bottom.

The saved notebook output shows a dataset download size of about `942M`.

Results may change across hardware, TensorFlow or Keras versions, PyTorch
versions, random initialization, and GPU availability.
