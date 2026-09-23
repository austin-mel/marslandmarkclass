# Mars Landmark Classification

Mars Landmark Classification trains image models to identify Mars surface
features from satellite image patches. The notebook uses a Kaggle HiRISE Mars
landmark dataset, prepares the images, trains three model types, and compares
the saved results.

The notebook identifies this work as a CSC 180 Intelligent Systems final
project at California State University, Sacramento.

## What The Project Does

`MarsLandmark.ipynb` classifies each Mars image patch into one terrain label.
The modeled labels are:

- `other`
- `crater`
- `dark_dune`
- `slope_streak`
- `bright_dune`
- `swiss_cheese`

The notebook trains and evaluates:

| Model | Purpose |
| --- | --- |
| Custom CNN | Builds a task-specific convolutional neural network in TensorFlow/Keras. |
| VGG16 transfer learning | Adapts a pretrained VGG16 image model to the Mars classes. |
| Vision Transformer | Adapts Hugging Face `google/vit-base-patch16-224` to the Mars classes. |

## Results

The Vision Transformer produced the best saved result in the notebook. It
classified 97.07% of the test images correctly. In practical terms, it labeled
about 97 out of every 100 held-out Mars image patches correctly in this run.

The VGG16 transfer learning model also learned useful patterns, but its saved
weighted scores were lower. It reached 0.8350 weighted precision, 0.8338
weighted recall, and 0.8342 weighted F1 score.

The confusion matrix shows where the model made correct and incorrect
predictions. Mistakes appear
outside the diagonal and show which landmark types the model confused.

These results show that the Vision Transformer can separate the six modeled Mars
landmark classes well on the saved test split.

## Dataset

The notebook downloads the dataset from Kaggle:

`https://www.kaggle.com/datasets/glorianikitchuk/hirise-mars-landmark`

It reads labels from:

`hirise-mars-landmark/labels-map-proj-v3.txt`

Dataset sizes from the notebook:

| Stage | Count | Labels |
| --- | ---: | --- |
| Original label file | 73,031 records | 8 labels |
| Modeled dataset | 16,270 images | 6 labels |
| Training split | 12,202 images | 6 labels |
| Test split | 4,068 images | 6 labels |

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

Modeled class counts after preprocessing:

| Class | Count |
| --- | ---: |
| `other` | 5,000 |
| `crater` | 4,900 |
| `dark_dune` | 1,141 |
| `slope_streak` | 2,331 |
| `bright_dune` | 1,750 |
| `swiss_cheese` | 1,148 |

## Preprocessing

The notebook performs these steps:

1. Loads the label file into a pandas DataFrame.
2. Reduces the `other` class to 5,000 samples.
3. Removes the `impact_ejecta` and `spider` classes.
4. Remaps the remaining labels into six classes.
5. Loads images with OpenCV.
6. Resizes images to `128 x 128 x 3`.
7. One-hot encodes labels for TensorFlow/Keras.
8. Splits the data with `test_size=0.25` and `random_state=42`.

The Vision Transformer section prints an extra balancing step that selects 2,000
training samples each from `other` and `crater`. The following DataLoader cell
uses the regular training arrays. Review that section before changing the ViT
experiment.

## Models

### Custom CNN

The custom CNN uses TensorFlow/Keras. It includes four convolution blocks, batch
normalization, max pooling, dropout, and dense classification layers. The
notebook includes code to evaluate accuracy, precision, recall, F1 score, a
classification report, a confusion matrix, and ROC curves. The checked-in
notebook does not store the final CNN metric output.

### VGG16 Transfer Learning

The VGG16 model uses ImageNet pretrained weights. The notebook freezes the VGG16
layers and adds a six-class softmax classification head.

Saved VGG16 results:

| Metric | Value |
| --- | ---: |
| Weighted precision | 0.8350 |
| Weighted recall | 0.8338 |
| Weighted F1 score | 0.8342 |
| Highest saved validation accuracy in log | 0.9068 |

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

Class-level Vision Transformer results:

| Class | Precision | Recall | F1 score | Test images |
| --- | ---: | ---: | ---: | ---: |
| `other` | 0.9512 | 0.9543 | 0.9528 | 1,247 |
| `crater` | 0.9711 | 0.9742 | 0.9726 | 1,241 |
| `dark_dune` | 0.9789 | 1.0000 | 0.9893 | 278 |
| `slope_streak` | 0.9822 | 0.9534 | 0.9676 | 580 |
| `bright_dune` | 1.0000 | 0.9953 | 0.9977 | 427 |
| `swiss_cheese` | 0.9800 | 0.9966 | 0.9882 | 295 |

The Vision Transformer produces the strongest saved result in the notebook. A
97.07% test accuracy means it classified about 97 out of 100 test images
correctly.

## Metric Definitions

- Accuracy measures the share of test images classified correctly.
- Precision measures how often a predicted class is correct.
- Recall measures how many real examples of a class the model finds.
- F1 score combines precision and recall into one score.
- A confusion matrix shows correct predictions on the diagonal and mistakes
  outside the diagonal.
- AUC measures how well the model ranks the correct class above incorrect
  classes.

## Limitations

- The original dataset has strong class imbalance. `other` has 61,054 records.
  `impact_ejecta` has 231 records. `spider` has 476 records.
- The modeled dataset removes `impact_ejecta` and `spider`.
- The six modeled classes still have uneven sample counts.
- Resizing images to `128 x 128` can reduce surface detail.
- The notebook does not document a separate image-quality filtering step.
- The CNN and VGG16 cells use the test split as validation data.
- The notebook uses one train/test split and does not use cross-validation.
- Some filenames indicate rotated, flipped, or brightness-adjusted variants.
- The project fits a course or portfolio scope. It does not validate a
  production planetary science system.

## Run The Notebook

Use a Python environment with Jupyter or Google Colab. A GPU helps with VGG16
and Vision Transformer training.

1. Clone or download this repository.
2. Open `MarsLandmark.ipynb`.
3. Install the required packages:

   ```bash
   pip install opendatasets pandas numpy tensorflow opencv-python scikit-learn seaborn matplotlib transformers torch pillow
   ```

4. Configure Kaggle access for `opendatasets`, or download the Kaggle dataset
   manually.
5. Place the dataset at `./hirise-mars-landmark/` if you download it manually.
6. Run the notebook cells from top to bottom.

The saved notebook output shows a dataset download size of about `942M`.
