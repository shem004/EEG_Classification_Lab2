# EEG Classification with EEGNet and DeepConvNet

This repository implements **EEGNet** and **DeepConvNet** in PyTorch for motor imagery EEG signal classification using the **BCI Competition III – Dataset IIIb**.

## Problem Description

Brain-Computer Interfaces (BCIs) decode neural activity from electroencephalography (EEG) signals into actionable commands. In this project, we classify 2-class motor imagery signals recorded from 2 bipolar EEG channels (`S4b` and `X11b` subjects):

- **Input Shape**: `(Batch, 1, 2, 750)` representing 1 input feature map, 2 EEG channels ($C = 2$), and 750 time points ($T = 750$).
- **Output Shape**: `(Batch, 2)` representing the class logits for the 2 motor imagery classes.

### Models Implemented
1. **EEGNet**: A compact CNN tailored for EEG signals combining:
   - **Temporal Convolution** to learn frequency filters.
   - **Depthwise Convolution** to learn frequency-specific spatial filters across channels.
   - **Separable Convolution** to decouple learning temporal summaries within feature maps from mixing channels via pointwise convolutions.
2. **DeepConvNet**: A deeper convolutional architecture consisting of 4 temporal-spatial convolution blocks with max-pooling and dropout layers.

---

## Project Structure

```text
├── data/                   # Place S4b_train.npz, X11b_train.npz, S4b_test.npz, X11b_test.npz here
├── plots/                  # Generated training/testing accuracy and loss plots
├── weights/                # Saved best model checkpoints (.pt)
├── dataloader.py           # Dataset loading, NaN imputation, and PyTorch Dataset wrapper
├── models.py               # EEGNet and DeepConvNet PyTorch implementations
├── main.py                 # Training, evaluation, plotting, and experiment runner script
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```

---

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone <your-github-repo-url>
   cd <your-repo-folder>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**:
   Create a `data/` folder in the project root and place the four `.npz` dataset files inside:
   - `data/S4b_train.npz`
   - `data/X11b_train.npz`
   - `data/S4b_test.npz`
   - `data/X11b_test.npz`

---

## How to Run

### 1. Run a Single Model (Quick Test)
To train and evaluate a single model configuration (e.g., `EEGNet` or `DeepConvNet` with `elu`, `relu`, or `leaky_relu`):

```bash
python main.py --mode single --model EEGNet --act elu --dropout 0.25 --lr 0.001 --epochs 350
```

### 2. Reproduce All Experiments
To run the complete suite of experiments (comparing activation functions, filter sizes, dropout rates, DeepConvNet variations, and the Top-3 ensemble):

```bash
python main.py --mode all
```

- Best model checkpoints are automatically saved to `./weights/`.
- Accuracy and loss curves are saved as `.png` images in `./plots/`.
- A full summary table of all experiments is printed to the console and saved to `experiment_summary.csv`.