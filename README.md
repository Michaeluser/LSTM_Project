# LSTM Cryptocurrency Trading Agent

> Forked from [Iskandor/UI-zadanie3](https://github.com/Iskandor/UI-zadanie3). The project skeleton was provided by the original repository; the model architecture, agent logic, dataset classes, and training pipeline were added by the author.

A PyTorch-based trading agent that uses a stacked **LSTM** network to predict whether the next-day price of a cryptocurrency will go up (long) or down (short). The model is trained on a combination of real historical prices and predictions from multiple external forecasting models (Moirai, Chronos, TiRex, Sundial).

## How It Works

Each training sample is a sequence of daily feature vectors, where each vector contains the real return for that day alongside the predicted returns from each external model for the same timestamp. The label is a binary long/short signal derived from whether the close price exceeded the open price.

The model is a 3-layer stacked LSTM followed by a linear output layer, trained with BCE loss and Adam optimizer. Sigmoid activation is applied at inference time only, and predictions are thresholded at 0.5 to produce the final long/short decision. Performance is evaluated using F1 score, precision, and recall.

## Project Structure

```
main.py                          # Entry point — configures and runs train/test
agents/agent.py                  # Agent class: training loop, evaluation, dataset prep
models/LSTM_MC.py                # 3-layer stacked LSTM model definition
dataset/dataset_sequential.py    # Sequential dataset with optional augmentation
dataset/dataset_flat.py          # Flat (flattened sequence) dataset variant
dataset/model_data.py            # ModelData dataclass
loaders/                         # CSV loaders and providers for real and predicted prices
timeseries/                      # TimeseriesDate and TimeseriesInterval utilities
model_train.ipynb                # Training notebook
requirements.txt
```

## Data

Place the following files in a `data/` directory:

- `prices_updated.csv` — real historical prices (symbol, date, open, close, volume, …)
- `predictions.csv` — predicted prices from external models (model, ctx, symbol, date, predicted)

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

Configure symbols, date intervals, and model hyperparameters directly in `main.py` and `agents/agent.py` (`AgentConfig`).
