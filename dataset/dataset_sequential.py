import torch
from torch.utils.data import Dataset

class DatasetSequential(Dataset):
    """
    Dataset for supervised training of the trading agent using sequential data.
    Each sample is a sequence of features up to timeseries t, with label y at timeseries t.
    """

    def __init__(self, model_data, seq_len: int = 14):
        """
        Args:
            model_data: ModelData object (contains state, flips, open/close prices).
            seq_len: Length of the sequence window for input.
        """
        self.state = torch.tensor(model_data.state, dtype=torch.float32)  # shape: (days_cnt, feature_dim)
        open_price = torch.tensor(model_data.open_price, dtype=torch.float32)
        close_price = torch.tensor(model_data.close_price, dtype=torch.float32)
        self.target = (close_price < open_price).long() # 0=long, 1=short

        self.seq_len = seq_len
        self.days_cnt = model_data.days_cnt

    def __len__(self):
        return self.days_cnt - self.seq_len

    def __getitem__(self, idx):
        # sequence from idx → idx+seq_len
        x_seq = self.state[idx : idx + self.seq_len]  # (seq_len, feature_dim) and starting from index ending at index + sequence length

        # target action = long/short at the last day of the sequence
        last_idx = idx + self.seq_len
        y = self.target[last_idx]

        return (
            x_seq,  # (seq_len, feature_dim)
            y
        )

class DatasetSequentialAugmented(DatasetSequential):
    def __init__(self, model_data, seq_len: int = 14):
        super().__init__(model_data, seq_len)
        self.aug_window = seq_len * 2

    def __len__(self):
        return self.days_cnt - self.aug_window

    def __getitem__(self, idx):
        # Choose window (larger if augmentation enabled)
        window = self.state[idx : idx + self.aug_window]

        # Ensure last timestep is always included
        last_idx_in_window = self.aug_window - 1
        # Randomly choose seq_len-1 indices from earlier part
        rand_idx = torch.randperm(self.aug_window - 1)[: self.seq_len - 1]
        selected_idx = torch.cat([
            torch.sort(rand_idx)[0],
            torch.tensor([last_idx_in_window])
        ])
        x_seq = window[selected_idx]

        last_idx = idx + self.aug_window
        y = self.target[last_idx]

        return x_seq, y
