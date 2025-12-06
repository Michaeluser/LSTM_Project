import torch.nn as nn

class Compound_LSTM(nn.Module):
    def __init__(self, batch_size, input_features, batch_first, out_size, device, dtype, hidden_size):
        super().__init__()
        self.seq_amnt = batch_size
        self.feature_amnt = input_features
        self.h1_s = hidden_size
        self.h2_s = hidden_size
        self.h3_s = hidden_size

        self.lstm1 = nn.LSTM(self.feature_amnt, self.h1_s, batch_first = batch_first, device = device, dtype = dtype)
        self.lstm2 = nn.LSTM(self.h1_s, self.h2_s, batch_first = batch_first, device = device, dtype = dtype)
        self.lstm3 = nn.LSTM(self.h2_s, self.h3_s, batch_first = batch_first, device = device, dtype = dtype)

        self.fc = nn.Linear(self.h3_s, out_size, device = device)

    def forward(self, seq_batch):
        out1, hc = self.lstm1(seq_batch)
        
        out2, hc = self.lstm2(out1)
        
        out3, hc = self.lstm3(out2)
        out3 = self.fc(out3[:, -1, :]) # since we need a single last output

        return out3
