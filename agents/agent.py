from dataclasses import dataclass
import sys
from typing import List

import numpy as np
import torch

from dataset.dataset_flat import DatasetFlatAugmented, DatasetFlat
from dataset.dataset_sequential import DatasetSequential, DatasetSequentialAugmented
from dataset.model_data import ModelData
from loaders.pred_prices_provider import PredPricesProvider
from loaders.real_prices_provider import RealPricesProvider
from timeseries.TimeseriesInterval import TimeseriesInterval
from models.LSTM_MC import Compound_LSTM


@dataclass
class AgentConfig:
    #model parameters
    device: str = "cpu"
    learning_rate: float = 0.0003
    batch_size: int = 15
    generation: int = 75
    seq_len: int = 20
    hidden_size: int = 160
    
    #test and train parameters
    dtype = torch.float32
    batch_first: bool = True
    out_size: int = 1
    test_step: int = 7
    freq_rate_train: int = 10
    freq_rate_test: int = 10

class Agent:
    def __init__(self, config: AgentConfig, real_prices_provider: RealPricesProvider, pred_prices_provider: PredPricesProvider, symbols_amnt):
        self.config = config
        self.real_prices_provider = real_prices_provider
        self.pred_prices_provider = pred_prices_provider

        # batch_size, input_features, batch_first, out_size, device, dtype, hidden_size, layer_amnt, dropout
        self.model = Compound_LSTM(self.config.batch_size, symbols_amnt, self.config.batch_first,
                                   self.config.out_size, self.config.device, self.config.dtype, self.config.hidden_size)


    def train(self, interval: TimeseriesInterval, symbols: List[str]):
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr = self.config.learning_rate)
        criterion = torch.nn.BCEWithLogitsLoss()
        model.train()

        dataset = self.prepare_dataset(symbols, interval, augmentation = True)

        episode = 0
        error_rtt = []
        batch_rtt = []
        total_batches = 0
        while episode < self.config.generation:
            idx = 0
            stop = False
            
            
            episode_error = 0 #error per episode
            num_batches = 0 #number of batches per episode
            while not stop:
                batch_x = []
                batch_y = []

                for i in range(self.config.batch_size):
                    start_pos = idx * self.config.batch_size + i
                    if start_pos >= len(dataset):
                        stop = True
                        break

                    x_seq, y = dataset[start_pos]  #returns a tensor x_seq of size (seq_number, feature_amnt) and a tensor y with real values
                    if x_seq.size(0) < self.config.seq_len: #since we have less than needed number of records in a sequence
                        stop = True
                        break
                    
                    batch_x.append(x_seq)
                    batch_y.append(y)

                if len(batch_x) == 0:
                    break

                batch_x = torch.stack(batch_x, dim = 0)  #shape: (batch, seq_len, feature_dim) basically stack up one after another tables of size (seq_len, feature_dim)

                batch_y = torch.tensor(batch_y, dtype = self.config.dtype)
                batch_y = batch_y.unsqueeze(1)

                batch_x = batch_x.to(self.config.device)
                batch_y = batch_y.to(self.config.device)

                optimizer.zero_grad()
                outputs = model(batch_x) #call model, not forward directly
                loss = criterion(outputs, batch_y) #compute loss
                episode_error += loss.item()
                loss.backward()
                optimizer.step()

                num_batches += 1
                idx += 1
            total_batches += num_batches
            
            avg_loss = episode_error / num_batches if num_batches > 0 else float('nan')
            error_rtt.append(avg_loss)
            batch_rtt.append(total_batches)
            
            episode += 1

        self.test(interval, symbols, True)
        return error_rtt, batch_rtt






    def test(self, interval: TimeseriesInterval, symbols: List[str], train = False):
        TP, TN, FP, FN = 0, 0, 0, 0
        criterion = torch.nn.BCEWithLogitsLoss()
        error_sequence = []
        frequen_rate = self.config.freq_rate_test

        model = self.model
        model.eval()
        dataset = self.prepare_dataset(symbols, interval, augmentation = False)
        tstp_amnt = interval.get_steps_cnt()
        seq_len = self.config.seq_len
        device = self.config.device
        
        seq_ind = 0 #represents sequence_window index as well as number of predictions
        with torch.no_grad():
            while True:
                if(seq_ind + seq_len >= tstp_amnt): break
                x_seq, y = dataset[seq_ind]

                x_seq = x_seq.to(device).float()
                x_seq = x_seq.to(device).unsqueeze(0)

                pred_y = model(x_seq) #tensor of 1x1
                real_y = y.to(device).float().view(1, 1) # so now it also has size of 1x1
                
                if((seq_ind + 1) % frequen_rate == 0):
                    error = criterion(pred_y, real_y)
                    error_sequence.append(error.item())
                    
                
                pred_y = torch.sigmoid(pred_y)
                pred_y = pred_y.item() #because pred_y previosuly was a tensor

                pred_y = 1 if pred_y > 0.5 else 0 #now it's 0 or 1 depending on its value



                if(pred_y == real_y.item()):
                    if(real_y.item()):
                        TP += 1
                    else:
                        TN += 1
                else:
                    if(real_y.item()):
                        FN += 1
                    else:
                        FP += 1

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                seq_ind += 1

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_score *= 100
        precision *= 100
        recall *= 100
        if(not train):
            print("Test scores:")
            print(f"F1_score: {f1_score:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"Precision: {precision:.2f}")
            print("\n")
        else:
            print("Train scores:\n")
            print(f"F1_score: {f1_score:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"Precision: {precision:.2f}")
            print("\n")
            
        return error_sequence, frequen_rate



    def prepare_dataset(self, symbols: List[str], timeseries_interval: TimeseriesInterval, augmentation=False):
        model_data = self.__get_model_state(
            timeseries_interval,
            symbols=symbols
        )

        # choose between sequential and flat representation
        if augmentation:
            # dataset = DatasetFlatAugmented(model_data, seq_len=self.config.seq_len)
            dataset = DatasetSequentialAugmented(model_data, seq_len=self.config.seq_len)

        else:
            # dataset = DatasetFlat(model_data, seq_len=self.config.seq_len)
            dataset = DatasetSequential(model_data, seq_len=self.config.seq_len)

        return dataset

    # DO NOT MODIFY!
    def __get_model_state(self, timeseries_interval: TimeseriesInterval, symbols: List[str], hist_items_cnt: int = 1):
        symbols_cnt = len(symbols)
        timeseries_cnt = timeseries_interval.get_steps_cnt()
        models_defs = self.pred_prices_provider.get_models_defs()

        seq_features = []

        open_price = np.ndarray((symbols_cnt, timeseries_cnt), dtype=np.float32)
        close_price = np.ndarray((symbols_cnt, timeseries_cnt), dtype=np.float32)

        for symbol_idx, symbol in enumerate(symbols):
            real_prices_hist = self.real_prices_provider.get_prices_np(symbol=symbol, date=timeseries_interval.get_date_to(), hist_cnt=timeseries_cnt + 2)
            real_return_hist = (real_prices_hist[1:] - real_prices_hist[:-1]) / real_prices_hist[:-1]

            for i in range(timeseries_cnt):
                real_return = [real_return_hist[i]]
                close_price[symbol_idx][i] = real_prices_hist[i + 2]
                open_price[symbol_idx][i] = real_prices_hist[i + 1]

                # --- Predictions ---
                date_to = timeseries_interval.get_next_timeseries_date(i).get_date()
                pred_features = []
                for model_idx, model_id in enumerate(models_defs):
                    pred_prices_hist = self.pred_prices_provider.get_prices_np(models_defs[model_id], symbol, date=date_to, hist_cnt=hist_items_cnt + 1)
                    pred_return_hist = (pred_prices_hist[1:] - pred_prices_hist[:-1]) / pred_prices_hist[:-1]
                    pred_features.extend(pred_return_hist.tolist())

                # Combine features: [real_return_hist + all_pred_returns]
                day_features = np.concatenate([real_return, pred_features])
                seq_features.append(day_features)

        seq_features = np.array(seq_features, dtype=np.float32)  # shape: (days_cnt, feature_dim)

        return ModelData(
            days_cnt=timeseries_cnt,
            symbols_cnt=symbols_cnt,
            state=seq_features,
            open_price=open_price.flatten(),
            close_price=close_price.flatten(),
        )
