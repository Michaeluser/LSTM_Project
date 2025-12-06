import numpy as np
import torch

class LSTM_Classifier:
    @staticmethod
    def sigmoid(input):
        
        return 1 / (1 + np.exp(-input))
    
    def forget_gate(self, x_curr):
        conc_mtr = np.concatenate((self.STM_hs, x_curr), axis = 1)
        f1 = conc_mtr @ self.W_forg.T + self.b_forg
        f1 = LSTM_Classifier.sigmoid(f1)
        self.LTM_cs *= f1
    
    
    def input_gate(self, x_curr):
        conc_mtr = np.concatenate((self.STM_hs, x_curr), axis = 1)
        f2 = conc_mtr @ self.W_inpg.T + self.b_inpg
        f2 = LSTM_Classifier.sigmoid(f2)
        
        f3 = conc_mtr @ self.W_cndg.T + self.b_cndg
        f3 = np.tanh(f3)
        
        self.LTM_cs += f2 * f3
    
    
    def output_gate(self, x_curr):
        conc_mtr = np.concatenate((self.STM_hs, x_curr), axis = 1)
        f4 = conc_mtr @ self.W_outg.T + self.b_outg
        f4 = LSTM_Classifier.sigmoid(f4)
        
        upd_hidd = np.tanh(self.LTM_cs)
        upd_hidd *= f4
        self.STM_hs = upd_hidd
        
        
    def reset_states(self, current_batch_size: int):
        H = self.hidden_state_size
        if current_batch_size < self.max_batch_size:
            
            # We slice the first dimension (Batch) to match the incoming batch size.
            # This creates a *view* of the NumPy array with the correct dimensions.
            self.LTM_cs = self.LTM_cs[:current_batch_size, :]
            self.STM_hs = self.STM_hs[:current_batch_size, :]
            
            # IMPORTANT: Now, the max_batch_size must be updated to the current size 
            # so that subsequent batches check against the correct dimension.
            # However, to avoid permanent state mutation in the class instance across
            # different batches, the standard practice is to use local variables
            # for slicing or ensure the states are always max size and only *used*
            # up to current_batch_size.
            
            # --- Simpler, safer approach without permanent self-resizing ---
            # The LSTM must be designed to *always* take the input x_curr size
            # and match the state size implicitly via slicing/viewing, not
            # by re-assigning the state variables permanently.
            
            # If you must re-assign the class variable, ensure you initialize
            # it back to the max size after training!
            
            # --- Recommended Alternative (Using the correct slice in forward) ---
            # Instead of permanently re-assigning self.LTM_cs, we should ensure the
            # LTM/STM variables in the gates are local slices of the max array. 
            # But sticking to your current structure:
            
            # Re-initialize the states to the correct smaller size (if needed)
            self.LTM_cs = np.zeros((current_batch_size, H), dtype = np.float32)
            self.STM_hs = np.zeros((current_batch_size, H), dtype = np.float32)

        # --- 2. Zero-out ---
        # If current_batch_size == max_batch_size, they are already the correct size, 
        # but we still need to reset them to zero to clear previous memory.
        else:
            self.LTM_cs[:] = 0.0
            self.STM_hs[:] = 0.0
        
    def classify(self):
        """Computes the final binary prediction using the last hidden state (STM_hs)."""
        # The result of the final hidden state (B, H) multiplied by W_out (H, 1)
        # gives a raw logit (B, 1)
        logits = self.STM_hs @ self.W_out + self.b_out

        prediction = LSTM_Classifier.sigmoid(logits)
        
        return prediction
    
    def update_weights(self):
        pass
    
    def __init__(self, batch_size, sequence_length, input_features, hidden_state_size):
        self.LTM_cs = np.zeros((batch_size, hidden_state_size), dtype = np.float32) #Long Time Memory cell state
        self.STM_hs = np.zeros((batch_size, hidden_state_size), dtype = np.float32) #Short Time Memory hidden state
        self.batch_size = batch_size # batch_size represents number of batches that have a size of sequence length,
        #input_features is number of features in each sequence's subsample, and there's a static number of 
        
        self.input_features = input_features
        self.sequence_length = sequence_length
        
        self.W_forg = np.zeros((hidden_state_size, hidden_state_size + input_features))
        self.b_forg = np.zeros((1, hidden_state_size), dtype = np.float32)
        
        self.W_inpg = np.zeros((hidden_state_size, hidden_state_size + input_features))
        self.b_inpg = np.zeros((1, hidden_state_size), dtype = np.float32)
        
        self.W_cndg = np.zeros((hidden_state_size, hidden_state_size + input_features))
        self.b_cndg = np.zeros((1, hidden_state_size), dtype = np.float32)
        
        self.W_outg = np.zeros((hidden_state_size, hidden_state_size + input_features))
        self.b_outg = np.zeros((1, hidden_state_size), dtype = np.float32)
        
        self.W_out = np.random.rand(hidden_state_size, 1).astype(np.float32)
        self.b_out = np.zeros((1,1), dtype = np.float32)
        
    def forward(self, batch): #batch consists of an array of (tensor, y) structures, each structure has the same size, so size of tensor corresponds to the size of y
        #run through every sub_batch individually in a batch to train the model
        #tensor has a size (NxK) where N is number of rows and K is number of features
        batch_x_list = [item[0] for item in batch]
        batch_y_list = [item[1] for item in batch]
        
        batch_tensor_3d = torch.stack(batch_x_list, dim=0) # Shape: (B, L, I)
        
        # Determine the actual batch size
        current_batch_size = batch_tensor_3d.shape[0]

        # 3. Handle State Resizing/Reset (Crucial for the last batch)
        #    You need to implement this method based on our previous discussion:
        self.reset_states(current_batch_size) 
        batch_np = batch_tensor_3d.detach().cpu().numpy()

        # 5. Loop over the Sequence (Time Steps)
        #    batch_np.shape[1] is the Sequence Length (L)
        for t in range(self.sequence_length):
            x_curr = batch_np[:, t, :] 
            
            # Process the entire batch in parallel through the gates
            self.forget_gate(x_curr)
            self.input_gate(x_curr)
            self.output_gate(x_curr)
            
        final_prediction = self.classify()
        
        # After computing loss and gradients (BPTT), you would call:
        # self.update_weights()
        
        return final_prediction