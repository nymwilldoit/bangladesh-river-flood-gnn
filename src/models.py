"""
Bangladesh River Flood Forecasting - Model Architectures
Spatio-Temporal Graph Neural Network implementations

Extracted from Phase 3: Section 9.1 (DCRNN) and Section 9.2 (GraphConvLSTM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DCGRUCell(nn.Module):
    """Diffusion Convolutional Gated Recurrent Unit Cell"""
    
    def __init__(self, input_dim, hidden_dim, max_diffusion_step, num_nodes, bias=True):
        super(DCGRUCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_diffusion_step = max_diffusion_step
        self.num_nodes = num_nodes
        self.bias = bias
        
        # Gates: reset and update
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        self.candidate_gate = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
    
    def forward(self, inputs, hidden_state, adj_matrix):
        """
        Args:
            inputs: (batch_size, num_nodes, input_dim)
            hidden_state: (batch_size, num_nodes, hidden_dim)
            adj_matrix: (num_nodes, num_nodes)
        """
        batch_size, num_nodes = inputs.shape[:2]
        
        # Diffusion convolution for inputs and hidden states
        input_and_state = torch.cat([inputs, hidden_state], dim=-1)
        
        # Reset gate
        reset_input = torch.sigmoid(self.reset_gate(input_and_state))
        
        # Update gate
        update_input = torch.sigmoid(self.update_gate(input_and_state))
        
        # Candidate hidden state
        candidate_input = torch.cat([inputs, reset_input * hidden_state], dim=-1)
        candidate_state = torch.tanh(self.candidate_gate(candidate_input))
        
        # Update hidden state
        new_hidden_state = update_input * hidden_state + (1 - update_input) * candidate_state
        
        return new_hidden_state


class DCRNN(nn.Module):
    """Diffusion Convolutional Recurrent Neural Network"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, num_layers=2, 
                 max_diffusion_step=2, dropout=0.3):
        super(DCRNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.max_diffusion_step = max_diffusion_step
        
        # DCGRU layers
        self.dcgru_layers = nn.ModuleList([
            DCGRUCell(input_dim if i == 0 else hidden_dim, 
                     hidden_dim, max_diffusion_step, num_nodes)
            for i in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, inputs, adj_matrix):
        """
        Args:
            inputs: (batch_size, seq_length, num_nodes, input_dim)
            adj_matrix: (num_nodes, num_nodes)
        """
        batch_size, seq_length, num_nodes, _ = inputs.shape
        
        # Initialize hidden states
        hidden_states = [torch.zeros(batch_size, num_nodes, self.hidden_dim, 
                                   device=inputs.device) for _ in range(self.num_layers)]
        
        outputs = []
        
        # Process each time step
        for t in range(seq_length):
            current_input = inputs[:, t, :, :]  # (batch_size, num_nodes, input_dim)
            
            # Pass through DCGRU layers
            for layer_idx, dcgru_layer in enumerate(self.dcgru_layers):
                if layer_idx == 0:
                    layer_input = current_input
                else:
                    layer_input = self.dropout(hidden_states[layer_idx-1])
                
                hidden_states[layer_idx] = dcgru_layer(
                    layer_input, hidden_states[layer_idx], adj_matrix
                )
            
            # Store output from last layer
            outputs.append(hidden_states[-1])
        
        # Use last time step output for prediction
        final_output = outputs[-1]  # (batch_size, num_nodes, hidden_dim)
        
        # Project to output dimension
        predictions = self.output_projection(final_output)  # (batch_size, num_nodes, output_dim)
        
        return predictions


class GraphConvLSTMCell(nn.Module):
    """Graph Convolutional LSTM Cell"""
    
    def __init__(self, input_dim, hidden_dim, num_nodes, bias=True):
        super(GraphConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.bias = bias
        
        # LSTM gates
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        self.candidate_gate = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim, bias=bias)
        
        # Graph convolution weights
        self.graph_conv_weight = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.graph_conv_weight)
    
    def graph_convolution(self, x, adj_matrix):
        """Apply graph convolution operation"""
        # Normalize adjacency matrix
        degree = torch.sum(adj_matrix, dim=1, keepdim=True)
        degree_inv_sqrt = torch.pow(degree + 1e-6, -0.5)
        norm_adj = degree_inv_sqrt * adj_matrix * degree_inv_sqrt.T
        
        # Apply convolution: AXW
        return torch.matmul(torch.matmul(norm_adj, x), self.graph_conv_weight)
    
    def forward(self, inputs, hidden_state, cell_state, adj_matrix):
        """
        Args:
            inputs: (batch_size, num_nodes, input_dim)
            hidden_state: (batch_size, num_nodes, hidden_dim)
            cell_state: (batch_size, num_nodes, hidden_dim)
            adj_matrix: (num_nodes, num_nodes)
        """
        # Combine input and hidden state
        combined = torch.cat([inputs, hidden_state], dim=-1)
        
        # LSTM gates
        forget = torch.sigmoid(self.forget_gate(combined))
        input_gate = torch.sigmoid(self.input_gate(combined))
        candidate = torch.tanh(self.candidate_gate(combined))
        output = torch.sigmoid(self.output_gate(combined))
        
        # Apply graph convolution to hidden state
        graph_hidden = self.graph_convolution(hidden_state, adj_matrix)
        
        # Update cell state
        new_cell_state = forget * cell_state + input_gate * candidate
        
        # Apply graph convolution to cell state
        graph_cell = self.graph_convolution(new_cell_state, adj_matrix)
        
        # Update hidden state
        new_hidden_state = output * torch.tanh(graph_cell)
        
        return new_hidden_state, new_cell_state


class GraphConvLSTM(nn.Module):
    """Graph Convolutional LSTM Network"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_nodes, num_layers=2, dropout=0.3):
        super(GraphConvLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        
        # GraphConvLSTM layers
        self.gclstm_layers = nn.ModuleList([
            GraphConvLSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim, num_nodes)
            for i in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, inputs, adj_matrix):
        """
        Args:
            inputs: (batch_size, seq_length, num_nodes, input_dim)
            adj_matrix: (num_nodes, num_nodes)
        """
        batch_size, seq_length, num_nodes, _ = inputs.shape
        
        # Initialize hidden and cell states
        hidden_states = [torch.zeros(batch_size, num_nodes, self.hidden_dim, 
                                   device=inputs.device) for _ in range(self.num_layers)]
        cell_states = [torch.zeros(batch_size, num_nodes, self.hidden_dim, 
                                 device=inputs.device) for _ in range(self.num_layers)]
        
        outputs = []
        
        # Process each time step
        for t in range(seq_length):
            current_input = inputs[:, t, :, :]
            
            # Pass through GraphConvLSTM layers
            for layer_idx, gclstm_layer in enumerate(self.gclstm_layers):
                if layer_idx == 0:
                    layer_input = current_input
                else:
                    layer_input = self.dropout(hidden_states[layer_idx-1])
                
                hidden_states[layer_idx], cell_states[layer_idx] = gclstm_layer(
                    layer_input, hidden_states[layer_idx], cell_states[layer_idx], adj_matrix
                )
            
            outputs.append(hidden_states[-1])
        
        # Use last time step output
        final_output = outputs[-1]
        
        # Project to output dimension
        predictions = self.output_projection(final_output)
        
        return predictions


class RiverFloodLoss(nn.Module):
    """Custom loss function for river flood forecasting"""
    
    def __init__(self, alpha=0.7, beta=0.3):
        super(RiverFloodLoss, self).__init__()
        self.alpha = alpha  # Weight for MSE
        self.beta = beta    # Weight for MAE
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, num_nodes, 1)
            targets: (batch_size, num_nodes)
        """
        predictions = predictions.squeeze(-1)  # Remove last dimension
        
        # Combined loss
        mse_loss = self.mse(predictions, targets)
        mae_loss = self.mae(predictions, targets)
        
        total_loss = self.alpha * mse_loss + self.beta * mae_loss
        
        return total_loss, mse_loss, mae_loss
