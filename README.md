# 🌊 Bangladesh River Flood Forecasting with Graph Neural Networks

## Project Overview
This project implements Spatio-Temporal Graph Neural Networks (STGNNs) to forecast river flood levels across Bangladesh's complex river network system. The models capture both temporal patterns and spatial relationships between gauge stations for improved flood prediction accuracy.

## 🎯 Key Features
- **Graph-based River Network**: Models Bangladesh's major rivers (Brahmaputra, Ganges, Meghna) as interconnected nodes
- **Dual STGNN Architecture**: Implements both DCRNN and GraphConvLSTM models
- **Seasonal Pattern Recognition**: Captures monsoon-driven flood dynamics
- **Multi-station Forecasting**: Predicts water levels across multiple gauge stations simultaneously

## 🏗️ Model Architecture

### 1. DCRNN (Diffusion Convolutional RNN)
- **Core Innovation**: Combines graph convolution with recurrent neural networks
- **Spatial Modeling**: Captures diffusion of flood waves through river network
- **Temporal Modeling**: Handles sequential dependencies in water level time series
- **Key Components**: DCGRUCell with reset, update, and candidate gates

### 2. GraphConvLSTM 
- **Core Innovation**: Graph-aware LSTM cells with spatial message passing
- **Memory Management**: Maintains long-term temporal memory with spatial awareness
- **Graph Operations**: Applies convolution operations on graph-structured data
- **Key Components**: GraphConvLSTMCell with forget, input, output gates

## 📁 Project Structure
bangladesh-river-flood-gnn/
├── src/ # Core implementation modules
│ ├── models.py # DCRNN and GraphConvLSTM architectures
│ ├── data_utils.py # Data generation and preprocessing
│ ├── graph_utils.py # River network graph construction
│ └── training_utils.py # Training and evaluation utilities
├── notebooks/ # Development notebooks by phase
│ ├── phase1_setup_and_graph.ipynb
│ ├── phase2_data_preprocessing.ipynb
│ └── phase3_model_training.ipynb
├── app/ # Streamlit web application
├── models/ # Trained model weights and checkpoints
├── data/ # Sample datasets and processed data
├── docs/ # Additional documentation
├── requirements.txt # Python package dependencies
└── README.md # This file

text

## 🚀 Quick Start

### Installation
Clone the repository
git clone https://github.com/nymwilldoit/bangladesh-river-flood-gnn.git
cd bangladesh-river-flood-gnn

Install dependencies
pip install -r requirements.txt

text

### Basic Usage
from src.models import DCRNN, GraphConvLSTM, RiverFloodLoss
from src.data_utils import create_bangladesh_gauge_network
from src.graph_utils import create_river_network_graph, create_adjacency_matrix

Create Bangladesh gauge station network
gauge_df = create_bangladesh_gauge_network()
print(f"Created network with {len(gauge_df)} gauge stations")

Build river network graph
river_graph = create_river_network_graph(gauge_df)
adj_matrix = create_adjacency_matrix(river_graph, len(gauge_df))

Initialize models
input_dim = 10 # Number of features per node
hidden_dim = 64
output_dim = 1 # Predicting water level
num_nodes = len(gauge_df)

DCRNN model
dcrnn_model = DCRNN(
input_dim=input_dim,
hidden_dim=hidden_dim,
output_dim=output_dim,
num_nodes=num_nodes,
num_layers=2,
dropout=0.3
)

GraphConvLSTM model
gclstm_model = GraphConvLSTM(
input_dim=input_dim,
hidden_dim=hidden_dim,
output_dim=output_dim,
num_nodes=num_nodes,
num_layers=2,
dropout=0.3
)

Custom loss function
criterion = RiverFloodLoss(alpha=0.7, beta=0.3)

text

### Training Example
from src.training_utils import train_model
import torch

Setup training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.Adam(dcrnn_model.parameters(), lr=0.001)

Train model (assuming you have train_loader, val_loader)
train_losses, val_losses = train_model(
model=dcrnn_model,
train_loader=train_loader,
val_loader=val_loader,
adj_matrix=adj_matrix,
criterion=criterion,
optimizer=optimizer,
num_epochs=100,
device=device,
model_name="DCRNN"
)

text

## 🌍 Bangladesh River Network

### Gauge Stations
- **Brahmaputra River**: Chilmari, Bahadurabad, Sirajganj
- **Ganges River**: Rajshahi, Hardinge Bridge, Goalundo  
- **Meghna River**: Bhairab Bazar, Chandpur

### River System Characteristics
- **Monsoon-driven patterns**: Peak flooding June-September
- **Confluence dynamics**: Complex interactions between major rivers
- **Seasonal variations**: Distinct pre-monsoon, monsoon, post-monsoon, and winter periods

## 🔬 Research Context

### Domain
Flood forecasting for Bangladesh river systems with focus on improving early warning capabilities for monsoon and cyclone events.

### Innovation
- First application of Spatio-Temporal Graph Neural Networks to Bangladesh's river network
- Novel integration of spatial and temporal dependencies in flood prediction
- Graph-based representation of complex river confluence dynamics

### Impact
- Improved flood warning accuracy through multi-station spatial awareness
- Enhanced prediction capabilities for extreme weather events
- Scalable framework applicable to other river systems globally

## 📊 Model Performance
*Results will be updated after training completion*

| Model | RMSE | MAE | MAPE | R² |
|-------|------|-----|------|-----|
| DCRNN | TBD | TBD | TBD% | TBD |
| GraphConvLSTM | TBD | TBD | TBD% | TBD |

## 🤝 Contributing
Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Bangladesh Water Development Board (BWDB) for hydrological data context
- Flood Forecasting and Warning Centre (FFWC) for flood monitoring insights
- PyTorch Geometric community for graph neural network frameworks

## 📞 Contact
For questions about this research project, please open an issue on GitHub.
or mail: s2211056112@ru.ac.bd
---
**Built with ❤️ for Bangladesh flood forecasting research by Nayeemul Islam Nayeem**  
*Powered by PyTorch Geometric • Designed for real-world impact*
