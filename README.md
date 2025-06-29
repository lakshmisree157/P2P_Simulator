# ⚡ P2P Smart Grid Simulator

An interactive demonstration of peer-to-peer energy trading in smart grids using AI algorithms.

## 🏗️ Project Structure

The project has been organized into a modular structure for better maintainability:

```
AIML lab el/
├── main.py                 # Main Streamlit application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── models/                # Data models and structures
│   ├── __init__.py
│   └── data_models.py     # House, Connection, Trade classes
├── algorithms/            # AI algorithms implementation
│   ├── __init__.py
│   ├── pathfinding.py     # A* pathfinding algorithm
│   └── auction.py         # Double auction mechanism
├── simulation/            # Core simulation logic
│   ├── __init__.py
│   └── simulator.py       # Main SmartGridSimulator class
└── visualization/         # Plotting and visualization
    ├── __init__.py
    └── plots.py           # Network and metrics visualization
```

## 🚀 Features

### Core Functionality
- **Interactive Network Visualization**: Real-time display of houses, connections, and energy flow
- **Multiple Network Topologies**: Ring, Mesh, and Star network configurations
- **Dynamic House Management**: Add houses manually with custom parameters
- **Real-time Metrics**: Live tracking of trades, prices, efficiency, and congestion

### AI Algorithms
- **A* Pathfinding**: Optimal energy flow path discovery considering capacity constraints
- **Double Auction**: Fair price discovery and trade matching between buyers and sellers
- **Priority Queues**: Efficient management of bid/ask ordering and trade execution

### Network Topologies
- **Ring**: Circular connections with diagonal redundancy
- **Mesh**: Fully connected network within distance constraints
- **Star**: Centralized hub with edge connections

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "AIML lab el"
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run main.py
   ```

## 📊 Usage

### Getting Started
1. Open the application in your browser
2. Use the sidebar to configure the network:
   - Set number of houses (4-15)
   - Choose network topology (ring/mesh/star)
   - Click "Initialize Network"

### Simulation Controls
- **Step**: Execute one simulation step manually
- **Auto Run**: Automatically run simulation steps
- **Reset**: Clear simulation and start fresh

### Adding Houses
- Use the "Add House Manually" section in the sidebar
- Set position (X, Y coordinates)
- Define energy balance (positive = surplus, negative = deficit)
- Set bid price and house type

### Visualization Features
- **Network View**: Interactive plot showing houses and connections
- **Trade Paths**: Purple dashed lines show active energy trades
- **Connection Status**: Color-coded lines based on utilization
- **Live Metrics**: Real-time statistics and performance indicators

## 🧠 Algorithm Details

### A* Pathfinding
- **Purpose**: Find optimal energy flow paths between houses
- **Heuristic**: Euclidean distance between nodes
- **Constraints**: Line capacity and network congestion
- **Optimization**: Minimizes path cost while avoiding overloaded connections

### Double Auction
- **Purpose**: Match energy buyers with sellers at fair prices
- **Mechanism**: 
  - Buyers sorted by highest bid price
  - Sellers sorted by lowest ask price (80% of bid)
  - Trades executed when buyer bid ≥ seller ask
- **Price Discovery**: Average of buyer bid and seller ask

## 📈 Metrics Explained

- **Total Trades**: Number of successful energy transactions
- **Average Price**: Mean price per kWh across all trades
- **Market Efficiency**: Percentage of total supply/demand satisfied
- **Network Congestion**: Average utilization of power lines
- **Price Standard Deviation**: Price volatility indicator

## 🔧 Customization

### Adding New Algorithms
1. Create new file in `algorithms/` directory
2. Implement algorithm class with required interface
3. Import and integrate in `simulation/simulator.py`

### Extending Data Models
1. Add new classes to `models/data_models.py`
2. Update visualization functions in `visualization/plots.py`
3. Modify simulator to handle new data types

### New Network Topologies
1. Add topology method in `simulation/simulator.py`
2. Update network type selection in `main.py`
3. Test with different house configurations

## 🐛 Troubleshooting

### Common Issues
- **Import Errors**: Ensure all dependencies are installed via `requirements.txt`
- **Visualization Issues**: Check that plotly is properly installed
- **Performance**: Reduce number of houses for better performance on slower machines

### Dependencies
- **Streamlit**: Web framework for the interactive interface
- **NumPy**: Numerical computations and random number generation
- **Pandas**: Data manipulation and display
- **NetworkX**: Graph theory and network analysis
- **Plotly**: Interactive visualizations

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📚 References

- A* Pathfinding Algorithm
- Double Auction Mechanisms
- Smart Grid Energy Trading
- Network Topology Design
