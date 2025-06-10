# P2P Smart Grid Simulator

⚡ An interactive Streamlit-based simulator for peer-to-peer (P2P) energy trading in a smart grid network using AI algorithms.

---

## Project Description

This project simulates a decentralized smart grid where houses act as prosumers (both producers and consumers) or consumers of energy. The simulator models energy trading between houses using advanced algorithms such as A* pathfinding for optimal energy flow and a Continuous Double Auction mechanism for fair market pricing.

The application provides a rich interactive visualization of the network topology, energy trades, and live market metrics, allowing users to explore different network configurations and trading dynamics.

---

## Features

- Configurable network topologies: ring, mesh, and star
- Houses with dynamic energy surplus or deficit and bid prices
- A* pathfinding algorithm to find optimal energy flow paths considering line capacities and congestion
- Continuous Double Auction mechanism to match buyers and sellers fairly
- Real-time visualization of the network, trades, and congestion using Plotly
- Interactive controls to initialize the network, simulate trading steps, and add houses manually
- Live metrics dashboard showing trades, prices, efficiency, and congestion

---

## Installation

1. Clone the repository or download the source code.

2. Install the required Python packages. It is recommended to use a virtual environment.

```bash
pip install streamlit numpy pandas networkx plotly
```

---

## Usage

Run the Streamlit app with the following command:

```bash
streamlit run main.py
```

This will open the app in your default web browser.

---

## How to Use the Simulator

- Use the sidebar controls to configure the number of houses and network topology.
- Click **Initialize Network** to create the network.
- Use the **Step** button to simulate one trading step or enable **Auto Run** for continuous simulation.
- Add houses manually by specifying their position, energy surplus/deficit, bid price, and type.
- Visualize the network with color-coded houses and connections indicating energy status and congestion.
- View live metrics such as total trades, average price, market efficiency, and network congestion.
- Explore recent trades and house statuses in the main panel.

---

## Key Components and Algorithms

### A* Pathfinding

The simulator uses the A* algorithm to find optimal paths for energy flow between buyers and sellers. It considers:

- Euclidean distance as the heuristic
- Line capacity constraints to avoid congested connections
- Network congestion to optimize routing

### Continuous Double Auction

Energy trading is facilitated by a Continuous Double Auction mechanism that:

- Sorts buyers by highest bid price and sellers by lowest ask price
- Matches trades where buyer bid meets or exceeds seller ask
- Determines fair trade prices and quantities
- Updates energy balances of houses accordingly

---

## License

This project is provided as-is under the MIT License. Feel free to use and modify it for your own purposes.
