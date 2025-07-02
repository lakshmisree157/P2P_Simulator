"""
P2P Smart Grid Simulator - Main Application
Streamlit-based interactive demo for energy trading with AI algorithms
"""

import streamlit as st
import pandas as pd
import time
import copy
import io
import datetime
from algorithms.ai_price_predictor import (
    append_training_data, train_and_evaluate_model, predict_price, export_training_data, reset_training_data,
    generate_initial_training_data
)

# Handle orjson import issues
try:
    import plotly.io as pio
    # Set renderer to avoid orjson issues
    pio.renderers.default = "plotly_mimetype"
except ImportError:
    pass

# Import from our modular structure
from simulation.simulator import SmartGridSimulator
from visualization.plots import create_network_plot, create_metrics_plots

# Configure Streamlit page
st.set_page_config(
    page_title="P2P Smart Grid Simulator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Streamlit Application
def main():
    # Title and description
    st.title("⚡ P2P Smart Grid Simulator")
    st.markdown("**Interactive Demo - Energy Trading with AI Algorithms**")
    
    # Initialize session state
    if 'simulator' not in st.session_state:
        st.session_state.simulator = SmartGridSimulator()
        st.session_state.metrics_history = []
        st.session_state.is_running = False
        st.session_state.auto_step = False
        st.session_state.sim_history = []  # For undo feature
    if 'ai_training_data' not in st.session_state or not st.session_state['ai_training_data']:
        st.session_state['ai_training_data'] = generate_initial_training_data(50)
    if 'ai_model' not in st.session_state:
        st.session_state['ai_model'] = None
    if 'ai_r2' not in st.session_state:
        st.session_state['ai_r2'] = None
    if 'ai_predicted_price' not in st.session_state:
        st.session_state['ai_predicted_price'] = None
    # Train model on initial data if not already trained
    if st.session_state['ai_model'] is None and len(st.session_state['ai_training_data']) >= 10:
        model, r2 = train_and_evaluate_model(st.session_state['ai_training_data'])
        st.session_state['ai_model'] = model
        st.session_state['ai_r2'] = r2
        # Predict for next round
        time_of_day = (0) % 24
        total_generation = st.session_state['ai_training_data'][-1]['total_generation']
        total_demand = st.session_state['ai_training_data'][-1]['total_demand']
        next_time_of_day = (time_of_day + 1) % 24
        pred_price = predict_price(model, next_time_of_day, total_generation, total_demand)
        st.session_state['ai_predicted_price'] = pred_price
    
    # --- AI PATCHED SIMULATION STEP AND HELPERS ---
    def get_time_of_day():
        # Use simulation step as hour (0-23) for demo, or use real time if desired
        return (st.session_state.simulator.step_count - 1) % 24
    def get_total_generation_and_demand():
        houses = st.session_state.simulator.houses.values()
        total_generation = sum(h.energy for h in houses if h.energy > 0)
        total_demand = sum(abs(h.energy) for h in houses if h.energy < 0)
        return total_generation, total_demand
    def get_clearing_price():
        # Use avg_price from latest metrics
        if st.session_state.metrics_history:
            return st.session_state.metrics_history[-1]['avg_price']
        return 0.0
    def patched_simulate_step():
        result = st.session_state.simulator.simulate_step()
        st.session_state.metrics_history.append(result["metrics"])
        # Collect AI data
        time_of_day = get_time_of_day()
        total_generation, total_demand = get_total_generation_and_demand()
        clearing_price = get_clearing_price()
        st.session_state['ai_training_data'] = append_training_data(
            st.session_state['ai_training_data'],
            time_of_day, total_generation, total_demand, clearing_price
        )
        # Train model if enough data
        if len(st.session_state['ai_training_data']) >= 10:
            model, r2 = train_and_evaluate_model(st.session_state['ai_training_data'])
            st.session_state['ai_model'] = model
            st.session_state['ai_r2'] = r2
            # Predict for next round
            next_time_of_day = (time_of_day + 1) % 24
            next_total_generation, next_total_demand = get_total_generation_and_demand()
            pred_price = predict_price(model, next_time_of_day, next_total_generation, next_total_demand)
            st.session_state['ai_predicted_price'] = pred_price
        else:
            st.session_state['ai_model'] = None
            st.session_state['ai_r2'] = None
            st.session_state['ai_predicted_price'] = None
        st.rerun()
    # --- END PATCHED ---

    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Network configuration
        st.subheader("Network Configuration")
        num_houses = st.slider("Number of Houses", 4, 15, 8)
        network_type = st.selectbox("Network Topology", ["ring", "mesh", "star", "random", "grid"])
        
        # Ensure network_type is not None
        if network_type is None:
            network_type = "ring"
        
        if st.button("🔄 Initialize Network"):
            st.session_state.simulator.initialize_network(num_houses, network_type)
            st.session_state.metrics_history = []
            st.session_state.initial_houses = copy.deepcopy(st.session_state.simulator.houses)
            st.rerun()
        
        st.divider()
        
        # Simulation controls
        st.subheader("Simulation Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Step"):
                # Store current state for undo
                st.session_state.sim_history.append({
                    'simulator': copy.deepcopy(st.session_state.simulator),
                    'metrics_history': copy.deepcopy(st.session_state.metrics_history)
                })
                patched_simulate_step()
        
        with col2:
            if st.button("⏹️ Reset"):
                st.session_state.simulator.initialize_network(num_houses, network_type)
                st.session_state.metrics_history = []
                st.session_state.initial_houses = copy.deepcopy(st.session_state.simulator.houses)
                st.session_state.sim_history = []
                st.rerun()
        
        # Undo button
        if st.button("↩️ Undo Last Step"):
            if st.session_state.sim_history:
                last_state = st.session_state.sim_history.pop()
                st.session_state.simulator = last_state['simulator']
                st.session_state.metrics_history = last_state['metrics_history']
                st.rerun()
            else:
                st.warning("No previous step to undo.")
        
        # Auto-run toggle
        auto_run = st.checkbox("🔄 Auto Run", value=st.session_state.auto_step)
        if auto_run != st.session_state.auto_step:
            st.session_state.auto_step = auto_run
            st.rerun()
        
        if auto_run:
            time.sleep(2)
            patched_simulate_step()
        
        st.divider()
        
        # Manual house addition
        st.subheader("Add House Manually")
        with st.form("add_house_form"):
            col1, col2 = st.columns(2)
            with col1:
                x_pos = st.number_input("X Position", value=0.0)
                y_pos = st.number_input("Y Position", value=0.0)
            with col2:
                energy = st.number_input("Energy (kW)", value=0.0, help="Positive=surplus, Negative=deficit")
                bid_price = st.number_input("Bid Price ($)", value=20.0, min_value=0.1)
            
            house_type = st.selectbox("House Type", ["prosumer", "consumer"])
            
            # Ensure house_type is not None
            if house_type is None:
                house_type = "prosumer"
            
            if st.form_submit_button("➕ Add House"):
                new_id = st.session_state.simulator.add_house_manually(x_pos, y_pos, energy, bid_price, house_type)
                st.success(f"Added House {new_id}!")
                st.rerun()
        
        st.divider()
        st.header('🤖 AI Price Predictor')
        # Reset button
        if st.button('Reset Model'):
            st.session_state['ai_training_data'] = generate_initial_training_data(50)
            st.session_state['ai_model'] = None
            st.session_state['ai_r2'] = None
            st.session_state['ai_predicted_price'] = None
            st.success('AI model and training data reset to 50 initial rows.')
            st.rerun()
        # Download button
        if st.session_state['ai_training_data']:
            csv = export_training_data(st.session_state['ai_training_data'])
            st.download_button('Download Training Data (CSV)', data=csv, file_name='ai_training_data.csv', mime='text/csv')
            st.caption('You can share this CSV with others or use it to initialize another simulator instance.')
        # Model status
        st.markdown(f"**Training batch size:** {len(st.session_state['ai_training_data'])}")
        if st.session_state['ai_model'] is None or len(st.session_state['ai_training_data']) < 10:
            st.info(f"Model not trained. ({len(st.session_state['ai_training_data'])}/10 data points)")
        else:
            st.success(f"Model trained. R² score: {st.session_state['ai_r2']:.3f}")
            if st.session_state['ai_r2'] > 0.8 and st.session_state['ai_predicted_price'] is not None:
                st.markdown(f"**Predicted Price (next round):** ${st.session_state['ai_predicted_price']:.2f}")
            elif st.session_state['ai_r2'] <= 0.8:
                st.warning('Model accuracy is too low for prediction (R² ≤ 0.8).')
    
    # Main content area
    if not st.session_state.simulator.houses:
        st.info("👆 Please initialize the network using the sidebar controls.")
        return
    
    # Get current network data
    network_data = st.session_state.simulator.get_network_data()
    
    # Create main layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Network visualization
        st.subheader("🔗 Network Visualization")
        show_trade_paths = st.checkbox("Show Trade Paths", value=True)
        
        try:
            network_fig = create_network_plot(network_data, show_trade_paths)
            st.plotly_chart(network_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying network visualization: {str(e)}")
            st.info("Please try refreshing the page or reinstalling dependencies.")

    # House state comparison tables (moved for better visibility)
    st.markdown("### House State Comparison")
    st.markdown(
        "Below you can compare the <b>initial</b> and <b>current</b> state of all houses in the network after each simulation step.",
        unsafe_allow_html=True
    )
    col_init, col_evol = st.columns(2)
    with col_init:
        st.markdown("#### 🟦 Initial House State")
        if "initial_houses" in st.session_state:
            init_houses_df = pd.DataFrame([
                {
                    "ID": h.id,
                    "Energy": f"{h.energy:.1f}",
                    "Bid": f"${h.bid_price:.2f}",
                    "Type": h.house_type,
                    "Status": "Seller" if h.is_seller else "Buyer" if h.is_buyer else "Neutral"
                }
                for h in st.session_state.initial_houses.values()
            ])
            st.dataframe(init_houses_df, use_container_width=True)
        else:
            st.info("No initial state available. Please initialize the network.")
    with col_evol:
        st.markdown("#### 🟩 Current House State")
        houses_df = pd.DataFrame([
            {
                "ID": h.id,
                "Energy": f"{h.energy:.1f}",
                "Bid": f"${h.bid_price:.2f}",
                "Type": h.house_type,
                "Status": "Seller" if h.is_seller else "Buyer" if h.is_buyer else "Neutral"
            }
            for h in network_data["houses"].values()
        ])
        st.dataframe(houses_df, use_container_width=True)

    # Legend (moved here after the tables)
    st.markdown("""
    **Legend:**
    - 🟢 Green Houses: Energy Surplus (Sellers)
    - 🔴 Red Houses: Energy Deficit (Buyers)  
    - 🟢 Green Lines: Low Usage (<50%)
    - 🟠 Orange Lines: Medium Usage (50-80%)
    - 🔴 Red Lines: High Usage (>80%)
    - 🟣 Purple Dashed: Active Trade Paths
    """)

    with col2:
        # Current metrics
        st.subheader("📊 Live Metrics")
        
        if st.session_state.metrics_history:
            latest_metrics = st.session_state.metrics_history[-1]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Trades", latest_metrics["total_trades"])
                st.metric("Efficiency", f"{latest_metrics['efficiency']:.1f}%")
            
            with col2:
                st.metric("Avg Price", f"${latest_metrics['avg_price']:.2f}")
                st.metric("Congestion", f"{latest_metrics['congestion']:.1f}%")
        
        # Recent trades
        st.subheader("💰 Recent Trades")
        recent_trades = network_data["trades_history"]
        
        if recent_trades:
            for trade in recent_trades[-5:]:  # Show last 5 trades
                with st.expander(f"Trade: House {trade.seller_id} → House {trade.buyer_id}"):
                    st.write(f"**Amount:** {trade.amount:.1f} kW")
                    st.write(f"**Price:** ${trade.price:.2f}")
                    st.write(f"**Path:** {' → '.join(map(str, trade.path))}")
        else:
            st.info("No trades yet. Run a simulation step!")
        
        # Export buttons
        st.markdown("**Export Data**")
        # Export current house state
        csv_buffer = io.StringIO()
        houses_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Current House State (CSV)",
            data=csv_buffer.getvalue(),
            file_name="current_house_state.csv",
            mime="text/csv"
        )
        # Export metrics history
        if st.session_state.metrics_history:
            metrics_df = pd.DataFrame(st.session_state.metrics_history)
            csv_buffer2 = io.StringIO()
            metrics_df.to_csv(csv_buffer2, index=False)
            st.download_button(
                label="Download Metrics History (CSV)",
                data=csv_buffer2.getvalue(),
                file_name="metrics_history.csv",
                mime="text/csv"
            )
    
    # Metrics history
    if st.session_state.metrics_history:
        st.subheader("📈 Performance Metrics")
        try:
            metrics_fig = create_metrics_plots(st.session_state.metrics_history)
            st.plotly_chart(metrics_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying metrics: {str(e)}")
        # Metrics table
        metrics_df = pd.DataFrame(st.session_state.metrics_history)
        st.markdown("**Metrics Table (per step):**")
        st.dataframe(metrics_df, use_container_width=True)
    
    # Algorithm information
    st.subheader("🧠 Algorithms in Action")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **A* Pathfinding**
        
        Finds optimal energy flow paths while considering:
        - Line capacity constraints
        - Network congestion
        - Shortest path optimization
        """)
    
    with col2:
        st.success("""
        **Double Auction**
        
        Matches energy trades by:
        - Sorting buyers by highest bid
        - Sorting sellers by lowest ask
        - Fair price discovery
        """)
    
    with col3:
        st.warning("""
        **Priority Queues**
        
        Efficiently manages:
        - Bid/ask ordering
        - Path finding priorities
        - Trade execution sequence
        """)

if __name__ == "__main__":
    main()