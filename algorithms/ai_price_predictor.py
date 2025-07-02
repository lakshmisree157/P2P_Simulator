import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# In-memory data store (to be managed by Streamlit session_state)
def append_training_data(data_list, time_of_day, total_generation, total_demand, clearing_price):
    data_list.append({
        'time_of_day': time_of_day,
        'total_generation': total_generation,
        'total_demand': total_demand,
        'clearing_price': clearing_price
    })
    return data_list

def train_and_evaluate_model(data_list):
    df = pd.DataFrame(data_list)
    X = df[['time_of_day', 'total_generation', 'total_demand']].values
    y = df['clearing_price'].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    return model, r2

def predict_price(model, time_of_day, total_generation, total_demand):
    X_next = np.array([[time_of_day, total_generation, total_demand]])
    return float(model.predict(X_next)[0])

def export_training_data(data_list):
    df = pd.DataFrame(data_list)
    return df.to_csv(index=False)

def reset_training_data():
    return []

def load_training_data_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df.to_dict(orient='records')

def save_training_data_to_csv(data_list, csv_path):
    df = pd.DataFrame(data_list)
    df.to_csv(csv_path, index=False)

def generate_initial_training_data(n=50, seed=42):
    np.random.seed(seed)
    data = []
    for _ in range(n):
        time_of_day = np.random.randint(0, 24)
        total_generation = np.random.uniform(100, 500)
        total_demand = np.random.uniform(80, 480)
        # Simulate clearing price as a function of features + noise
        clearing_price = 15 + 0.1 * (total_demand - total_generation) + 0.2 * time_of_day + np.random.normal(0, 2)
        data.append({
            'time_of_day': int(time_of_day),
            'total_generation': float(total_generation),
            'total_demand': float(total_demand),
            'clearing_price': float(clearing_price)
        })
    return data 