import streamlit as st
import pandas as pd
import pybaseball as pyb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

@st.cache
def load_data():
    # Load Statcast data for a specific season
    data = pyb.statcast(start_dt='2023-04-01', end_dt='2023-09-30')
    return data

@st.cache
def preprocess_data(data):
    # Filter relevant columns
    pitch_data = data[['player_name', 'pitch_type', 'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'release_pos_x', 'release_pos_y', 'release_pos_z', 'zone', 'woba_value']]
    pitch_data = pitch_data.dropna()
    pitch_data['target'] = pitch_data['woba_value']
    pitch_data = pitch_data.drop(columns=['woba_value'])
    return pitch_data

@st.cache
def train_model(pitch_data):
    X = pitch_data.drop(columns=['target', 'player_name', 'pitch_type'])
    y = pitch_data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def calculate_stuff_plus(model, pitch_data):
    X = pitch_data.drop(columns=['target', 'player_name', 'pitch_type'])
    y_pred = model.predict(X)
    stuff_plus = (y_pred - y_pred.mean()) / y_pred.std() * 100 + 100
    pitch_data['stuff_plus'] = stuff_plus
    return pitch_data

st.title('Pitcher Stuff+ Score Viewer')

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text('Loading data...done!')

pitch_data = preprocess_data(data)
model, X_train, X_test, y_train, y_test = train_model(pitch_data)
pitch_data = calculate_stuff_plus(model, pitch_data)

pitchers = pitch_data['player_name'].unique()
selected_pitcher = st.selectbox('Select a pitcher:', pitchers)

pitcher_data = pitch_data[pitch_data['player_name'] == selected_pitcher]

st.write(f"Stuff+ Score for {selected_pitcher}:")
st.write(pitcher_data[['pitch_type', 'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'stuff_plus']])
