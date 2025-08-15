"""
Bangladesh River Flood Forecasting - Data Utilities
Data generation, preprocessing, and feature engineering functions

Extracted from Phase 2: Steps 5-8
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

def create_bangladesh_gauge_network():
    """Create a realistic Bangladesh gauge station network"""
    
    # Major rivers and approximate gauge locations
    major_rivers = {
        'Brahmaputra': [
            {'name': 'Chilmari', 'lat': 25.86, 'lon': 89.64, 'river': 'Brahmaputra'},
            {'name': 'Bahadurabad', 'lat': 25.20, 'lon': 89.68, 'river': 'Brahmaputra'},
            {'name': 'Sirajganj', 'lat': 24.45, 'lon': 89.70, 'river': 'Brahmaputra'},
        ],
        'Ganges': [
            {'name': 'Rajshahi', 'lat': 24.37, 'lon': 88.60, 'river': 'Ganges'},
            {'name': 'Hardinge Bridge', 'lat': 24.06, 'lon': 89.03, 'river': 'Ganges'},
            {'name': 'Goalundo', 'lat': 23.75, 'lon': 89.85, 'river': 'Ganges'},
        ],
        'Meghna': [
            {'name': 'Bhairab Bazar', 'lat': 24.05, 'lon': 90.98, 'river': 'Meghna'},
            {'name': 'Chandpur', 'lat': 23.23, 'lon': 90.85, 'river': 'Meghna'},
        ]
    }
    
    # Create gauge station dataframe
    gauge_stations = []
    station_id = 1
    
    for river, stations in major_rivers.items():
        for station in stations:
            gauge_stations.append({
                'station_id': station_id,
                'station_name': station['name'],
                'latitude': station['lat'],
                'longitude': station['lon'],
                'river_name': river,
                'elevation': np.random.normal(10, 5),  # Simulated elevation
                'catchment_area': np.random.uniform(1000, 50000)  # km²
            })
            station_id += 1
    
    return pd.DataFrame(gauge_stations)


def generate_river_time_series(gauge_df, start_date='2020-01-01', end_date='2023-12-31'):
    """Generate realistic time series data for river gauges"""
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Initialize data storage
    gauge_data = {}
    
    # Seasonal patterns for Bangladesh (monsoon-driven)
    def seasonal_pattern(day_of_year):
        """Create monsoon seasonal pattern"""
        # Pre-monsoon (March-May): gradual increase
        # Monsoon (June-September): high levels with peaks
        # Post-monsoon (October-November): gradual decrease  
        # Winter (December-February): low levels
        
        monsoon_peak = 200  # Around mid-July (day 200)
        
        if day_of_year < 60:  # Jan-Feb (Winter)
            return 0.2 + 0.1 * np.sin(2 * np.pi * day_of_year / 365)
        elif day_of_year < 120:  # Mar-Apr (Pre-monsoon)
            return 0.3 + 0.2 * (day_of_year - 60) / 60
        elif day_of_year < 270:  # May-Sep (Monsoon)
            peak_factor = np.exp(-((day_of_year - monsoon_peak) ** 2) / (2 * 30 ** 2))
            return 0.5 + 0.4 * peak_factor + 0.1 * np.sin(2 * np.pi * day_of_year / 30)
        elif day_of_year < 330:  # Oct-Nov (Post-monsoon)
            return 0.6 - 0.3 * (day_of_year - 270) / 60
        else:  # Dec (Winter)
            return 0.3 - 0.1 * (day_of_year - 330) / 35
    
    # Generate data for each gauge station
    for idx, station in gauge_df.iterrows():
        station_id = station['station_id']
        river_name = station['river_name']
        elevation = station['elevation']
        
        # Base water level depends on elevation and river characteristics
        if river_name == 'Brahmaputra':
            base_level = 15 - elevation * 0.1  # Higher base for major river
            volatility = 0.3
        elif river_name == 'Ganges':
            base_level = 12 - elevation * 0.1
            volatility = 0.25
        else:  # Meghna
            base_level = 10 - elevation * 0.1
            volatility = 0.2
        
        # Generate time series
        water_levels = []
        discharge_rates = []
        
        for i, date in enumerate(dates):
            day_of_year = date.timetuple().tm_yday
            
            # Seasonal component
            seasonal = seasonal_pattern(day_of_year)
            
            # Random weather events (cyclones, heavy rainfall)
            weather_event = 0
            if np.random.random() < 0.05:  # 5% chance of extreme weather
                weather_event = np.random.exponential(0.3)
            
            # Upstream influence (for downstream stations)
            upstream_influence = 0
            if station['station_name'] in ['Sirajganj', 'Goalundo', 'Chandpur']:
                upstream_influence = 0.1 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
            
            # Combine all factors
            daily_level = (base_level * (1 + seasonal) + 
                          weather_event + 
                          upstream_influence + 
                          np.random.normal(0, volatility))
            
            # Ensure positive values
            daily_level = max(daily_level, 0.5)
            
            # Discharge rate correlates with water level
            discharge = daily_level ** 1.5 * np.random.normal(100, 20)
            discharge = max(discharge, 10)
            
            water_levels.append(daily_level)
            discharge_rates.append(discharge)
        
        # Store station data
        gauge_data[station_id] = {
            'station_name': station['station_name'],
            'dates': dates,
            'water_level': water_levels,
            'discharge': discharge_rates,
            'latitude': station['latitude'],
            'longitude': station['longitude'],
            'river_name': river_name
        }
    
    return gauge_data


def generate_weather_data(gauge_df, start_date='2020-01-01', end_date='2023-12-31'):
    """Generate weather data (precipitation, temperature) for each station"""
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    weather_data = {}
    
    for idx, station in gauge_df.iterrows():
        station_id = station['station_id']
        lat = station['latitude']
        
        precipitation = []
        temperature = []
        humidity = []
        
        for date in dates:
            day_of_year = date.timetuple().tm_yday
            
            # Monsoon precipitation pattern
            if 150 <= day_of_year <= 270:  # Monsoon season
                rain_prob = 0.7
                rain_intensity = np.random.exponential(15)
            elif 120 <= day_of_year <= 150:  # Pre-monsoon
                rain_prob = 0.4
                rain_intensity = np.random.exponential(8)
            else:  # Dry season
                rain_prob = 0.1
                rain_intensity = np.random.exponential(3)
            
            daily_rain = rain_intensity if np.random.random() < rain_prob else 0
            precipitation.append(daily_rain)
            
            # Temperature (varies by season and latitude)
            base_temp = 25 + 5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            daily_temp = base_temp + np.random.normal(0, 3) - (lat - 24) * 0.5
            temperature.append(max(daily_temp, 10))
            
            # Humidity (higher during monsoon)
            base_humidity = 70 + 20 * np.sin(2 * np.pi * (day_of_year - 150) / 365)
            daily_humidity = base_humidity + np.random.normal(0, 5)
            humidity.append(max(min(daily_humidity, 100), 30))
        
        weather_data[station_id] = {
            'dates': dates,
            'precipitation': precipitation,
            'temperature': temperature,
            'humidity': humidity
        }
    
    return weather_data


def create_sequences(processed_df, seq_length=7, forecast_horizon=1):
    """Create sequences for spatio-temporal GNN training"""
    
    sequences = []
    targets = []
    dates = []
    station_ids = []
    
    # Group by date to create temporal snapshots
    daily_data = processed_df.groupby('date')
    
    # Sort dates
    unique_dates = sorted(processed_df['date'].unique())
    
    # Create sequences
    for i in range(seq_length, len(unique_dates) - forecast_horizon + 1):
        # Input sequence (past seq_length days)
        input_sequence = []
        
        for j in range(i - seq_length, i):
            daily_snapshot = daily_data.get_group(unique_dates[j])
            
            # Create node features for this day
            node_features = []
            for station_id in sorted(daily_snapshot['station_id'].unique()):
                station_data = daily_snapshot[daily_snapshot['station_id'] == station_id]
                
                if not station_data.empty:
                    # Get scaled features (need to define feature columns)
                    feature_cols = [col for col in station_data.columns if col.endswith('_scaled')]
                    if feature_cols:
                        features = station_data[feature_cols].values[0]
                    else:
                        features = np.zeros(10)  # Default feature vector
                    node_features.append(features)
                else:
                    # Handle missing data with zeros
                    node_features.append(np.zeros(10))
            
            input_sequence.append(np.array(node_features))
        
        # Target (forecast_horizon days ahead)
        target_date = unique_dates[i + forecast_horizon - 1]
        target_snapshot = daily_data.get_group(target_date)
        
        target_values = []
        for station_id in sorted(target_snapshot['station_id'].unique()):
            station_data = target_snapshot[target_snapshot['station_id'] == station_id]
            if not station_data.empty:
                target_values.append(station_data['water_level_scaled'].values[0] if 'water_level_scaled' in station_data.columns else 0.0)
            else:
                target_values.append(0.0)
        
        sequences.append(np.array(input_sequence))
        targets.append(np.array(target_values))
        dates.append(target_date)
        station_ids.append(sorted(target_snapshot['station_id'].unique()))
    
    return np.array(sequences), np.array(targets), dates, station_ids


class RiverDataset(torch.utils.data.Dataset):
    """Custom dataset for river flood forecasting"""
    
    def __init__(self, X_sequences, y_sequences):
        self.X = torch.FloatTensor(X_sequences)
        self.y = torch.FloatTensor(y_sequences)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
