import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error



# Cargar los datos
data = pd.read_csv('trabajo/bbdd/redcard_data.csv')  # Reemplazar con la ruta correcta del archivo CSV

# Convertir fecha de nacimiento en edad
data['age'] = 2024 - pd.to_datetime(data['birthday']).dt.year

# Selección de características y variable objetivo
features = ['club', 'leagueCountry', 'age', 'height', 'weight', 'position', 'games', 'victories', 'ties', 'defeats',
            'yellowCards', 'yellowReds', 'redCards', 'rater1', 'rater2', 'refCountry', 'meanIAT', 'meanExp']
target = 'goals'