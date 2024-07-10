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

print(data)


# Convertir fecha de nacimiento en edad
data['age'] = 2024 - pd.to_datetime(data['birthday']).dt.year

# Selección de características y variable objetivo
features = ['club', 'leagueCountry', 'age', 'height', 'weight', 'position', 'games', 'victories', 'ties', 'defeats',
            'yellowCards', 'yellowReds', 'redCards', 'rater1', 'rater2', 'refCountry', 'meanIAT', 'meanExp']
target = 'goals'

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesamiento
numeric_features = ['age', 'height', 'weight', 'games', 'victories', 'ties', 'defeats', 'yellowCards', 'yellowReds', 'redCards', 'rater1', 'rater2', 'meanIAT', 'meanExp']
categorical_features = ['club', 'leagueCountry', 'position', 'refCountry']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Modelo
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Entrenamiento del modelo
model.fit(X_train, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Predicción
new_data = pd.DataFrame({
    'club': ['TeamA'],
    'leagueCountry': ['England'],
    'age': [25],
    'height': [180],
    'weight': [75],
    'position': ['Forward'],
    'games': [20],
    'victories': [10],
    'ties': [5],
    'defeats': [5],
    'yellowCards': [2],
    'yellowReds': [1],
    'redCards': [0],
    'rater1': [3],
    'rater2': [3],
    'refCountry': ['England'],
    'meanIAT': [0.5],
    'meanExp': [0.3]
})

goals_pred = model.predict(new_data)
print(f'Predicted Goals: {goals_pred[0]}')

# Ver todos los objetos creados
import pprint
pprint.pprint(dir())

X