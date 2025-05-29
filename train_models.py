import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_and_save_models():
    """Entrenar modelos y guardar todos los componentes necesarios"""
    
    print("🚀 Iniciando entrenamiento de modelos...")
    
    # Crear directorio para modelos si no existe
    os.makedirs('models', exist_ok=True)
    
    # 1. Cargar y preparar datos
    print("📊 Cargando datos...")
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    df = pd.read_csv(url)
    df.fillna(df.mean(), inplace=True)
    
    # 2. Limpieza de outliers
    print("🧹 Limpiando datos...")
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)].copy()
    
    # 3. Transformaciones
    print("🔄 Aplicando transformaciones...")
    cols_to_transform = ['crim', 'indus', 'nox', 'tax', 'ptratio', 'lstat']
    
    df_transformed = df_cleaned.copy()
    power_transformer = PowerTransformer()
    df_transformed[cols_to_transform] = power_transformer.fit_transform(
        df_transformed[cols_to_transform]
    )
    
    # 4. Separar características y target
    X = df_transformed.drop('medv', axis=1)
    y = df_transformed['medv']
    
    # 5. Escalado
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 6. División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.1, random_state=42
    )
    
    # 7. Entrenar modelos
    print("🤖 Entrenando modelos...")
    
    models = {}
    
    # Random Forest
    print("  - Random Forest...")
    models['Random Forest'] = RandomForestRegressor(
        n_estimators=100, 
        max_depth=None, 
        min_samples_split=2, 
        random_state=42
    )
    models['Random Forest'].fit(X_train, y_train)
    
    # Regresión Lineal
    print("  - Regresión Lineal...")
    models['Regresión Lineal'] = LinearRegression()
    models['Regresión Lineal'].fit(X_train, y_train)
    
    # XGBoost
    print("  - XGBoost...")
    models['XGBoost'] = xgb.XGBRegressor(random_state=42)
    models['XGBoost'].fit(X_train, y_train)
    
    # 8. Evaluar modelos
    print("📈 Evaluando modelos...")
    metrics = {}
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        metrics[model_name] = {'r2': round(r2, 4), 'mse': round(mse, 4)}
        print(f"  - {model_name}: R² = {r2:.4f}, MSE = {mse:.4f}")
    
    # 9. Guardar todo
    print("💾 Guardando modelos y transformadores...")
    
    # Guardar modelos
    for model_name, model in models.items():
        filename = model_name.lower().replace(' ', '_').replace('ó', 'o')
        joblib.dump(model, f'models/{filename}_model.pkl')
    
    # Guardar transformadores
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(power_transformer, 'models/power_transformer.pkl')
    
    # Guardar metadatos
    metadata = {
        'feature_names': X.columns.tolist(),
        'cols_to_transform': cols_to_transform,
        'metrics': metrics,
        'test_data': {
            'X_test': X_test,
            'y_test': y_test.tolist()
        }
    }
    joblib.dump(metadata, 'models/metadata.pkl')
    
    print("✅ ¡Modelos entrenados y guardados exitosamente!")
    print("📂 Archivos generados en la carpeta 'models/':")
    for file in os.listdir('models'):
        print(f"  - {file}")
    
    return models, scaler, power_transformer, metadata

if __name__ == "__main__":
    train_and_save_models()