import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              VotingRegressor, ExtraTreesRegressor, StackingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import KNNImputer
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import warnings
from scipy import stats

# Ignorar advertencias
warnings.filterwarnings('ignore')

# --- IMPORTANTE: Configuración global para scikit-learn para que los transformadores devuelvan DataFrames ---
from sklearn import set_config
set_config(transform_output="pandas")


def create_advanced_features(df, feature_thresholds=None):
    """
    Aplica ingeniería de características avanzada al DataFrame
    
    Args:
        df: DataFrame original
        feature_thresholds: Diccionario con umbrales para características binarias (e.g., de metadata)
        
    Returns:
        DataFrame con características mejoradas
    """
    print("🔧 Aplicando ingeniería de características avanzada...")
    df_enhanced = df.copy()
    
    # Asegúrate de que las características originales de Boston Housing están presentes
    # Agrega 'age' a key_features si tu pipeline de entrenamiento la usa para transformaciones polinómicas
    key_features = ['rm', 'lstat', 'ptratio', 'dis', 'tax', 'crim', 'age'] 
    for feature in key_features:
        if feature in df_enhanced.columns:
            # Transformaciones no lineales
            df_enhanced[f'{feature}_squared'] = df_enhanced[feature] ** 2
            # Manejo de valores no positivos para sqrt y log
            df_enhanced[f'{feature}_sqrt'] = np.sqrt(np.abs(df_enhanced[feature]))
            df_enhanced[f'{feature}_log'] = np.log1p(np.abs(df_enhanced[feature]))
            # Evitar división por cero
            df_enhanced[f'{feature}_inv'] = 1 / (df_enhanced[feature] + 1e-8)
    
    # 2. Interacciones importantes basadas en conocimiento del dominio
    interactions = [
        ('rm', 'lstat'),        # Habitaciones x Estatus socioeconómico
        ('crim', 'dis'),        # Crimen x Distancia a centros
        ('tax', 'ptratio'),     # Impuestos x Ratio educativo
        ('nox', 'dis'),         # Contaminación x Distancia
        ('age', 'rm')           # Edad de edificios x Habitaciones
    ]
    
    for feat1, feat2 in interactions:
        if all(col in df_enhanced.columns for col in [feat1, feat2]):
            df_enhanced[f'{feat1}_{feat2}_interaction'] = df_enhanced[feat1] * df_enhanced[feat2]
    
    # 3. Ratios significativos
    ratio_combinations = [
        ('rm', 'age'),          # Habitaciones por edad del edificio
        ('tax', 'rm'),          # Impuestos por habitación
        ('crim', 'dis'),        # Crimen relativo a distancia
        ('lstat', 'rm'),        # Estatus por habitaciones
        ('ptratio', 'rm')       # Ratio educativo por habitaciones
    ]
    
    for num, den in ratio_combinations:
        if all(col in df_enhanced.columns for col in [num, den]):
            df_enhanced[f'{num}_per_{den}'] = df_enhanced[num] / (df_enhanced[den] + 1e-8)
    
    # 4. Características categóricas binarias (banderas) - Usar umbrales pasados o calcular si no existen
    if 'crim' in df_enhanced.columns:
        if feature_thresholds and 'crim_quantile_75' in feature_thresholds:
            df_enhanced['high_crime'] = (df_enhanced['crim'] > feature_thresholds['crim_quantile_75']).astype(int)
        else:
            # Fallback: calcular si no hay umbrales predefinidos (solo para entrenamiento si no se guardaron)
            df_enhanced['high_crime'] = (df_enhanced['crim'] > df_enhanced['crim'].quantile(0.75)).astype(int)
    
    if 'rm' in df_enhanced.columns:
        if feature_thresholds and 'rm_large_threshold' in feature_thresholds:
            df_enhanced['large_rooms'] = (df_enhanced['rm'] > feature_thresholds['rm_large_threshold']).astype(int)
        else:
            df_enhanced['large_rooms'] = (df_enhanced['rm'] > 7).astype(int) # Usar 7 como default
    
    if 'lstat' in df_enhanced.columns:
        if feature_thresholds and 'lstat_quantile_75' in feature_thresholds:
            df_enhanced['low_status'] = (df_enhanced['lstat'] > feature_thresholds['lstat_quantile_75']).astype(int)
        else:
            # Fallback: calcular si no hay umbrales predefinidos
            df_enhanced['low_status'] = (df_enhanced['lstat'] > df_enhanced['lstat'].quantile(0.75)).astype(int)
    
    # 5. Índices compuestos
    if all(col in df_enhanced.columns for col in ['rm', 'lstat', 'ptratio']):
        df_enhanced['livability_index'] = (
            df_enhanced['rm'] * 0.4 - 
            df_enhanced['lstat'] * 0.3 - 
            df_enhanced['ptratio'] * 0.3
        )
    
    print(f"   ✅ Características transformadas: {len(df.columns)} → {len(df_enhanced.columns)}")
    return df_enhanced

def advanced_outlier_removal(df, target_col='medv', method='isolation'):
    """
    Elimina outliers usando múltiples métodos avanzados
    
    Args:
        df: DataFrame de entrada
        target_col: Columna objetivo
        method: 'isolation', 'iqr', 'multi' (combinación)
        
    Returns:
        DataFrame sin outliers
    """
    print(f"🎯 Eliminando outliers usando método: {method}")
    
    if method == 'isolation':
        from sklearn.ensemble import IsolationForest
        
        X = df.drop(target_col, axis=1) if target_col in df.columns else df
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outliers = iso_forest.fit_predict(X) == -1
        
        print(f"   📊 Isolation Forest detectó: {outliers.sum()} outliers")
        
    elif method == 'iqr':
        outliers = np.zeros(len(df), dtype=bool)
        
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2 * IQR  # Más conservador que 1.5
            upper_bound = Q3 + 2 * IQR
            
            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers = outliers | col_outliers
        
        print(f"   📊 IQR extendido detectó: {outliers.sum()} outliers")
        
    elif method == 'multi':
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        
        X = df.drop(target_col, axis=1) if target_col in df.columns else df
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        iso_outliers = iso_forest.fit_predict(X) == -1
        
        # Local Outlier Factor
        lof = LocalOutlierFactor(contamination=0.05)
        lof_outliers = lof.fit_predict(X) == -1
        
        # Z-score para el target
        if target_col in df.columns:
            z_scores = np.abs(stats.zscore(df[target_col]))
            zscore_outliers = z_scores > 3
        else:
            zscore_outliers = np.zeros(len(df), dtype=bool)
        
        # Combinar métodos - outlier si al menos 2 métodos lo detectan
        total_outliers = (iso_outliers.astype(int) + 
                          lof_outliers.astype(int) + 
                          zscore_outliers.astype(int))
        outliers = total_outliers >= 2
        
        print(f"   Método combinado:")
        print(f"     - Isolation Forest: {iso_outliers.sum()}")
        print(f"     - Local Outlier Factor: {lof_outliers.sum()}")
        print(f"     - Z-score: {zscore_outliers.sum()}")
        print(f"     - Total removidos: {outliers.sum()}")
    
    df_clean = df[~outliers].copy()
    removal_percentage = (outliers.sum() / len(df)) * 100
    
    print(f"   Outliers removidos: {outliers.sum()} ({removal_percentage:.1f}%)")
    print(f"   Datos restantes: {len(df_clean)}")
    
    return df_clean

def optimize_hyperparameters(X_train, y_train, model, model_name):
    """
    Optimiza hiperparámetros usando GridSearchCV
    
    Args:
        X_train, y_train: Datos de entrenamiento
        model: Modelo base
        model_name: Nombre del modelo
        
    Returns:
        Modelo optimizado
    """
    print(f"     🔧 Optimizando hiperparámetros para {model_name}...")
    
    param_grids = {
        'RandomForestRegressor': {
            'n_estimators': [100, 200], # Reducido para mayor velocidad en ejemplo
            'max_depth': [10, 15],      # Reducido para mayor velocidad
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        },
        'XGBRegressor': {
            'n_estimators': [100, 200], # Reducido
            'max_depth': [3, 6],        # Reducido
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        },
        'GradientBoostingRegressor': {
            'n_estimators': [100, 200], # Reducido
            'max_depth': [3, 6],        # Reducido
            'learning_rate': [0.01, 0.1],
            'subsample': [0.8, 1.0]
        },
        'SVR': {
            'C': [1, 10],               # Reducido
            'gamma': ['scale', 'auto'], # Reducido
            'epsilon': [0.01, 0.1]
        },
        'Ridge': {
            'alpha': [1, 10]            # Reducido
        },
        'LGBMRegressor': { # Añadido para LightGBM
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [20, 31],
            'max_depth': [-1, 10]
        },
        'KNeighborsRegressor': { # Añadido para KNN
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        'BayesianRidge': { # Añadido para BayesianRidge
            'alpha_1': [1e-6, 1e-5],
            'lambda_1': [1e-6, 1e-5]
        }
    }
    
    model_class_name = model.__class__.__name__
    
    if model_class_name in param_grids:
        cv = KFold(n_splits=3, shuffle=True, random_state=42) # Reducido CV para mayor velocidad
        
        try:
            grid_search = GridSearchCV(
                model, 
                param_grids[model_class_name],
                cv=cv,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            print(f"       ✅ Mejor score CV para {model_name}: {-grid_search.best_score_:.4f}")
            return grid_search.best_estimator_
        except Exception as e:
            print(f"       ❌ Error en GridSearchCV para {model_name}: {str(e)}. Usando modelo sin optimizar.")
            return model.fit(X_train, y_train) # Entrenar el modelo base si falla el grid search
    
    return model

def train_enhanced_models():
    """
    Pipeline principal de entrenamiento mejorado
    
    Returns:
        Diccionario con modelos entrenados, métricas y metadata
    """
    print("🚀 INICIANDO PIPELINE DE ENTRENAMIENTO MEJORADO")
    print("="*60)
    
    # Crear directorio para modelos
    os.makedirs('models', exist_ok=True)
    
    # 1. Cargar datos
    print("📊 Cargando dataset Boston Housing...")
    url = "https://housing-data-ml.s3.us-east-2.amazonaws.com/HousingData.csv"
    df = pd.read_csv(url)
    print(f"   📏 Dataset cargado: {df.shape}")
    
    # Validar que tenemos las 13 características + target
    expected_columns = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 
                        'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']
    
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Advertencia: Columnas faltantes en dataset original: {missing_cols}")
    
    # Usar solo las columnas que existen del set esperado
    available_cols = [col for col in expected_columns if col in df.columns]
    df = df[available_cols]
    print(f"   📊 Usando {len(available_cols)} columnas del dataset original: {available_cols}")
    
    # 2. Imputación de valores faltantes con KNN
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    joblib.dump(imputer, 'models/imputer.pkl')
    print("🔄 Imputador KNN entrenado y guardado.")

    # 3. Calcular umbrales para características binarias (ANTES de crear características)
    feature_thresholds = {}
    if 'crim' in df_imputed.columns:
        feature_thresholds['crim_quantile_75'] = df_imputed['crim'].quantile(0.75)
        print(f"   Cuantil 75 'crim' calculado: {feature_thresholds['crim_quantile_75']:.4f}")
    if 'lstat' in df_imputed.columns:
        feature_thresholds['lstat_quantile_75'] = df_imputed['lstat'].quantile(0.75)
        print(f"   Cuantil 75 'lstat' calculado: {feature_thresholds['lstat_quantile_75']:.4f}")
    if 'rm' in df_imputed.columns:
        # Esto es un umbral fijo, pero lo incluimos para consistencia en metadata
        feature_thresholds['rm_large_threshold'] = 7.0 
        print(f"   Umbral 'rm' (large_rooms) fijo: {feature_thresholds['rm_large_threshold']:.1f}")

    # 4. Ingeniería de características
    df_enhanced = create_advanced_features(df_imputed, feature_thresholds) # Pasa los umbrales
    
    # 5. Remoción de outliers
    df_clean = advanced_outlier_removal(df_enhanced, method='multi')
    
    # 6. Separar características y target
    X = df_clean.drop('medv', axis=1)
    y = df_clean['medv']
    
    print(f"📊 Datos finales después de FE y Outliers: {X.shape[0]} muestras, {X.shape[1]} características")
    
    # 7. Transformaciones de variables con sesgo
    print("🔄 Aplicando transformaciones a variables sesgadas...")
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    skewed_cols = []
    for col in numeric_cols:
        # Asegúrate de que no estás aplicando transformación a características binarias o categóricas
        if col not in ['chas', 'high_crime', 'large_rooms', 'low_status'] and abs(X[col].skew()) > 0.5:  
            skewed_cols.append(col)
    
    print(f"   📊 Columnas sesgadas detectadas para transformación: {len(skewed_cols)}")
    
    power_transformer = None
    if skewed_cols:
        power_transformer = PowerTransformer(method='yeo-johnson')
        #fit_transform devolverá un DataFrame debido a set_config
        X_transformed = X.copy() 
        X_transformed[skewed_cols] = power_transformer.fit_transform(X[skewed_cols])
        joblib.dump(power_transformer, 'models/power_transformer.pkl')
        print("   ✅ Transformación Yeo-Johnson aplicada y guardada")
    else:
        X_transformed = X.copy()
        print("   ✅ No hay columnas significativamente sesgadas para transformar.")
    
    # 8. Escalado
    print("📏 Aplicando escalado MinMax...")
    scaler = MinMaxScaler()
    # fit_transform devolverá un DataFrame debido a set_config
    X_scaled = scaler.fit_transform(X_transformed)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("   ✅ Scaler entrenado y guardado.")
    
    # 9. Selección de características con RFE
    print("🎯 Selección de características con RFECV (RandomForestRegressor)...")
    
    # Usar RandomForest como estimador base para RFE
    rf_estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # RFE con validación cruzada para encontrar número óptimo
    # rfecv devolverá un DataFrame con las características seleccionadas debido a set_config
    rfecv = RFECV(
        estimator=rf_estimator,
        step=1,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    # X_scaled es un DataFrame, y rfecv.fit_transform lo manejará y devolverá un DataFrame
    X_selected = rfecv.fit_transform(X_scaled, y)
    
    # Obtener los nombres de las características seleccionadas directamente del selector
    # rfecv.get_feature_names_out() es el método preferido si transform_output="pandas"
    # O bien, si X_scaled era un DataFrame, X_selected.columns.tolist() funciona.
    selected_features_names = X_selected.columns.tolist()

    print(f"   ✅ Características seleccionadas: {X_selected.shape[1]} de {X_scaled.shape[1]}")
    # Calcular score de RFE con las características seleccionadas
    # rfecv.estimator_ es el modelo reentrenado en el número óptimo de features
    final_estimator_for_scoring = rfecv.estimator_ 
    
    cv_scores_rfe = cross_val_score(final_estimator_for_scoring, X_selected, y, cv=5, scoring='neg_root_mean_squared_error')
    best_score_rfe = -cv_scores_rfe.mean()
    
    print(f"   📊 Mejor Score CV (RMSE) con RFE: {best_score_rfe:.4f}")
    
    joblib.dump(rfecv, 'models/feature_selector.pkl')
    print("   ✅ Selector de características (RFECV) entrenado y guardado.")

    # 10. División de datos (usando las características ya seleccionadas)
    # X_selected ahora es el DataFrame final con las columnas correctas.
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"📊 División de datos - Entrenamiento: {X_train.shape[0]} muestras, Prueba: {X_test.shape[0]} muestras")
    
    # 11. Definir modelos
    print("🤖 Configurando modelos base...")
    
    base_models = {
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'Ridge': Ridge(random_state=42),
        'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1, eval_metric='rmse'), # Add eval_metric for XGBoost
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Support Vector Regression': SVR(),
        'Extra Trees': ExtraTreesRegressor(random_state=42, n_jobs=-1),
        'KNN': KNeighborsRegressor(),
        'Bayesian Ridge': BayesianRidge()
    }
    
    # Agregar LightGBM si está disponible
    try:
        base_models['LightGBM'] = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
    except Exception:
        print("   ⚠️ LightGBM no disponible, se omitirá.")
    
    # 12. Entrenar modelos con optimización
    print("🚀 Entrenando modelos con optimización de hiperparámetros y evaluación...")
    
    trained_models = {}
    metrics = {}
    
    for model_name, model in base_models.items():
        print(f"   🔄 Entrenando {model_name}...")
        
        try:
            # Optimizar hiperparámetros
            optimized_model = optimize_hyperparameters(X_train, y_train, model, model_name)
            
            # Predicciones
            y_pred = optimized_model.predict(X_test)
            y_pred_train = optimized_model.predict(X_train)
            
            # Validación cruzada (en el conjunto de entrenamiento)
            cv_scores = cross_val_score(
                optimized_model, X_train, y_train, 
                cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1
            )
            
            # Calcular métricas
            r2_test = r2_score(y_test, y_pred)
            r2_train = r2_score(y_train, y_pred_train)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            
            # MAPE con manejo de división por cero
            mape = np.mean(np.abs((y_test - y_pred) / np.clip(y_test, 1e-8, None))) * 100
            
            metrics[model_name] = {
                'r2_test': round(r2_test, 4),
                'r2_train': round(r2_train, 4),
                'r2_cv_mean': round(-cv_scores.mean(), 4),
                'r2_cv_std': round(cv_scores.std(), 4),
                'mse': round(mse, 4),
                'rmse': round(rmse, 4),
                'mae': round(mae, 4),
                'mape': round(mape, 2),
                'overfitting': round(r2_train - r2_test, 4)
            }
            
            print(f"     ✅ R²: {r2_test:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.2f}%")
            
            # Guardar modelo entrenado y optimizado
            trained_models[model_name] = optimized_model # Almacenar en el diccionario
            safe_name = model_name.lower().replace(' ', '_')
            joblib.dump(optimized_model, f'models/{safe_name}_model.pkl')
            
        except Exception as e:
            print(f"     ❌ Error con {model_name} durante el entrenamiento o evaluación: {str(e)}")
            continue
    
    # 13. Crear ensemble si hay modelos suficientes
    print("🎭 Creando modelo ensemble VotingRegressor...")
    
    # Filtrar modelos para el ensemble: solo aquellos que se entrenaron exitosamente
    ensemble_estimators = []
    for name, model_obj in trained_models.items():
        if name != 'Voting Ensemble' and name in metrics: # Asegurarse de que el modelo se evaluó y tiene métricas
            # Usar r2_test para seleccionar modelos para el ensemble
            if metrics[name]['r2_test'] > 0.6: # Un umbral razonable para incluir en el ensemble
                ensemble_estimators.append((name.lower().replace(' ', '_'), model_obj))

    ensemble_model = None
    if len(ensemble_estimators) >= 2:
        ensemble_model = VotingRegressor(estimators=ensemble_estimators, n_jobs=-1)
        
        try:
            ensemble_model.fit(X_train, y_train)
            
            # Evaluar ensemble
            y_pred_ensemble = ensemble_model.predict(X_test)
            r2_ensemble = r2_score(y_test, y_pred_ensemble)
            rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
            mape_ensemble = np.mean(np.abs((y_test - y_pred_ensemble) / np.clip(y_test, 1e-8, None))) * 100
            
            metrics['Voting Ensemble'] = {
                'r2_test': round(r2_ensemble, 4),
                'rmse': round(rmse_ensemble, 4),
                'mape': round(mape_ensemble, 2),
                'mae': round(mean_absolute_error(y_test, y_pred_ensemble), 4)
            }
            
            print(f"   ✅ Ensemble R²: {r2_ensemble:.4f} | RMSE: {rmse_ensemble:.4f}")
            
            # Guardar ensemble
            joblib.dump(ensemble_model, 'models/voting_ensemble.pkl')
            trained_models['Voting Ensemble'] = ensemble_model
        except Exception as e:
            print(f"   ❌ Error creando o entrenando Voting Ensemble: {str(e)}")
            ensemble_model = None
    else:
        print("   ⚠️ No hay suficientes modelos con buen rendimiento para crear un Voting Ensemble.")
    
    # 14. Guardar metadata
    # **IMPORTANTE**: 'final_feature_names' ahora guarda las columnas de X_selected
    metadata = {
        'final_feature_names': selected_features_names, # Las características finales que el modelo espera
        'original_input_features': df.columns.tolist(), # Las características RAW que llegan al app.py
        'skewed_columns': skewed_cols,
        'original_shape': df.shape,
        'final_shape': X_selected.shape,
        'test_data_sample': { # Guardar una muestra pequeña o None para evitar archivos grandes
            'X_test_sample': X_test.head(5).tolist() if not X_test.empty else [],
            'y_test_sample': y_test.head(5).tolist() if not y_test.empty else []
        },
        'feature_engineering_thresholds': feature_thresholds, # Guarda los umbrales usados
        'metrics': metrics,
        'data_processing': {
            'outliers_removed_count': len(df_enhanced) - len(df_clean),
            'features_selected_count': len(selected_features_names),
            'total_features_created_count': len(df_enhanced.columns)
        }
    }
    
    joblib.dump(metadata, 'models/metadata.pkl')
    print("   ✅ Metadata guardada en models/metadata.pkl")
    
    # 15. Mostrar resumen final
    print("\n" + "="*70)
    print("🏆 RESUMEN FINAL DE ENTRENAMIENTO")
    print("="*70)
    
    if not metrics:
        print("No se generaron métricas de modelos.")
        return {
            'models': trained_models,
            'metrics': metrics,
            'metadata': metadata,
            'best_model': None
        }

    # Ordenar modelos por R²
    sorted_models = sorted(metrics.items(), key=lambda x: x[1].get('r2_test', -np.inf), reverse=True)
    
    print(f"{'Modelo':<25} {'R² Test':<10} {'RMSE':<10} {'MAPE%':<10} {'Overfitting':<12}")
    print("-" * 70)
    
    for model_name, metric in sorted_models:
        r2 = metric.get('r2_test', np.nan)
        rmse = metric.get('rmse', np.nan)
        mape = metric.get('mape', np.nan)
        overfit = metric.get('overfitting', np.nan)
        
        print(f"{model_name:<25} {r2:<10.4f} {rmse:<10.4f} {mape:<10.2f} {overfit:<12.4f}")
    
    best_model_name = sorted_models[0][0]
    best_r2 = sorted_models[0][1].get('r2_test', np.nan)
    
    print(f"\n🏆 MEJOR MODELO: {best_model_name}")
    print(f"   📊 R² Test Score: {best_r2:.4f}")
    print(f"   📈 RMSE: {sorted_models[0][1].get('rmse', np.nan):.4f}")
    print(f"   🎯 MAPE: {sorted_models[0][1].get('mape', np.nan):.2f}%")
    
    print(f"\n📊 ESTADÍSTICAS DEL PROCESAMIENTO:")
    print(f"   • Dataset original: {metadata['original_shape']}")
    print(f"   • Outliers removidos: {metadata['data_processing']['outliers_removed_count']}")
    print(f"   • Características creadas (totales): {metadata['data_processing']['total_features_created_count']}")
    print(f"   • Características seleccionadas (finales): {metadata['data_processing']['features_selected_count']}")
    print(f"   • Modelos entrenados: {len(trained_models)}")
    
    print("\n✅ Todos los modelos y archivos guardados en directorio 'models/'")
    print("🎯 ¡ENTRENAMIENTO COMPLETADO CON ÉXITO!")
    
    return {
        'models': trained_models,
        'metrics': metrics,
        'metadata': metadata,
        'best_model': best_model_name
    }

# Función auxiliar para evaluar modelos después del entrenamiento
def evaluate_models():
    """
    Evalúa todos los modelos entrenados y muestra métricas detalladas
    """
    try:
        metadata = joblib.load('models/metadata.pkl')
        metrics = metadata['metrics']
        
        print("\n📊 EVALUACIÓN DETALLADA DE MODELOS")
        print("="*50)
        
        if not metrics:
            print("No se encontraron métricas en el archivo metadata.pkl.")
            return

        for model_name, model_metrics in metrics.items():
            print(f"\n🤖 {model_name.upper()}")
            print("-" * 40)
            print(f"   R² Test: {model_metrics.get('r2_test', np.nan):.4f}")
            if 'r2_train' in model_metrics:
                print(f"   R² Train: {model_metrics.get('r2_train', np.nan):.4f}")
            if 'r2_cv_mean' in model_metrics:
                print(f"   R² CV: {model_metrics.get('r2_cv_mean', np.nan):.4f} ± {model_metrics.get('r2_cv_std', np.nan):.4f}")
            print(f"   RMSE: {model_metrics.get('rmse', np.nan):.4f}")
            print(f"   MAE: {model_metrics.get('mae', np.nan):.4f}")
            print(f"   MAPE: {model_metrics.get('mape', np.nan):.2f}%")
            if 'overfitting' in model_metrics:
                overfitting_value = model_metrics.get('overfitting', np.nan)
                overfitting_status = "🔴 Alto" if overfitting_value > 0.1 else "🟢 Bajo"
                print(f"   Overfitting: {overfitting_status} ({overfitting_value:.4f})")
        
    except FileNotFoundError:
        print("❌ No se encontraron modelos entrenados o metadata. Ejecuta train_enhanced_models() primero.")
    except Exception as e:
        print(f"❌ Error al evaluar modelos: {str(e)}")

# Función principal
def main():
    """
    Ejecuta el pipeline completo de entrenamiento
    """
    print("🚀 INICIANDO PIPELINE COMPLETO")
    print("="*40)
    
    try:
        results = train_enhanced_models()
        
        print("\n🎯 Pipeline completado exitosamente!")
        print("💾 Todos los archivos guardados en directorio 'models/'")
        
        # Opcional: Ejecutar evaluación detallada después del entrenamiento
        # evaluate_models() 
        
        return results
        
    except Exception as e:
        print(f"❌ Error en pipeline: {str(e)}")
        return None

if __name__ == "__main__":
    main()
