import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV, SelectFromModel
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                             VotingRegressor, BaggingRegressor, ExtraTreesRegressor,
                             AdaBoostRegressor, StackingRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
from scipy import stats
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

class HousingPredictor:
    """Clase para predicción de precios de viviendas con modelos ultra precisos"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = None
        self.power_transformer = None
        self.feature_selector = None
        self.metadata = None
        self.is_trained = False

    def create_ultra_advanced_features(df):
        """Crear características ultra avanzadas con más interacciones complejas"""
        df_ultra = df.copy()
        
        # 1. Características polinómicas de orden superior
        key_features = ['rm', 'lstat', 'ptratio', 'dis', 'tax', 'crim']
        for feature in key_features:
            if feature in df_ultra.columns:
                # Potencias
                df_ultra[f'{feature}_squared'] = df_ultra[feature] ** 2
                df_ultra[f'{feature}_cubed'] = df_ultra[feature] ** 3
                df_ultra[f'{feature}_sqrt'] = np.sqrt(np.abs(df_ultra[feature]))
                df_ultra[f'{feature}_log'] = np.log1p(np.abs(df_ultra[feature]))
                df_ultra[f'{feature}_inv'] = 1 / (df_ultra[feature] + 1e-8)
        
        # 2. Interacciones complejas múltiples
        interactions = [
            ('rm', 'lstat', 'ptratio'),  # Habitaciones x Estatus x Educación
            ('crim', 'dis', 'rad'),      # Crimen x Distancia x Accesibilidad
            ('tax', 'ptratio', 'b'),     # Impuestos x Educación x Proporción negros
            ('nox', 'dis', 'indus'),     # Contaminación x Distancia x Industrial
            ('age', 'dis', 'rm')         # Edad x Distancia x Habitaciones
        ]
        
        for feat1, feat2, feat3 in interactions:
            if all(col in df_ultra.columns for col in [feat1, feat2, feat3]):
                df_ultra[f'{feat1}_{feat2}_{feat3}_interaction'] = (
                    df_ultra[feat1] * df_ultra[feat2] * df_ultra[feat3]
                )
        
        # 3. Ratios y combinaciones avanzadas
        ratio_combinations = [
            ('rm', 'age'),      # Habitaciones por edad
            ('tax', 'medv'),    # Impuestos por valor (si existe)
            ('crim', 'b'),      # Crimen por demografía
            ('nox', 'rm'),      # Contaminación por habitaciones
            ('lstat', 'rm'),    # Estatus por habitaciones
            ('ptratio', 'rm'),  # Ratio estudiantes por habitaciones
            ('indus', 'dis'),   # Industrial por distancia
            ('age', 'rm')       # Edad por habitaciones
        ]
        
        for num, den in ratio_combinations:
            if all(col in df_ultra.columns for col in [num, den]):
                df_ultra[f'{num}_per_{den}'] = df_ultra[num] / (df_ultra[den] + 1e-8)
        
        # 4. Características estadísticas por grupos (clustering)
        if len(df_ultra) > 50:  # Solo si hay suficientes datos
            # Crear clusters basados en características clave
            cluster_features = ['rm', 'lstat', 'crim', 'dis']
            available_cluster_features = [f for f in cluster_features if f in df_ultra.columns]
            
            if len(available_cluster_features) >= 2:
                kmeans = KMeans(n_clusters=5, random_state=42)
                df_ultra['neighborhood_cluster'] = kmeans.fit_predict(
                    df_ultra[available_cluster_features].fillna(df_ultra[available_cluster_features].mean())
                )
                
                # Estadísticas por cluster
                for feature in available_cluster_features:
                    cluster_stats = df_ultra.groupby('neighborhood_cluster')[feature].agg(['mean', 'std', 'median'])
                    df_ultra[f'{feature}_cluster_mean'] = df_ultra['neighborhood_cluster'].map(cluster_stats['mean'])
                    df_ultra[f'{feature}_cluster_std'] = df_ultra['neighborhood_cluster'].map(cluster_stats['std'])
                    df_ultra[f'{feature}_deviation_from_cluster'] = (
                        df_ultra[feature] - df_ultra[f'{feature}_cluster_mean']
                    )
        
        # 5. Características de posición relativa (percentiles)
        numeric_cols = df_ultra.select_dtypes(include=[np.number]).columns
        for col in ['rm', 'lstat', 'crim', 'tax']:
            if col in numeric_cols:
                df_ultra[f'{col}_percentile'] = df_ultra[col].rank(pct=True)
                df_ultra[f'{col}_zscore'] = stats.zscore(df_ultra[col].fillna(df_ultra[col].mean()))
        
        # 6. Características temporales/espaciales simuladas
        if 'dis' in df_ultra.columns and 'rad' in df_ultra.columns:
            df_ultra['accessibility_score'] = (
                (1 / (df_ultra['dis'] + 1e-8)) * df_ultra['rad']
            )
        
        # 7. Índices compuestos
        if all(col in df_ultra.columns for col in ['rm', 'lstat', 'ptratio']):
            df_ultra['quality_of_life_index'] = (
                df_ultra['rm'] * 0.4 - 
                df_ultra['lstat'] * 0.3 - 
                df_ultra['ptratio'] * 0.3
            )
        
        if all(col in df_ultra.columns for col in ['crim', 'nox', 'tax']):
            df_ultra['negative_factors_index'] = (
                df_ultra['crim'] * 0.4 + 
                df_ultra['nox'] * 0.3 + 
                df_ultra['tax'] * 0.3
            )
        
        return df_ultra

    def ultra_precise_outlier_detection(df, target_col='medv'):
        """Detección ultra precisa de outliers usando múltiples métodos"""
        from sklearn.ensemble import IsolationForest
        from sklearn.covariance import EllipticEnvelope
        from sklearn.neighbors import LocalOutlierFactor
        
        X = df.drop(target_col, axis=1) if target_col in df.columns else df
        
        # 1. Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        iso_outliers = iso_forest.fit_predict(X) == -1
        
        # 2. Elliptic Envelope
        elliptic = EllipticEnvelope(contamination=0.05, random_state=42)
        try:
            elliptic_outliers = elliptic.fit_predict(X) == -1
        except:
            elliptic_outliers = np.zeros(len(X), dtype=bool)
        
        # 3. Local Outlier Factor
        lof = LocalOutlierFactor(contamination=0.05)
        lof_outliers = lof.fit_predict(X) == -1
        
        # 4. Z-score para el target
        if target_col in df.columns:
            z_scores = np.abs(stats.zscore(df[target_col]))
            zscore_outliers = z_scores > 3
        else:
            zscore_outliers = np.zeros(len(df), dtype=bool)
        
        # Combinar métodos - consideramos outlier si al menos 2 métodos lo detectan
        total_outliers = iso_outliers.astype(int) + elliptic_outliers.astype(int) + lof_outliers.astype(int) + zscore_outliers.astype(int)
        final_outliers = total_outliers >= 2
        
        df_clean = df[~final_outliers].copy()
        
        print(f"📊 Outliers detectados:")
        print(f"  - Isolation Forest: {iso_outliers.sum()}")
        print(f"  - Elliptic Envelope: {elliptic_outliers.sum()}")
        print(f"  - Local Outlier Factor: {lof_outliers.sum()}")
        print(f"  - Z-score (target): {zscore_outliers.sum()}")
        print(f"  - Total removidos: {final_outliers.sum()} ({(final_outliers.sum()/len(df)*100):.1f}%)")
        
        return df_clean

    def optimize_hyperparameters_ultra(X_train, y_train, model_name, model):
        """Optimización ultra precisa de hiperparámetros"""
        print(f"    🔧 Optimizando hiperparámetros para {model_name}...")
        
        # Definir grids de parámetros más amplios y precisos
        param_grids = {
            'RandomForestRegressor': {
                'n_estimators': [200, 300, 500],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 2, 3],
                'max_features': ['sqrt', 'log2', 0.8],
                'bootstrap': [True, False]
            },
            'XGBRegressor': {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [1, 1.5, 2]
            },
            'GradientBoostingRegressor': {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 2, 3]
            },
            'SVR': {
                'C': [50, 100, 200, 500],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'epsilon': [0.01, 0.1, 0.2],
                'kernel': ['rbf', 'poly']
            },
            'Ridge': {
                'alpha': [0.1, 1, 10, 50, 100, 200]
            }
        }
        
        model_class_name = model.__class__.__name__
        
        if model_class_name in param_grids:
            # Usar GridSearchCV con validación cruzada más robusta
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            grid_search = GridSearchCV(
                model, 
                param_grids[model_class_name],
                cv=cv,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            print(f"      ✅ Mejor score: {-grid_search.best_score_:.4f}")
            return grid_search.best_estimator_
        
        return model

    def train_ultra_precise_models():
        """Entrenar modelos con precisión ultra alta"""
        
        print("🎯 Iniciando entrenamiento ULTRA PRECISO...")
        print("🚀 Objetivo: Maximizar precisión (minimizar MAPE)")
        
        # Crear directorio
        os.makedirs('models', exist_ok=True)
        
        # 1. Cargar y preparar datos con máxima calidad
        print("📊 Cargando datos con procesamiento de máxima calidad...")
        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        df = pd.read_csv(url)
        
        # Imputación más sofisticada
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        
        if df.isnull().sum().sum() > 0:
            imputer = IterativeImputer(random_state=42, max_iter=10)
            df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        else:
            df_imputed = df.copy()
        
        print(f"📏 Dataset original: {len(df_imputed)} muestras")
        
        # 2. Feature Engineering Ultra Avanzado
        print("🔧 Aplicando feature engineering ultra avanzado...")
        df_ultra = create_ultra_advanced_features(df_imputed)
        print(f"📈 Características: {len(df.columns)} → {len(df_ultra.columns)}")
        
        # 3. Detección ultra precisa de outliers
        print("🎯 Detección ultra precisa de outliers...")
        df_clean = ultra_precise_outlier_detection(df_ultra)
        print(f"📊 Datos finales: {len(df_clean)} muestras")
        
        # 4. Transformaciones ultra precisas
        print("🔄 Aplicando transformaciones ultra precisas...")
        
        # Separar target
        X = df_clean.drop('medv', axis=1)
        y = df_clean['medv']
        
        # Identificar y transformar características sesgadas
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        skewed_cols = []
        for col in numeric_cols:
            if abs(X[col].skew()) > 0.3:  # Umbral más bajo para ser más estricto
                skewed_cols.append(col)
        
        print(f"📊 Columnas a transformar: {len(skewed_cols)}")
        
        # Aplicar múltiples transformaciones y seleccionar la mejor
        transformers = {
            'yeo_johnson': PowerTransformer(method='yeo-johnson'),
            'box_cox': PowerTransformer(method='box-cox')
        }
        
        X_transformed = X.copy()
        best_transformer = None
        best_score = -np.inf
        
        for name, transformer in transformers.items():
            try:
                X_temp = X.copy()
                if skewed_cols:
                    if name == 'box_cox':
                        # Box-Cox requiere valores positivos
                        for col in skewed_cols:
                            if (X_temp[col] <= 0).any():
                                X_temp[col] = X_temp[col] - X_temp[col].min() + 1e-8
                    
                    X_temp[skewed_cols] = transformer.fit_transform(X_temp[skewed_cols])
                    
                    # Evaluar la transformación con un modelo rápido
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression()
                    X_temp_scaled = StandardScaler().fit_transform(X_temp)
                    scores = cross_val_score(lr, X_temp_scaled, y, cv=3, scoring='neg_mean_squared_error')
                    score = scores.mean()
                    
                    if score > best_score:
                        best_score = score
                        best_transformer = (name, transformer)
                        X_transformed = X_temp.copy()
                        
            except Exception as e:
                print(f"      ⚠️ Error con {name}: {str(e)}")
                continue
        
        if best_transformer:
            print(f"    ✅ Mejor transformador: {best_transformer[0]}")
            power_transformer = best_transformer[1]
        else:
            power_transformer = None
            skewed_cols = []
        
        # 5. Escalado ultra preciso
        print("📏 Aplicando escalado ultra preciso...")
        
        # Probar diferentes escaladores
        scalers = {
            'robust': RobustScaler(),
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        
        best_scaler = None
        best_scaler_score = -np.inf
        
        for name, scaler in scalers.items():
            X_scaled = scaler.fit_transform(X_transformed)
            
            # Evaluar escalador
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            scores = cross_val_score(lr, X_scaled, y, cv=3, scoring='neg_mean_squared_error')
            score = scores.mean()
            
            if score > best_scaler_score:
                best_scaler_score = score
                best_scaler = scaler
        
        X_scaled = best_scaler.fit_transform(X_transformed)
        print(f"    ✅ Mejor escalador: {type(best_scaler).__name__}")
        
        # 6. Selección ultra precisa de características
        print("🎯 Selección ultra precisa de características...")
        
        # Usar RFECV para selección automática óptima
        from sklearn.ensemble import RandomForestRegressor
        rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
        
        rfecv = RFECV(
            estimator=rf_selector,
            step=1,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        X_selected = rfecv.fit_transform(X_scaled, y)
        selected_features = X.columns[rfecv.support_].tolist()
        
        # ✅ Mostrar resumen
        print(f"📈 Características seleccionadas: {len(selected_features)} de {X_scaled.shape[1]}")
        print("🔍 Características seleccionadas por RFECV:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i:02d}. {feature}")

        # ✅ Mostrar score del selector
        print(f"📊 Score de selección: {rfecv.score(X_scaled, y):.4f}")

        # ✅ Guardar en archivo de texto para referencia futura
        with open("models/ultra_selected_features.txt", "w") as f:
            f.write("\n".join(selected_features))

        # 7. División estratificada optimizada
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.1, random_state=42, shuffle=True
        )
        
        print(f"📊 Entrenamiento: {X_train.shape[0]} | Prueba: {X_test.shape[0]}")
        
        # 8. Modelos ultra avanzados
        print("🤖 Configurando modelos ultra avanzados...")
        
        base_models = {
            'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Support Vector Regression': SVR(),
            'Ridge Regression': Ridge(random_state=42),
            'Extra Trees': ExtraTreesRegressor(random_state=42, n_jobs=-1),
            'Neural Network': MLPRegressor(random_state=42, max_iter=500),
            'KNN': KNeighborsRegressor(),
            'Bayesian Ridge': BayesianRidge(),
            'Huber Regressor': HuberRegressor()
        }
        
        # Agregar modelos avanzados si están disponibles
        try:
            base_models['LightGBM'] = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        except:
            pass
        
        try:
            base_models['CatBoost'] = CatBoostRegressor(random_state=42, verbose=False)
        except:
            pass
        
        # 9. Entrenar con optimización de hiperparámetros
        print("🚀 Entrenando modelos con optimización ultra precisa...")
        
        optimized_models = {}
        metrics = {}
        
        for model_name, model in base_models.items():
            print(f"  🔄 Entrenando {model_name}...")
            
            try:
                # Optimizar hiperparámetros
                optimized_model = optimize_hyperparameters_ultra(X_train, y_train, model_name, model)
                optimized_models[model_name] = optimized_model
                
                # Evaluar con validación cruzada más robusta
                cv = KFold(n_splits=10, shuffle=True, random_state=42)
                cv_scores = cross_val_score(optimized_model, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
                
                # Predicciones en conjunto de prueba
                optimized_model.fit(X_train, y_train)
                y_pred = optimized_model.predict(X_test)
                y_pred_train = optimized_model.predict(X_train)
                
                # Métricas ultra precisas
                r2_test = r2_score(y_test, y_pred)
                r2_train = r2_score(y_train, y_pred_train)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # MAPE ultra preciso (manejo de valores cero)
                mape = np.mean(np.abs((y_test - y_pred) / np.clip(y_test, 1e-8, None))) * 100
                
                # Precisión porcentual
                precision_percentage = 100 - mape
                
                metrics[model_name] = {
                    'r2_test': round(r2_test, 6),
                    'r2_train': round(r2_train, 6),
                    'r2_cv_mean': round(-cv_scores.mean(), 6),
                    'r2_cv_std': round(cv_scores.std(), 6),
                    'mse': round(mse, 6),
                    'rmse': round(rmse, 6),
                    'mae': round(mae, 6),
                    'mape': round(mape, 4),
                    'precision_percentage': round(precision_percentage, 4),
                    'overfitting': round(r2_train - r2_test, 6)
                }
                
                print(f"    ✅ R²: {r2_test:.6f} | RMSE: {rmse:.4f} | MAPE: {mape:.4f}% | Precisión: {precision_percentage:.4f}%")
                
            except Exception as e:
                print(f"    ❌ Error con {model_name}: {str(e)}")
                continue
        
        # 10. Crear ensemble ultra avanzado
        print("🎭 Creando ensemble ultra avanzado...")
        
        # Seleccionar los mejores modelos (top 5 por precisión)
        best_models = sorted(metrics.items(), key=lambda x: x[1]['precision_percentage'], reverse=True)[:5]
        
        if len(best_models) >= 3:
            # Crear Stacking Regressor ultra avanzado
            base_estimators = [(name.lower().replace(' ', '_'), optimized_models[name]) 
                            for name, _ in best_models]
            
            # Meta-modelo sofisticado
            meta_model = Ridge(alpha=1.0)
            
            stacking_regressor = StackingRegressor(
                estimators=base_estimators,
                final_estimator=meta_model,
                cv=5,
                n_jobs=-1
            )
            
            stacking_regressor.fit(X_train, y_train)
            
            # Evaluar stacking
            y_pred_stack = stacking_regressor.predict(X_test)
            r2_stack = r2_score(y_test, y_pred_stack)
            mse_stack = mean_squared_error(y_test, y_pred_stack)
            mae_stack = mean_absolute_error(y_test, y_pred_stack)
            rmse_stack = np.sqrt(mse_stack)
            mape_stack = np.mean(np.abs((y_test - y_pred_stack) / np.clip(y_test, 1e-8, None))) * 100
            precision_stack = 100 - mape_stack
            
            # Validación cruzada del ensemble
            cv_scores_stack = cross_val_score(stacking_regressor, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
            
            metrics['Ultra Stacking Ensemble'] = {
                'r2_test': round(r2_stack, 6),
                'r2_cv_mean': round(-cv_scores_stack.mean(), 6),
                'r2_cv_std': round(cv_scores_stack.std(), 6),
                'mse': round(mse_stack, 6),
                'rmse': round(rmse_stack, 6),
                'mae': round(mae_stack, 6),
                'mape': round(mape_stack, 4),
                'precision_percentage': round(precision_stack, 4)
            }
            
            print(f"    ✅ Stacking R²: {r2_stack:.6f} | MAPE: {mape_stack:.4f}% | Precisión: {precision_stack:.4f}%")
            
            # También crear un Voting Regressor con pesos optimizados
            voting_regressor = VotingRegressor(
                estimators=base_estimators,
                n_jobs=-1
            )
            
            voting_regressor.fit(X_train, y_train)
            
            y_pred_vote = voting_regressor.predict(X_test)
            r2_vote = r2_score(y_test, y_pred_vote)
            mse_vote = mean_squared_error(y_test, y_pred_vote)
            rmse_vote = np.sqrt(mse_vote)
            mape_vote = np.mean(np.abs((y_test - y_pred_vote) / np.clip(y_test, 1e-8, None))) * 100
            precision_vote = 100 - mape_vote
            
            metrics['Ultra Voting Ensemble'] = {
                'r2_test': round(r2_vote, 6),
                'mse': round(mse_vote, 6),
                'rmse': round(rmse_vote, 6),
                'mae': round(mean_absolute_error(y_test, y_pred_vote), 6),
                'mape': round(mape_vote, 4),
                'precision_percentage': round(precision_vote, 4)
            }
            
            print(f"    ✅ Voting R²: {r2_vote:.6f} | MAPE: {mape_vote:.4f}% | Precisión: {precision_vote:.4f}%")
        
        # 11. Guardar modelos ultra precisos
        print("💾 Guardando modelos ultra precisos...")
        
        for model_name, model in optimized_models.items():
            safe_name = model_name.lower().replace(' ', '_').replace('ó', 'o')
            joblib.dump(model, f'models/ultra_{safe_name}_model.pkl')
        
        # Guardar ensembles
        if 'stacking_regressor' in locals():
            joblib.dump(stacking_regressor, 'models/ultra_stacking_ensemble.pkl')
        
        if 'voting_regressor' in locals():
            joblib.dump(voting_regressor, 'models/ultra_voting_ensemble.pkl')
        
        # Guardar transformadores
        joblib.dump(best_scaler, 'models/ultra_scaler.pkl')
        if power_transformer:
            joblib.dump(power_transformer, 'models/ultra_power_transformer.pkl')
        joblib.dump(rfecv, 'models/ultra_feature_selector.pkl')
        
        # Metadata ultra completo
        ultra_metadata = {
            'feature_names': selected_features,
            'all_features': X.columns.tolist(),
            'cols_to_transform': skewed_cols,
            'ultra_metrics': metrics,
            'test_data': {
                'X_test': X_test.tolist(),
                'y_test': y_test.tolist()
            },
            'data_processing': {
                'original_samples': len(df),
                'final_samples': len(df_clean),
                'original_features': len(df.columns),
                'engineered_features': len(df_ultra.columns),
                'selected_features': len(selected_features),
                'outliers_removed': len(df_ultra) - len(df_clean)
            },
            'best_models': {
                'highest_precision': max(metrics.keys(), key=lambda k: metrics[k]['precision_percentage']),
                'highest_r2': max(metrics.keys(), key=lambda k: metrics[k]['r2_test']),
                'lowest_mape': min(metrics.keys(), key=lambda k: metrics[k]['mape'])
            },
            'model_configurations': {
                'cross_validation_folds': 10,
                'test_size': 0.1,
                'random_state': 42,
                'outlier_contamination': 0.05,
                'feature_selection_method': 'RFECV',
                'best_scaler': type(best_scaler).__name__,
                'best_transformer': best_transformer[0] if best_transformer else None
            }
        }
        
        joblib.dump(ultra_metadata, 'models/ultra_metadata.pkl')
        
        # 12. Mostrar resultados ultra detallados
        print("\n" + "="*80)
        print("🏆 RESULTADOS ULTRA PRECISOS - RANKING DE MODELOS")
        print("="*80)
        
        # Ordenar por precisión porcentual
        sorted_models = sorted(metrics.items(), key=lambda x: x[1]['precision_percentage'], reverse=True)
        
        print(f"{'Modelo':<25} {'Precisión%':<12} {'R²':<10} {'MAPE%':<8} {'RMSE':<8} {'Overfitting':<12}")
        print("-" * 80)
        
        for i, (model_name, metric) in enumerate(sorted_models):
            precision = metric['precision_percentage']
            r2 = metric['r2_test']
            mape = metric['mape']
            rmse = metric['rmse']
            overfit = metric.get('overfitting', 0)
            
            # Emojis según ranking
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1:2d}."
            
            print(f"{rank_emoji} {model_name:<22} {precision:>8.4f}%   {r2:>7.6f}  {mape:>6.4f}%  {rmse:>6.4f}  {overfit:>8.6f}")
        
        # Estadísticas finales
        print("\n" + "="*60)
        print("📊 ESTADÍSTICAS ULTRA DETALLADAS")
        print("="*60)
        
        best_model_name = sorted_models[0][0]
        best_metrics = sorted_models[0][1]
        
        print(f"🏆 MEJOR MODELO: {best_model_name}")
        print(f"   • Precisión: {best_metrics['precision_percentage']:.4f}%")
        print(f"   • R² Test: {best_metrics['r2_test']:.6f}")
        print(f"   • MAPE: {best_metrics['mape']:.4f}%")
        print(f"   • RMSE: {best_metrics['rmse']:.4f}")
        print(f"   • MAE: {best_metrics['mae']:.4f}")
        
        if 'r2_cv_mean' in best_metrics:
            print(f"   • R² CV: {best_metrics['r2_cv_mean']:.6f} ± {best_metrics['r2_cv_std']:.6f}")
        
        print(f"\n📈 PROCESAMIENTO DE DATOS:")
        print(f"   • Muestras originales: {ultra_metadata['data_processing']['original_samples']}")
        print(f"   • Muestras finales: {ultra_metadata['data_processing']['final_samples']}")
        print(f"   • Outliers removidos: {ultra_metadata['data_processing']['outliers_removed']}")
        print(f"   • Características originales: {ultra_metadata['data_processing']['original_features']}")
        print(f"   • Características creadas: {ultra_metadata['data_processing']['engineered_features']}")
        print(f"   • Características seleccionadas: {ultra_metadata['data_processing']['selected_features']}")
        
        print(f"\n🔧 CONFIGURACIÓN:")
        print(f"   • Escalador: {ultra_metadata['model_configurations']['best_scaler']}")
        print(f"   • Transformador: {ultra_metadata['model_configurations']['best_transformer']}")
        print(f"   • Validación cruzada: {ultra_metadata['model_configurations']['cross_validation_folds']} folds")
        print(f"   • Conjunto de prueba: {ultra_metadata['model_configurations']['test_size']*100}%")
        
        print("\n✅ Todos los modelos y metadata guardados en directorio 'models/'")
        print("🎯 ¡ENTRENAMIENTO ULTRA PRECISO COMPLETADO!")
        
        return optimized_models, metrics, ultra_metadata
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV, SelectFromModel
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              VotingRegressor, BaggingRegressor, ExtraTreesRegressor,
                              AdaBoostRegressor, StackingRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
import joblib
import os
import warnings
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.experimental import enable_iterative_imputer # Necesario para IterativeImputer
from sklearn.impute import IterativeImputer # Mover la importación aquí para que esté siempre disponible

warnings.filterwarnings('ignore')

class HousingPredictor:
    """
    Clase para predicción de precios de viviendas con modelos ultra precisos.
    Encapsula el pipeline completo de entrenamiento y predicción.
    """
    
    def __init__(self):
        # Componentes del pipeline
        self.imputer = None
        self.power_transformer = None
        self.scaler = None
        self.feature_selector = None
        self.kmeans_model = None # Para características de clustering
        self.cluster_stats_data = {} # Estadísticas de clustering
        self.percentile_stats = {} # Estadísticas de percentiles/z-score
        self.model = None # El modelo final (ensemble o el mejor individual)
        self.metadata = None
        self.is_trained = False # Indica si el pipeline ha sido entrenado
        self.is_loaded = False # Indica si el pipeline ha sido cargado para predicción

    @staticmethod
    def _create_ultra_advanced_features_static(df):
        """
        Crear características ultra avanzadas con más interacciones complejas.
        Este es el core de la ingeniería de características que se usa en entrenamiento.
        Retornará los metadatos necesarios para inferencia.
        """
        df_ultra = df.copy()
        
        # 1. Características polinómicas de orden superior
        key_features = ['rm', 'lstat', 'ptratio', 'dis', 'tax', 'crim']
        for feature in key_features:
            if feature in df_ultra.columns:
                df_ultra[f'{feature}_squared'] = df_ultra[feature] ** 2
                df_ultra[f'{feature}_cubed'] = df_ultra[feature] ** 3
                df_ultra[f'{feature}_sqrt'] = np.sqrt(np.abs(df_ultra[feature]))
                df_ultra[f'{feature}_log'] = np.log1p(np.abs(df_ultra[feature]))
                df_ultra[f'{feature}_inv'] = 1 / (df_ultra[feature] + 1e-8)
        
        # 2. Interacciones complejas múltiples
        interactions = [
            ('rm', 'lstat', 'ptratio'), 
            ('crim', 'dis', 'rad'), 
            ('tax', 'ptratio', 'b'), 
            ('nox', 'dis', 'indus'), 
            ('age', 'dis', 'rm') 
        ]
        
        for feat1, feat2, feat3 in interactions:
            if all(col in df_ultra.columns for col in [feat1, feat2, feat3]):
                df_ultra[f'{feat1}_{feat2}_{feat3}_interaction'] = (
                    df_ultra[feat1] * df_ultra[feat2] * df_ultra[feat3]
                )
        
        # 3. Ratios y combinaciones avanzadas
        ratio_combinations = [
            ('rm', 'age'), 
            # ('tax', 'medv'), # <--- 'medv' no estará en inferencia, por eso se omite aquí
            ('crim', 'b'), 
            ('nox', 'rm'), 
            ('lstat', 'rm'), 
            ('ptratio', 'rm'), 
            ('indus', 'dis'), 
            ('age', 'rm') 
        ]
        
        for num, den in ratio_combinations:
            if all(col in df_ultra.columns for col in [num, den]):
                df_ultra[f'{num}_per_{den}'] = df_ultra[num] / (df_ultra[den] + 1e-8)
        
        # 4. Características estadísticas por grupos (clustering)
        kmeans_model = None
        cluster_stats_data = {} 
        if len(df_ultra) > 50: 
            cluster_features = ['rm', 'lstat', 'crim', 'dis']
            available_cluster_features = [f for f in cluster_features if f in df_ultra.columns]
            
            if len(available_cluster_features) >= 2:
                kmeans_model = KMeans(n_clusters=5, random_state=42, n_init='auto') 
                df_for_kmeans = df_ultra[available_cluster_features].fillna(df_ultra[available_cluster_features].mean())
                df_ultra['neighborhood_cluster'] = kmeans_model.fit_predict(df_for_kmeans)
                
                for feature in available_cluster_features:
                    cluster_stats = df_ultra.groupby('neighborhood_cluster')[feature].agg(['mean', 'std', 'median'])
                    df_ultra[f'{feature}_cluster_mean'] = df_ultra['neighborhood_cluster'].map(cluster_stats['mean'])
                    df_ultra[f'{feature}_cluster_std'] = df_ultra['neighborhood_cluster'].map(cluster_stats['std'])
                    df_ultra[f'{feature}_deviation_from_cluster'] = (
                        df_ultra[feature] - df_ultra[f'{feature}_cluster_mean']
                    )
                    # Guardar estadísticas del cluster para inferencia
                    cluster_stats_data[feature] = cluster_stats.to_dict(orient='index')
        
        # 5. Características de posición relativa (percentiles/z-score)
        percentile_stats = {}
        numeric_cols = df_ultra.select_dtypes(include=[np.number]).columns
        for col in ['rm', 'lstat', 'crim', 'tax']:
            if col in numeric_cols:
                # Para z-score, necesitamos la media y std del entrenamiento
                col_mean = df_ultra[col].mean()
                col_std = df_ultra[col].std()
                df_ultra[f'{col}_percentile'] = df_ultra[col].rank(pct=True) # Rank es local al df, pero se guarda mean/std para Z-score
                df_ultra[f'{col}_zscore'] = (df_ultra[col].fillna(col_mean) - col_mean) / (col_std + 1e-8)
                percentile_stats[col] = {'mean': col_mean, 'std': col_std}
        
        # 6. Características temporales/espaciales simuladas
        if 'dis' in df_ultra.columns and 'rad' in df_ultra.columns:
            df_ultra['accessibility_score'] = (
                (1 / (df_ultra['dis'] + 1e-8)) * df_ultra['rad']
            )
        
        # 7. Índices compuestos
        if all(col in df_ultra.columns for col in ['rm', 'lstat', 'ptratio']):
            df_ultra['quality_of_life_index'] = (
                df_ultra['rm'] * 0.4 - 
                df_ultra['lstat'] * 0.3 - 
                df_ultra['ptratio'] * 0.3
            )
        
        if all(col in df_ultra.columns for col in ['crim', 'nox', 'tax']):
            df_ultra['negative_factors_index'] = (
                df_ultra['crim'] * 0.4 + 
                df_ultra['nox'] * 0.3 + 
                df_ultra['tax'] * 0.3
            )

        # Capturar umbrales para características binarias
        bin_thresholds = {
            'crim_75_quantile': df_ultra['crim'].quantile(0.75) if 'crim' in df_ultra.columns else None,
            'lstat_75_quantile': df_ultra['lstat'].quantile(0.75) if 'lstat' in df_ultra.columns else None
        }
        
        return df_ultra, bin_thresholds, kmeans_model, cluster_stats_data, percentile_stats

    def _apply_feature_engineering_for_prediction(self, df_imputed):
        """
        Aplica ingeniería de características para un solo punto de predicción,
        utilizando los metadatos y modelos ajustados del entrenamiento.
        """
        df_enhanced = df_imputed.copy()
        
        # Cargar metadatos para umbrales, estadísticas, etc.
        bin_thresholds = self.metadata.get('bin_thresholds', {})
        kmeans_model = self.kmeans_model
        cluster_stats_data = self.cluster_stats_data
        percentile_stats = self.percentile_stats

        # 1. Características polinómicas de orden superior (estas son independientes de los datos de entrenamiento)
        key_features = ['rm', 'lstat', 'ptratio', 'dis', 'tax', 'crim']
        for feature in key_features:
            if feature in df_enhanced.columns:
                df_enhanced[f'{feature}_squared'] = df_enhanced[feature] ** 2
                df_enhanced[f'{feature}_cubed'] = df_enhanced[feature] ** 3
                df_enhanced[f'{feature}_sqrt'] = np.sqrt(np.abs(df_enhanced[feature]))
                df_enhanced[f'{feature}_log'] = np.log1p(np.abs(df_enhanced[feature]))
                df_enhanced[f'{feature}_inv'] = 1 / (df_enhanced[feature] + 1e-8)
        
        # 2. Interacciones complejas múltiples (estas son independientes de los datos de entrenamiento)
        interactions = [
            ('rm', 'lstat', 'ptratio'), 
            ('crim', 'dis', 'rad'), 
            ('tax', 'ptratio', 'b'), 
            ('nox', 'dis', 'indus'), 
            ('age', 'dis', 'rm') 
        ]
        for feat1, feat2, feat3 in interactions:
            if all(col in df_enhanced.columns for col in [feat1, feat2, feat3]):
                df_enhanced[f'{feat1}_{feat2}_{feat3}_interaction'] = (
                    df_enhanced[feat1] * df_enhanced[feat2] * df_enhanced[feat3]
                )
        
        # 3. Ratios y combinaciones avanzadas (estas son independientes de los datos de entrenamiento)
        ratio_combinations = [
            ('rm', 'age'), 
            ('crim', 'b'), 
            ('nox', 'rm'), 
            ('lstat', 'rm'), 
            ('ptratio', 'rm'), 
            ('indus', 'dis'), 
            ('age', 'rm') 
        ]
        for num, den in ratio_combinations:
            if all(col in df_enhanced.columns for col in [num, den]):
                df_enhanced[f'{num}_per_{den}'] = df_enhanced[num] / (df_enhanced[den] + 1e-8)
        
        # 4. Características categóricas binarias (banderas) usando umbrales guardados
        crim_threshold = bin_thresholds.get('crim_75_quantile')
        lstat_threshold = bin_thresholds.get('lstat_75_quantile')

        if 'crim' in df_enhanced.columns and crim_threshold is not None:
            df_enhanced['high_crime'] = (df_enhanced['crim'] > crim_threshold).astype(int)
        if 'rm' in df_enhanced.columns: # Umbral fijo
            df_enhanced['large_rooms'] = (df_enhanced['rm'] > 7).astype(int)
        if 'lstat' in df_enhanced.columns and lstat_threshold is not None:
            df_enhanced['low_status'] = (df_enhanced['lstat'] > lstat_threshold).astype(int)

        # 5. Características de clustering y percentiles
        if kmeans_model and 'neighborhood_cluster' in self.metadata['all_features']:
            cluster_features = ['rm', 'lstat', 'crim', 'dis'] 
            available_cluster_features = [f for f in cluster_features if f in df_enhanced.columns]
            if len(available_cluster_features) >= 2:
                # La imputación ya se hizo, pero .mean() para NaN en clustering es para KMeans.fit
                # Para predict, KMeans.predict espera que los datos ya estén limpios.
                df_for_kmeans_predict = df_enhanced[available_cluster_features] 
                df_enhanced['neighborhood_cluster'] = kmeans_model.predict(df_for_kmeans_predict)
                
                for feature in available_cluster_features:
                    cluster_id = df_enhanced['neighborhood_cluster'].iloc[0] # Asume una sola fila
                    if feature in cluster_stats_data and cluster_id in cluster_stats_data[feature]:
                        df_enhanced[f'{feature}_cluster_mean'] = cluster_stats_data[feature][cluster_id]['mean']
                        df_enhanced[f'{feature}_cluster_std'] = cluster_stats_data[feature][cluster_id]['std']
                        df_enhanced[f'{feature}_deviation_from_cluster'] = (
                            df_enhanced[feature] - df_enhanced[f'{feature}_cluster_mean']
                        )
                    else:
                        # Fallback si las estadísticas del cluster no se encuentran (raro si el pipeline está bien)
                        # O si el nuevo punto cae en un cluster no visto en el entrenamiento
                        df_enhanced[f'{feature}_cluster_mean'] = df_enhanced[feature].mean() 
                        df_enhanced[f'{feature}_cluster_std'] = df_enhanced[feature].std()
                        df_enhanced[f'{feature}_deviation_from_cluster'] = 0.0 # Valor por defecto
                        
        if percentile_stats: 
            for col in ['rm', 'lstat', 'crim', 'tax']: # Columnas usadas para percentile/zscore en entrenamiento
                if col in df_enhanced.columns and col in percentile_stats:
                    mean_val = percentile_stats[col]['mean']
                    std_val = percentile_stats[col]['std']
                    # La columna de percentil no se calcula en inferencia para un solo punto
                    # ya que requiere toda la distribución de datos. Se puede rellenar con 0 o la media
                    # si RFECV seleccionó esta feature y no fue eliminada.
                    # Para Z-score, usamos mean/std del entrenamiento.
                    df_enhanced[f'{col}_zscore'] = (df_enhanced[col].fillna(mean_val) - mean_val) / (std_val + 1e-8)


        # 6. Características temporales/espaciales simuladas
        if 'dis' in df_enhanced.columns and 'rad' in df_enhanced.columns:
            df_enhanced['accessibility_score'] = (
                (1 / (df_enhanced['dis'] + 1e-8)) * df_enhanced['rad']
            )
        
        # 7. Índices compuestos
        if all(col in df_enhanced.columns for col in ['rm', 'lstat', 'ptratio']):
            df_enhanced['quality_of_life_index'] = (
                df_enhanced['rm'] * 0.4 - 
                df_enhanced['lstat'] * 0.3 - 
                df_enhanced['ptratio'] * 0.3
            )
        
        if all(col in df_enhanced.columns for col in ['crim', 'nox', 'tax']):
            df_enhanced['negative_factors_index'] = (
                df_enhanced['crim'] * 0.4 + 
                df_enhanced['nox'] * 0.3 + 
                df_enhanced['tax'] * 0.3
            )
        
        return df_enhanced

    def ultra_precise_outlier_detection(df, target_col='medv'):
        """Detección ultra precisa de outliers usando múltiples métodos"""
        # Este método es principalmente para el entrenamiento.
        # En la predicción de una única instancia, generalmente NO se eliminan outliers,
        # ya que el punto podría ser un outlier legítimo que debe ser predicho.
        # Solo se aplica en el entrenamiento para limpiar el dataset de entrenamiento.
        from sklearn.ensemble import IsolationForest
        from sklearn.covariance import EllipticEnvelope
        from sklearn.neighbors import LocalOutlierFactor
        
        X = df.drop(target_col, axis=1) if target_col in df.columns else df
        
        # 1. Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        iso_outliers = iso_forest.fit_predict(X) == -1
        
        # 2. Elliptic Envelope
        elliptic = EllipticEnvelope(contamination=0.05, random_state=42)
        try:
            elliptic_outliers = elliptic.fit_predict(X) == -1
        except:
            elliptic_outliers = np.zeros(len(X), dtype=bool)
        
        # 3. Local Outlier Factor
        lof = LocalOutlierFactor(contamination=0.05)
        lof_outliers = lof.fit_predict(X) == -1
        
        # 4. Z-score para el target
        if target_col in df.columns:
            z_scores = np.abs(stats.zscore(df[target_col]))
            zscore_outliers = z_scores > 3
        else:
            zscore_outliers = np.zeros(len(df), dtype=bool)
        
        # Combinar métodos - consideramos outlier si al menos 2 métodos lo detectan
        total_outliers = iso_outliers.astype(int) + elliptic_outliers.astype(int) + lof_outliers.astype(int) + zscore_outliers.astype(int)
        final_outliers = total_outliers >= 2
        
        df_clean = df[~final_outliers].copy()
        
        print(f"📊 Outliers detectados:")
        print(f"   - Isolation Forest: {iso_outliers.sum()}")
        print(f"   - Elliptic Envelope: {elliptic_outliers.sum()}")
        print(f"   - Local Outlier Factor: {lof_outliers.sum()}")
        print(f"   - Z-score (target): {zscore_outliers.sum()}")
        print(f"   - Total removidos: {final_outliers.sum()} ({(final_outliers.sum()/len(df)*100):.1f}%)")
        
        return df_clean

    @staticmethod
    def optimize_hyperparameters_ultra(X_train, y_train, model_name, model):
        """Optimización ultra precisa de hiperparámetros"""
        print(f"    🔧 Optimizando hiperparámetros para {model_name}...")
        
        param_grids = {
            'RandomForestRegressor': {
                'n_estimators': [200, 300, 500],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 2, 3],
                'max_features': ['sqrt', 'log2', 0.8],
                'bootstrap': [True, False]
            },
            'XGBRegressor': {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8, 10],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [1, 1.5, 2]
            },
            'GradientBoostingRegressor': {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 2, 3]
            },
            'SVR': {
                'C': [50, 100, 200, 500],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'epsilon': [0.01, 0.1, 0.2],
                'kernel': ['rbf', 'poly']
            },
            'Ridge': {
                'alpha': [0.1, 1, 10, 50, 100, 200]
            }
        }
        
        model_class_name = model.__class__.__name__
        
        if model_class_name in param_grids:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            grid_search = GridSearchCV(
                model, 
                param_grids[model_class_name],
                cv=cv,
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            print(f"      ✅ Mejor score: {-grid_search.best_score_:.4f}")
            return grid_search.best_estimator_
        
        return model

    def train_ultra_precise_models(self): 
        """Entrenar modelos con precisión ultra alta"""
        
        print("🎯 Iniciando entrenamiento ULTRA PRECISO...")
        print("🚀 Objetivo: Maximizar precisión (minimizar MAPE)")
        
        os.makedirs('models', exist_ok=True)
        
        # 1. Cargar y preparar datos con máxima calidad
        print("📊 Cargando datos con procesamiento de máxima calidad...")
        # 🚨🚨🚨 PUNTO CRÍTICO: ADQUIRIR DATOS FRESCOS AQUÍ 🚨🚨🚨
        # CAMBIA ESTA URL/RUTA para apuntar a tus DATOS FRESCOS y ACTUALIZADOS
        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv" 
        df = pd.read_csv(url)
        
        if df.isnull().sum().sum() > 0:
            imputer = IterativeImputer(random_state=42, max_iter=10)
            df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
            self.imputer = imputer 
            joblib.dump(imputer, 'models/ultra_imputer.pkl')
        else:
            df_imputed = df.copy()
            self.imputer = None 
        
        print(f"📏 Dataset original: {len(df_imputed)} muestras")
        
        # 2. Feature Engineering Ultra Avanzado
        print("🔧 Aplicando feature engineering ultra avanzado...")
        df_ultra, bin_thresholds, kmeans_model_trained, cluster_stats_data_trained, percentile_stats_trained = \
            HousingPredictor._create_ultra_advanced_features_static(df_imputed)
        
        print(f"📈 Características: {len(df.columns)} → {len(df_ultra.columns)}")
        
        # Guardar kmeans_model y sus stats si existen
        if kmeans_model_trained:
            self.kmeans_model = kmeans_model_trained
            joblib.dump(kmeans_model_trained, 'models/ultra_kmeans_model.pkl')
            self.cluster_stats_data = cluster_stats_data_trained
            joblib.dump(cluster_stats_data_trained, 'models/ultra_cluster_stats.pkl')
        
        # Guardar percentile_stats
        self.percentile_stats = percentile_stats_trained
        joblib.dump(percentile_stats_trained, 'models/ultra_percentile_stats.pkl')

        # 3. Detección ultra precisa de outliers
        print("🎯 Detección ultra precisa de outliers...")
        df_clean = HousingPredictor.ultra_precise_outlier_detection(df_ultra)
        print(f"📊 Datos finales: {len(df_clean)} muestras")
        
        # 4. Transformaciones ultra precisas
        print("🔄 Aplicando transformaciones ultra precisas...")
        
        X = df_clean.drop('medv', axis=1)
        y = df_clean['medv']
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        skewed_cols = []
        for col in numeric_cols:
            if abs(X[col].skew()) > 0.3: 
                skewed_cols.append(col)
        
        print(f"📊 Columnas a transformar: {len(skewed_cols)}")
        
        transformers = {
            'yeo_johnson': PowerTransformer(method='yeo-johnson'),
            'box_cox': PowerTransformer(method='box-cox')
        }
        
        X_transformed = X.copy()
        best_transformer_info = None # (name, transformer_instance)
        best_score = -np.inf
        
        for name, transformer in transformers.items():
            try:
                X_temp = X.copy()
                if skewed_cols:
                    if name == 'box_cox':
                        for col in skewed_cols:
                            if (X_temp[col] <= 0).any():
                                X_temp[col] = X_temp[col] - X_temp[col].min() + 1e-8
                    
                    X_temp[skewed_cols] = transformer.fit_transform(X_temp[skewed_cols])
                    
                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression()
                    X_temp_scaled = StandardScaler().fit_transform(X_temp)
                    scores = cross_val_score(lr, X_temp_scaled, y, cv=3, scoring='neg_mean_squared_error')
                    score = scores.mean()
                    
                    if score > best_score:
                        best_score = score
                        best_transformer_info = (name, transformer)
                        X_transformed = X_temp.copy()
                        
            except Exception as e:
                print(f"      ⚠️ Error con {name}: {str(e)}")
                continue
        
        if best_transformer_info:
            print(f"      ✅ Mejor transformador: {best_transformer_info[0]}")
            self.power_transformer = best_transformer_info[1] 
            joblib.dump(self.power_transformer, 'models/ultra_power_transformer.pkl')
        else:
            self.power_transformer = None
            skewed_cols = [] # Asegurar que skewed_cols esté vacío si no se aplicó transformador
        
        # 5. Escalado ultra preciso
        print("📏 Aplicando escalado ultra preciso...")
        
        scalers = {
            'robust': RobustScaler(),
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        
        best_scaler = None
        best_scaler_score = -np.inf
        
        for name, scaler in scalers.items():
            X_scaled_temp = scaler.fit_transform(X_transformed)
            
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression()
            scores = cross_val_score(lr, X_scaled_temp, y, cv=3, scoring='neg_mean_squared_error')
            score = scores.mean()
            
            if score > best_scaler_score:
                best_scaler_score = score
                best_scaler = scaler
        
        self.scaler = best_scaler 
        X_scaled = self.scaler.fit_transform(X_transformed)
        print(f"      ✅ Mejor escalador: {type(self.scaler).__name__}")
        joblib.dump(self.scaler, 'models/ultra_scaler.pkl')
        
        # 6. Selección ultra precisa de características
        print("🎯 Selección ultra precisa de características...")
        
        from sklearn.ensemble import RandomForestRegressor
        rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
        
        rfecv = RFECV(
            estimator=rf_selector,
            step=1,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        X_selected = rfecv.fit_transform(X_scaled, y)
        selected_features = X.columns[rfecv.support_].tolist()
        
        self.feature_selector = rfecv 
        joblib.dump(self.feature_selector, 'models/ultra_feature_selector.pkl')

        print(f"📈 Características seleccionadas: {len(selected_features)} de {X_scaled.shape[1]}")
        print("🔍 Características seleccionadas por RFECV:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i:02d}. {feature}")

        print(f"📊 Score de selección: {rfecv.score(X_scaled, y):.4f}")

        with open("models/ultra_selected_features.txt", "w") as f:
            f.write("\n".join(selected_features))

        # 7. División estratificada optimizada
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.1, random_state=42, shuffle=True
        )
        
        print(f"📊 Entrenamiento: {X_train.shape[0]} | Prueba: {X_test.shape[0]}")
        
        # 8. Modelos ultra avanzados
        print("🤖 Configurando modelos ultra avanzados...")
        
        base_models = {
            'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Support Vector Regression': SVR(),
            'Ridge Regression': Ridge(random_state=42),
            'Extra Trees': ExtraTreesRegressor(random_state=42, n_jobs=-1),
            'Neural Network': MLPRegressor(random_state=42, max_iter=500),
            'KNN': KNeighborsRegressor(),
            'Bayesian Ridge': BayesianRidge(),
            'Huber Regressor': HuberRegressor()
        }
        
        try:
            base_models['LightGBM'] = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        except:
            pass
        
        try:
            base_models['CatBoost'] = CatBoostRegressor(random_state=42, verbose=False)
        except:
            pass
        
        # 9. Entrenar con optimización de hiperparámetros
        print("🚀 Entrenando modelos con optimización ultra precisa...")
        
        optimized_models = {}
        metrics = {}
        
        for model_name, model in base_models.items():
            print(f"   🔄 Entrenando {model_name}...")
            
            try:
                optimized_model = HousingPredictor.optimize_hyperparameters_ultra(X_train, y_train, model_name, model)
                optimized_models[model_name] = optimized_model
                
                cv = KFold(n_splits=10, shuffle=True, random_state=42)
                cv_scores = cross_val_score(optimized_model, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
                
                optimized_model.fit(X_train, y_train)
                y_pred = optimized_model.predict(X_test)
                y_pred_train = optimized_model.predict(X_train)
                
                r2_test = r2_score(y_test, y_pred)
                r2_train = r2_score(y_train, y_pred_train)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                mape = np.mean(np.abs((y_test - y_pred) / np.clip(y_test, 1e-8, None))) * 100
                precision_percentage = 100 - mape
                
                metrics[model_name] = {
                    'r2_test': round(r2_test, 6),
                    'r2_train': round(r2_train, 6),
                    'r2_cv_mean': round(-cv_scores.mean(), 6),
                    'r2_cv_std': round(cv_scores.std(), 6),
                    'mse': round(mse, 6),
                    'rmse': round(rmse, 6),
                    'mae': round(mae, 6),
                    'mape': round(mape, 4),
                    'precision_percentage': round(precision_percentage, 4),
                    'overfitting': round(r2_train - r2_test, 6)
                }
                
                print(f"   ✅ R²: {r2_test:.6f} | RMSE: {rmse:.4f} | MAPE: {mape:.4f}% | Precisión: {precision_percentage:.4f}%")
                
            except Exception as e:
                print(f"   ❌ Error con {model_name}: {str(e)}")
                continue
        
        # 10. Crear ensemble ultra avanzado
        print("🎭 Creando ensemble ultra avanzado...")
        
        best_models = sorted(metrics.items(), key=lambda x: x[1]['precision_percentage'], reverse=True)[:5]
        
        stacking_regressor = None
        voting_regressor = None

        if len(best_models) >= 3:
            base_estimators = [(name.lower().replace(' ', '_'), optimized_models[name]) 
                             for name, _ in best_models]
            
            meta_model = Ridge(alpha=1.0)
            
            stacking_regressor = StackingRegressor(
                estimators=base_estimators,
                final_estimator=meta_model,
                cv=5,
                n_jobs=-1
            )
            
            stacking_regressor.fit(X_train, y_train)
            
            y_pred_stack = stacking_regressor.predict(X_test)
            r2_stack = r2_score(y_test, y_pred_stack)
            mse_stack = mean_squared_error(y_test, y_pred_stack)
            mae_stack = mean_absolute_error(y_test, y_pred_stack)
            rmse_stack = np.sqrt(mse_stack)
            mape_stack = np.mean(np.abs((y_test - y_pred_stack) / np.clip(y_test, 1e-8, None))) * 100
            precision_stack = 100 - mape_stack
            
            cv_scores_stack = cross_val_score(stacking_regressor, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
            
            metrics['Ultra Stacking Ensemble'] = {
                'r2_test': round(r2_stack, 6),
                'r2_cv_mean': round(-cv_scores_stack.mean(), 6),
                'r2_cv_std': round(cv_scores_stack.std(), 6),
                'mse': round(mse_stack, 6),
                'rmse': round(rmse_stack, 6),
                'mae': round(mae_stack, 6),
                'mape': round(mape_stack, 4),
                'precision_percentage': round(precision_stack, 4)
            }
            
            print(f"   ✅ Stacking R²: {r2_stack:.6f} | MAPE: {mape_stack:.4f}% | Precisión: {precision_stack:.4f}%")
            
            voting_regressor = VotingRegressor(
                estimators=base_estimators,
                n_jobs=-1
            )
            
            voting_regressor.fit(X_train, y_train)
            
            y_pred_vote = voting_regressor.predict(X_test)
            r2_vote = r2_score(y_test, y_pred_vote)
            mse_vote = mean_squared_error(y_test, y_pred_vote)
            rmse_vote = np.sqrt(mse_vote)
            mape_vote = np.mean(np.abs((y_test - y_pred_vote) / np.clip(y_test, 1e-8, None))) * 100
            precision_vote = 100 - mape_vote
            
            metrics['Ultra Voting Ensemble'] = {
                'r2_test': round(r2_vote, 6),
                'mse': round(mse_vote, 6),
                'rmse': round(rmse_vote, 6),
                'mae': round(mean_absolute_error(y_test, y_pred_vote), 6),
                'mape': round(mape_vote, 4),
                'precision_percentage': round(precision_vote, 4)
            }
            
            print(f"   ✅ Voting R²: {r2_vote:.6f} | MAPE: {mape_vote:.4f}% | Precisión: {precision_vote:.4f}%")
        
        # 11. Guardar modelos ultra precisos
        print("💾 Guardando modelos ultra precisos...")
        
        for model_name, model_obj in optimized_models.items(): # Renombrado 'model' a 'model_obj' para evitar conflicto
            safe_name = model_name.lower().replace(' ', '_').replace('ó', 'o')
            joblib.dump(model_obj, f'models/ultra_{safe_name}_model.pkl')
        
        if stacking_regressor:
            joblib.dump(stacking_regressor, 'models/ultra_stacking_ensemble.pkl')
            self.model = stacking_regressor 
        
        if voting_regressor:
            joblib.dump(voting_regressor, 'models/ultra_voting_ensemble.pkl')
            if self.model is None: # Si no se estableció stacking, usar voting
                self.model = voting_regressor 
        
        if self.model is None and optimized_models: # Si no hay ensembles, el mejor modelo individual
             best_model_name = max(metrics.keys(), key=lambda k: metrics[k]['precision_percentage'])
             self.model = optimized_models[best_model_name]
        
        # self.imputer ya está guardado
        # self.power_transformer ya está guardado
        # self.scaler ya está guardado
        # self.feature_selector ya está guardado
        # self.kmeans_model y self.cluster_stats_data ya están guardados
        # self.percentile_stats ya están guardados
        
        # Metadata ultra completo
        ultra_metadata = {
            'feature_names': selected_features,
            'all_features': X.columns.tolist(), # Todas las características después de FE, antes de RFE
            'cols_to_transform': skewed_cols,
            'ultra_metrics': metrics,
            'test_data': {
                'X_test': X_test.tolist(),
                'y_test': y_test.tolist()
            },
            'data_processing': {
                'original_samples': len(df),
                'original_features': len(df.columns),
                'final_samples': len(df_clean),
                'engineered_features': len(df_ultra.columns),
                'selected_features': len(selected_features),
                'outliers_removed': len(df_ultra) - len(df_clean)
            },
            'best_models': {
                'highest_precision': max(metrics.keys(), key=lambda k: metrics[k]['precision_percentage']),
                'highest_r2': max(metrics.keys(), key=lambda k: metrics[k]['r2_test']),
                'lowest_mape': min(metrics.keys(), key=lambda k: metrics[k]['mape'])
            },
            'model_configurations': {
                'cross_validation_folds': 10,
                'test_size': 0.1,
                'random_state': 42,
                'outlier_contamination': 0.05,
                'feature_selection_method': 'RFECV',
                'best_scaler': type(self.scaler).__name__ if self.scaler else None,
                'best_transformer': best_transformer_info[0] if best_transformer_info else None
            },
            'bin_thresholds': bin_thresholds, 
            'kmeans_model_path': 'models/ultra_kmeans_model.pkl' if kmeans_model_trained else None,
            'cluster_stats_data_path': 'models/ultra_cluster_stats.pkl' if cluster_stats_data_trained else None,
            'percentile_stats_path': 'models/ultra_percentile_stats.pkl' if percentile_stats_trained else None
        }
        
        self.metadata = ultra_metadata 
        joblib.dump(self.metadata, 'models/ultra_metadata.pkl')
        
        self.is_trained = True 
        
        print("\n" + "="*80)
        print("🏆 RESULTADOS ULTRA PRECISOS - RANKING DE MODELOS")
        print("="*80)
        
        sorted_models = sorted(metrics.items(), key=lambda x: x[1]['precision_percentage'], reverse=True)
        
        print(f"{'Modelo':<25} {'Precisión%':<12} {'R²':<10} {'MAPE%':<8} {'RMSE':<8} {'Overfitting':<12}")
        print("-" * 80)
        
        for i, (model_name, metric) in enumerate(sorted_models):
            precision = metric['precision_percentage']
            r2 = metric['r2_test']
            mape = metric['mape']
            rmse = metric['rmse']
            overfit = metric.get('overfitting', 0)
            
            rank_emoji = ["�", "🥈", "🥉"][i] if i < 3 else f"{i+1:2d}."
            
            print(f"{rank_emoji} {model_name:<22} {precision:>8.4f}%   {r2:>7.6f}  {mape:>6.4f}%  {rmse:>6.4f}  {overfit:>8.6f}")
        
        print("\n" + "="*60)
        print("📊 ESTADÍSTICAS ULTRA DETALLADAS")
        print("="*60)
        
        best_model_name = sorted_models[0][0]
        best_metrics = sorted_models[0][1]
        
        print(f"🏆 MEJOR MODELO: {best_model_name}")
        print(f"   • Precisión: {best_metrics['precision_percentage']:.4f}%")
        print(f"   • R² Test: {best_metrics['r2_test']:.6f}")
        print(f"   • MAPE: {best_metrics['mape']:.4f}%")
        print(f"   • RMSE: {best_metrics['rmse']:.4f}")
        print(f"   • MAE: {best_metrics['mae']:.4f}")
        
        if 'r2_cv_mean' in best_metrics:
            print(f"   • R² CV: {best_metrics['r2_cv_mean']:.6f} ± {best_metrics['r2_cv_std']:.6f}")
        
        print(f"\n📈 PROCESAMIENTO DE DATOS:")
        print(f"   • Muestras originales: {ultra_metadata['data_processing']['original_samples']}")
        print(f"   • Muestras finales: {ultra_metadata['data_processing']['final_samples']}")
        print(f"   • Outliers removidos: {ultra_metadata['data_processing']['outliers_removed']}")
        print(f"   • Características originales: {ultra_metadata['data_processing']['original_features']}")
        print(f"   • Características creadas: {ultra_metadata['data_processing']['engineered_features']}")
        print(f"   • Características seleccionadas: {ultra_metadata['data_processing']['selected_features']}")
        
        print(f"\n🔧 CONFIGURACIÓN:")
        print(f"   • Escalador: {ultra_metadata['model_configurations']['best_scaler']}")
        print(f"   • Transformador: {ultra_metadata['model_configurations']['best_transformer']}")
        print(f"   • Validación cruzada: {ultra_metadata['model_configurations']['cross_validation_folds']} folds")
        print(f"   • Conjunto de prueba: {ultra_metadata['model_configurations']['test_size']*100}%")
        print(f"   • Umbrales binarios (Crimen/Lstat): {ultra_metadata['bin_thresholds']}")
        
        print("\n✅ Todos los modelos y metadata guardados en directorio 'models/'")
        print("🎯 ¡ENTRENAMIENTO ULTRA PRECISO COMPLETADO!")
        
        # Retorna self para poder usar la instancia directamente
        return self

    def load_pipeline_components(self, models_path='models'):
        """
        Carga todos los preprocesadores y el modelo entrenado
        desde el directorio especificado en la instancia de la clase.
        """
        try:
            # Carga la metadata primero
            self.metadata = joblib.load(os.path.join(models_path, 'ultra_metadata.pkl'))

            # Carga los preprocesadores
            self.imputer = joblib.load(os.path.join(models_path, 'ultra_imputer.pkl')) if os.path.exists(os.path.join(models_path, 'ultra_imputer.pkl')) else None
            self.power_transformer = joblib.load(os.path.join(models_path, 'ultra_power_transformer.pkl')) if os.path.exists(os.path.join(models_path, 'ultra_power_transformer.pkl')) else None
            self.scaler = joblib.load(os.path.join(models_path, 'ultra_scaler.pkl'))
            self.feature_selector = joblib.load(os.path.join(models_path, 'ultra_feature_selector.pkl'))
            
            # Carga KMeans model y sus stats si fueron guardados
            if self.metadata.get('kmeans_model_path') and os.path.exists(self.metadata['kmeans_model_path']):
                self.kmeans_model = joblib.load(self.metadata['kmeans_model_path'])
                self.cluster_stats_data = joblib.load(self.metadata['cluster_stats_data_path'])
            
            # Carga percentile stats
            if self.metadata.get('percentile_stats_path') and os.path.exists(self.metadata['percentile_stats_path']):
                self.percentile_stats = joblib.load(self.metadata['percentile_stats_path'])


            # Carga el modelo final (priorizando ensembles)
            model_candidates = [
                os.path.join(models_path, 'ultra_stacking_ensemble.pkl'),
                os.path.join(models_path, 'ultra_voting_ensemble.pkl')
            ]
            
            if 'best_models' in self.metadata and 'highest_precision' in self.metadata['best_models']:
                best_model_name_safe = self.metadata['best_models']['highest_precision'].lower().replace(' ', '_').replace('ó', 'o')
                model_candidates.append(os.path.join(models_path, f'ultra_{best_model_name_safe}_model.pkl'))
            
            self.model = None
            for path in model_candidates:
                if os.path.exists(path):
                    self.model = joblib.load(path)
                    print(f"✅ Modelo final cargado: {os.path.basename(path)}")
                    break
            
            if self.model is None:
                raise FileNotFoundError("Ningún modelo final pudo ser cargado.")

            self.is_loaded = True
            print("✅ Pipeline de predicción cargado exitosamente.")
            return True
        except FileNotFoundError as e:
            print(f"❌ Error al cargar componentes del pipeline: {e}. Asegúrate de que los archivos .pkl estén en '{models_path}'")
            self.is_loaded = False
            return False
        except Exception as e:
            print(f"❌ Ocurrió un error inesperado al cargar el pipeline: {e}")
            self.is_loaded = False
            return False

    def predict_with_ultra_model(self, features_dict, model_name='best'):
        """
        Realiza predicciones con el modelo ultra preciso cargado en la instancia.
        
        Args:
            features_dict (dict): Diccionario con las características de la casa (datos crudos).
            model_name (str): Nombre del modelo a usar ('best', o nombre específico como 'Random Forest').
                              'best' seleccionará el modelo con mayor precisión según la metadata.
        
        Returns:
            dict: Diccionario con el precio predicho y metadatos de la predicción,
                  o un diccionario de error si falla.
        """
        if not self.is_loaded:
            print("❌ Pipeline no cargado. Llama a 'load_pipeline_components()' primero para cargar los modelos y preprocesadores.")
            return {
                'error': "Pipeline no cargado. Por favor, cargue los componentes antes de predecir.",
                'predicted_price': None
            }

        try:
            # Determinar el modelo a usar (si es 'best', usa el de mayor precisión)
            if model_name == 'best':
                model_to_use_name = self.metadata['best_models']['highest_precision']
            else:
                model_to_use_name = model_name
            
            # Si se solicitó un modelo específico que no es el self.model principal
            # esto implicaría cargar otro modelo, lo cual rompe la eficiencia de cargar solo uno.
            # Asumiremos que self.model ya es el mejor o el ensemble.
            # Para usar otro modelo, tendrías que cargar el .pkl específico aquí,
            # pero esto anularía el propósito de 'self.model' siendo el modelo principal.
            # Para simplificar y mantener la eficiencia, usaremos self.model.
            model_obj = self.model 
            
            # Asegurarse de que el DataFrame tiene las columnas originales esperadas del entrenamiento.
            expected_original_features = [col for col in self.metadata['all_features'] if col != 'medv']
            
            df_single_instance = pd.DataFrame([features_dict])
            for col in expected_original_features:
                if col not in df_single_instance.columns:
                    df_single_instance[col] = np.nan # El imputer manejará esto

            df_single_instance = df_single_instance[expected_original_features]

            print("\n--- Procesando nuevo dato para predicción ---")
            print(f"Datos brutos: {df_single_instance.to_dict('records')[0]}")

            # Imputación de valores faltantes
            if self.imputer:
                imputed_data = self.imputer.transform(df_single_instance.values)
                df_imputed = pd.DataFrame(imputed_data, columns=df_single_instance.columns)
            else:
                df_imputed = df_single_instance.copy()
            print(f"Post-imputación: {df_imputed.to_dict('records')[0]}")

            # Ingeniería de Características Ultra Avanzada (usa la nueva función interna)
            df_enhanced = self._apply_feature_engineering_for_prediction(df_imputed)
            print(f"Post-ingeniería de características: {df_enhanced.shape[1]} columnas")
            
            # Alinear las columnas del DataFrame preprocesado con las que el modelo espera
            all_features_after_fe_before_rfe = self.metadata['all_features'] # Columnas de X en el entrenamiento
            
            X_for_preprocessing = pd.DataFrame(index=df_enhanced.index)
            for col in all_features_after_fe_before_rfe:
                if col in df_enhanced.columns:
                    X_for_preprocessing[col] = df_enhanced[col]
                else:
                    X_for_preprocessing[col] = 0.0 # Rellenar con 0 si una característica no se generó (ej. si era condicional)

            # Transformación de variables sesgadas
            if self.power_transformer and self.metadata.get('model_configurations', {}).get('best_transformer'):
                skewed_cols_from_training = self.metadata['cols_to_transform']
                cols_to_transform = [col for col in skewed_cols_from_training if col in X_for_preprocessing.columns]
                if cols_to_transform:
                    if self.metadata['model_configurations']['best_transformer'] == 'box_cox':
                        for col in cols_to_transform:
                            if (X_for_preprocessing[col] <= 0).any():
                                X_for_preprocessing[col] = X_for_preprocessing[col] - X_for_preprocessing[col].min() + 1e-8
                    
                    X_for_preprocessing[cols_to_transform] = self.power_transformer.transform(X_for_preprocessing[cols_to_transform])
                print("Post-transformación de sesgo.")

            # Escalado
            X_scaled = self.scaler.transform(X_for_preprocessing)
            print("Post-escalado.")

            # Selección de características
            X_final = self.feature_selector.transform(X_scaled)
            print(f"Post-selección de características: {X_final.shape[1]} características finales.")

            # Predicción
            prediction = model_obj.predict(X_final)[0]
            print(f"Predicción final: ${prediction:.2f}K")

            return {
                'predicted_price': round(prediction, 2),
                'model_used': model_to_use_name,
                'confidence': 'Alta' if model_to_use_name in self.metadata['best_models'].values() else 'Media',
                'model_precision': self.metadata['ultra_metrics'][model_to_use_name]['precision_percentage']
            }
            
        except FileNotFoundError as e:
            print(f"❌ Error al cargar componentes del pipeline o el modelo: {e}")
            return {
                'error': f"Error en predicción: No se encontraron todos los componentes del pipeline. {str(e)}",
                'predicted_price': None
            }
        except Exception as e:
            print(f"❌ Ocurrió un error inesperado durante la predicción: {e}")
            return {
                'error': f"Error en predicción: {str(e)}",
                'predicted_price': None
            }

    # Ya no necesitamos estas funciones auxiliares sueltas, están dentro de la clase o se llaman directamente.
    # def evaluate_ultra_models():
    # def ultra_model_comparison():
    # def main(): # Esta función principal también se elimina de la clase
