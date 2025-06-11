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

    def predict_with_ultra_model(features_dict, model_name='best'):
        """
        Realizar predicciones con el modelo ultra preciso
        
        Args:
            features_dict: Diccionario con las características de la casa
            model_name: Nombre del modelo a usar ('best', o nombre específico)
        
        Returns:
            Predicción del precio de la casa
        """
        try:
            # Cargar metadata
            metadata = joblib.load('models/ultra_metadata.pkl')
            
            # Determinar el mejor modelo si no se especifica
            if model_name == 'best':
                model_name = metadata['best_models']['highest_precision']
            
            # Cargar el modelo
            safe_name = model_name.lower().replace(' ', '_').replace('ó', 'o')
            
            # Intentar cargar diferentes versiones del modelo
            model_paths = [
                f'models/ultra_{safe_name}_model.pkl',
                f'models/ultra_{safe_name}.pkl',
                f'models/{safe_name}_model.pkl'
            ]
            
            model = None
            for path in model_paths:
                if os.path.exists(path):
                    model = joblib.load(path)
                    break
            
            if model is None:
                # Cargar el primer modelo disponible
                available_models = [f for f in os.listdir('models/') if f.endswith('_model.pkl')]
                if available_models:
                    model = joblib.load(f'models/{available_models[0]}')
                    print(f"⚠️ Modelo {model_name} no encontrado, usando {available_models[0]}")
                else:
                    raise FileNotFoundError("No se encontraron modelos entrenados")
            
            # Cargar transformadores
            scaler = joblib.load('models/ultra_scaler.pkl')
            feature_selector = joblib.load('models/ultra_feature_selector.pkl')
            
            power_transformer = None
            if os.path.exists('models/ultra_power_transformer.pkl'):
                power_transformer = joblib.load('models/ultra_power_transformer.pkl')
            
            # Crear DataFrame con todas las características necesarias
            all_features = metadata['all_features']
            
            # Inicializar con valores por defecto (medias del dataset original)
            default_values = {
                'crim': 3.61, 'zn': 11.36, 'indus': 11.14, 'chas': 0.07,
                'nox': 0.55, 'rm': 6.28, 'age': 68.57, 'dis': 3.80,
                'rad': 9.55, 'tax': 408.24, 'ptratio': 18.46, 'b': 356.67,
                'lstat': 12.65
            }
            
            # Crear DataFrame base
            input_data = pd.DataFrame([default_values])
            
            # Actualizar con los valores proporcionados
            for key, value in features_dict.items():
                if key in input_data.columns:
                    input_data[key] = value
            
            # Aplicar feature engineering
            input_data_ultra = create_ultra_advanced_features(input_data)
            
            # Asegurar que tenemos todas las características necesarias
            for feature in all_features:
                if feature not in input_data_ultra.columns:
                    input_data_ultra[feature] = 0  # Valor por defecto
            
            # Reordenar columnas según el orden original
            input_data_ultra = input_data_ultra[all_features]
            
            # Aplicar transformaciones
            if power_transformer and metadata['cols_to_transform']:
                cols_to_transform = [col for col in metadata['cols_to_transform'] 
                                if col in input_data_ultra.columns]
                if cols_to_transform:
                    input_data_ultra[cols_to_transform] = power_transformer.transform(
                        input_data_ultra[cols_to_transform]
                    )
            
            # Escalar
            input_scaled = scaler.transform(input_data_ultra)
            
            # Seleccionar características
            input_selected = feature_selector.transform(input_scaled)
            
            # Realizar predicción
            prediction = model.predict(input_selected)[0]
            
            return {
                'predicted_price': round(prediction, 2),
                'model_used': model_name,
                'confidence': 'Alta' if model_name in metadata['best_models'].values() else 'Media',
                'model_precision': metadata['ultra_metrics'][model_name]['precision_percentage']
            }
            
        except Exception as e:
            return {
                'error': f"Error en predicción: {str(e)}",
                'predicted_price': None
            }

    def evaluate_ultra_models():
        """Evaluar todos los modelos entrenados con métricas detalladas"""
        try:
            metadata = joblib.load('models/ultra_metadata.pkl')
            metrics = metadata['ultra_metrics']
            
            print("🎯 EVALUACIÓN ULTRA DETALLADA DE MODELOS")
            print("="*70)
            
            # Cargar datos de prueba
            X_test = np.array(metadata['test_data']['X_test'])
            y_test = np.array(metadata['test_data']['y_test'])
            
            for model_name, model_metrics in metrics.items():
                print(f"\n🤖 {model_name.upper()}")
                print("-" * 50)
                print(f"   Precisión: {model_metrics['precision_percentage']:.4f}%")
                print(f"   R² Score: {model_metrics['r2_test']:.6f}")
                print(f"   MAPE: {model_metrics['mape']:.4f}%")
                print(f"   RMSE: {model_metrics['rmse']:.4f}")
                print(f"   MAE: {model_metrics['mae']:.4f}")
                
                if 'overfitting' in model_metrics:
                    overfitting_status = "🔴 Alto" if model_metrics['overfitting'] > 0.1 else "🟢 Bajo"
                    print(f"   Overfitting: {overfitting_status} ({model_metrics['overfitting']:.6f})")
            
            # Recomendación final
            best_model = metadata['best_models']['highest_precision']
            print(f"\n🏆 RECOMENDACIÓN: Usar '{best_model}' para máxima precisión")
            print(f"   Precisión: {metrics[best_model]['precision_percentage']:.4f}%")
            
        except Exception as e:
            print(f"❌ Error al evaluar modelos: {str(e)}")

    def ultra_model_comparison():
        """Comparación visual de todos los modelos"""
        try:
            metadata = joblib.load('models/ultra_metadata.pkl')
            metrics = metadata['ultra_metrics']
            
            print("\n📊 COMPARACIÓN ULTRA DETALLADA")
            print("="*90)
            
            # Crear tabla comparativa
            models_data = []
            for name, metric in metrics.items():
                models_data.append({
                    'Modelo': name,
                    'Precisión%': metric['precision_percentage'],
                    'R²': metric['r2_test'],
                    'MAPE%': metric['mape'],
                    'RMSE': metric['rmse'],
                    'MAE': metric['mae']
                })
            
            # Ordenar por precisión
            models_data.sort(key=lambda x: x['Precisión%'], reverse=True)
            
            # Mostrar tabla
            header = f"{'Modelo':<25} {'Precisión%':<12} {'R²':<10} {'MAPE%':<8} {'RMSE':<8} {'MAE':<8}"
            print(header)
            print("-" * len(header))
            
            for data in models_data:
                print(f"{data['Modelo']:<25} {data['Precisión%']:>8.4f}%   "
                    f"{data['R²']:>7.6f}  {data['MAPE%']:>6.4f}%  "
                    f"{data['RMSE']:>6.4f}  {data['MAE']:>6.4f}")
            
        except Exception as e:
            print(f"❌ Error en comparación: {str(e)}")

    # Función principal para ejecutar todo el pipeline
    def main():
        """Ejecutar el pipeline completo de entrenamiento ultra preciso"""
        print("🚀 INICIANDO PIPELINE ULTRA PRECISO")
        print("="*50)
        
        try:
            # Entrenar modelos
            models, metrics, metadata = train_ultra_precise_models()
            
            print("\n🎯 Pipeline completado exitosamente!")
            print("💾 Todos los archivos guardados en directorio 'models/'")
            
            # Mostrar ejemplo de uso
            print("\n📝 EJEMPLO DE USO:")
            print("="*40)
            print("# Predicción de ejemplo")
            print("features = {")
            print("    'rm': 6.5,        # Número de habitaciones")
            print("    'lstat': 8.0,     # % población de bajo estatus")
            print("    'crim': 0.5,      # Tasa de crimen")
            print("    'ptratio': 15.0,  # Ratio profesor-estudiante")
            print("    'dis': 4.0        # Distancia a centros de empleo")
            print("}")
            print("result = predict_with_ultra_model(features)")
            print("print(f'Precio predicho: ${result[\"predicted_price\"]}k')")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en pipeline: {str(e)}")
            return False

    if __name__ == "__main__":
        main()