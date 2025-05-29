import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import base64
import io

class HousingPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.power_transformer = None
        self.metadata = None
        self.feature_names = None
        self.cols_to_transform = None
        self.X_test = None
        self.y_test = None
        
    def load_trained_models(self):
        """Cargar modelos y transformadores pre-entrenados"""
        try:
            models_dir = 'models'
            
            # Verificar que exista la carpeta de modelos
            if not os.path.exists(models_dir):
                raise FileNotFoundError("Carpeta 'models' no encontrada. Ejecuta train_models.py primero.")
            
            print("📦 Cargando modelos pre-entrenados...")
            
            # Cargar modelos
            model_files = {
                'Random Forest': 'random_forest_model.pkl',
                'Regresión Lineal': 'regresion_lineal_model.pkl',
                'XGBoost': 'xgboost_model.pkl'
            }
            
            for model_name, filename in model_files.items():
                filepath = os.path.join(models_dir, filename)
                if os.path.exists(filepath):
                    self.models[model_name] = joblib.load(filepath)
                    print(f"  ✅ {model_name} cargado")
                else:
                    print(f"  ❌ {filename} no encontrado")
            
            # Cargar transformadores
            scaler_path = os.path.join(models_dir, 'scaler.pkl')
            transformer_path = os.path.join(models_dir, 'power_transformer.pkl')
            metadata_path = os.path.join(models_dir, 'metadata.pkl')
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("  ✅ Scaler cargado")
            
            if os.path.exists(transformer_path):
                self.power_transformer = joblib.load(transformer_path)
                print("  ✅ Power Transformer cargado")
            
            if os.path.exists(metadata_path):
                self.metadata = joblib.load(metadata_path)
                self.feature_names = self.metadata['feature_names']
                self.cols_to_transform = self.metadata['cols_to_transform']
                self.X_test = self.metadata['test_data']['X_test']
                self.y_test = np.array(self.metadata['test_data']['y_test'])
                print("  ✅ Metadata cargado")
            
            if len(self.models) == 0:
                raise Exception("No se pudo cargar ningún modelo")
                
            print(f"✅ {len(self.models)} modelos cargados exitosamente!")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelos: {str(e)}")
            return False
    
    def predict(self, features_df, model_type):
        """Hacer predicción con el modelo seleccionado"""
        if model_type not in self.models:
            raise ValueError(f"Modelo '{model_type}' no disponible")
        
        # Transformar características
        features_transformed = features_df.copy()
        if self.power_transformer and self.cols_to_transform:
            features_transformed[self.cols_to_transform] = self.power_transformer.transform(
                features_transformed[self.cols_to_transform]
            )
        
        # Escalar características
        if self.scaler:
            features_scaled = self.scaler.transform(features_transformed)
        else:
            features_scaled = features_transformed.values
        
        # Hacer predicción
        prediction = self.models[model_type].predict(features_scaled)[0]
        
        return prediction
    
    def get_feature_names(self):
        """Obtener nombres de las características"""
        return self.feature_names if self.feature_names else []
    
    def get_model_metrics(self):
        """Obtener métricas pre-calculadas"""
        if self.metadata and 'metrics' in self.metadata:
            return self.metadata['metrics']
        
        # Si no hay métricas pre-calculadas, calcular sobre datos de prueba
        if self.X_test is not None and self.y_test is not None:
            metrics = {}
            for model_name, model in self.models.items():
                y_pred = model.predict(self.X_test)
                r2 = r2_score(self.y_test, y_pred)
                mse = mean_squared_error(self.y_test, y_pred)
                metrics[model_name] = {
                    'r2': round(r2, 4),
                    'mse': round(mse, 4)
                }
            return metrics
        
        return {}
    
    def generate_comparison_plot(self):
        """Generar gráfico de comparación de modelos"""
        if self.X_test is None or self.y_test is None:
            raise ValueError("Datos de prueba no disponibles")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        model_names = ['Regresión Lineal', 'Random Forest', 'XGBoost']
        short_names = ['LR', 'RF', 'XGB']
        
        plot_idx = 0
        for model_name, short_name, color in zip(model_names, short_names, colors):
            if model_name in self.models:
                y_pred = self.models[model_name].predict(self.X_test)
                
                axes[plot_idx].scatter(self.y_test, y_pred, color=color, alpha=0.7)
                axes[plot_idx].plot([self.y_test.min(), self.y_test.max()], 
                            [self.y_test.min(), self.y_test.max()], 'k--', lw=2)
                axes[plot_idx].set_xlabel('Valores Reales')
                axes[plot_idx].set_ylabel(f'Predicciones ({short_name})')
                axes[plot_idx].set_title(short_name)
                axes[plot_idx].grid(True)
                plot_idx += 1
        
        # Ocultar ejes no utilizados
        for i in range(plot_idx, 3):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Convertir a base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close()
        
        return img_base64
    
    def is_ready(self):
        """Verificar si el predictor está listo para usar"""
        return (len(self.models) > 0 and 
                self.scaler is not None and 
                self.power_transformer is not None and 
                self.feature_names is not None)