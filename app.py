from flask import Flask, render_template, request, jsonify, Response
import json
import pandas as pd
import numpy as np
import joblib
import os
import math
import logging # Mantener para otras bibliotecas que lo usen
import base64
import io
from datetime import datetime, timedelta # Importar timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
from flask.json.provider import DefaultJSONProvider

# Importaciones para el nuevo sistema de logging y monitoreo
from dotenv import load_dotenv
import atexit
from threading import Timer
import sys

# Ignorar advertencias, especialmente de sklearn si se usan versiones mixtas
warnings.filterwarnings('ignore')

# --- IMPORTANTE: Configuración global para scikit-learn para que los transformadores devuelvan DataFrames ---
from sklearn import set_config
set_config(transform_output="pandas")

# Cargar variables de entorno (al inicio para que estén disponibles globalmente)
load_dotenv()

# Importar funciones de monitoreo y chatbot (asegúrate de que estos archivos estén presentes)
from monitoring import calculate_psi_safe_percentiles, ModelMonitor, PredictionLog, DriftAlert
from log_monitoring import check_logs_and_send_email # Importar la función directamente
from chatbot import FAQChatbot

# --- Configuración del Logging con DualLogger ---
LOG_FILE_PATH = 'app_logs.log' # Archivo donde se escribirán los logs para el monitoreo

# Crear un logger personalizado que escribe tanto a la consola como a un archivo
class DualLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        # Asegurarse de que el directorio del archivo de log existe
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)

    def log(self, level, message, exc_info=False):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}"
        
        # Si se incluye información de excepción, añadirla al mensaje
        if exc_info:
            import traceback
            log_entry += "\n" + traceback.format_exc()
        
        # Escribir a consola (para los logs de Render)
        print(log_entry)
        
        # Escribir a archivo (para que log_monitoring.py pueda leerlo)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            print(f"Error escribiendo a archivo de log: {e}")

    def info(self, message):
        self.log("INFO", message)

    def warning(self, message):
        self.log("WARNING", message)

    def error(self, message, exc_info=False):
        self.log("ERROR", message, exc_info)

    def debug(self, message): # Añadir nivel DEBUG para logs más detallados
        self.log("DEBUG", message)

# Inicializar el DualLogger globalmente para esta aplicación
dual_logger = DualLogger(LOG_FILE_PATH)

# Configurar el logging estándar de Python para bibliotecas de terceros
# Esto asegura que los mensajes de otras bibliotecas también se dirijan a stdout
logging.basicConfig(
    level=logging.INFO, # Puedes cambiar a logging.DEBUG si necesitas más detalle
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
# Nota: La línea `logger = logging.getLogger(__name__)` ya no es estrictamente necesaria
# para tus logs directos, ya que usarás `dual_logger`, pero la mantendremos para coherencia
# con cómo otras bibliotecas podrían obtener su logger.

# Inicializar el monitor globalmente (antes del objeto app)
monitor = ModelMonitor()

app = Flask(__name__)

# --- Clase para manejar la serialización JSON segura (evitar NaN/Inf en JSON) ---
class SafeJSONProvider(DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        def sanitize(obj):
            if isinstance(obj, float):
                if math.isinf(obj):
                    return 9999.0 if obj > 0 else -9999.0
                elif math.isnan(obj):
                    return 0.0
            elif isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(v) for v in obj]
            return obj

        obj = sanitize(obj)
        return json.dumps(obj, **kwargs)

    def loads(self, s, **kwargs):
        return json.loads(s, **kwargs)
    
app.json_provider_class = SafeJSONProvider
app.json = app.json_provider_class(app) # ¡IMPORTANTE!

# --- Función de utilidad para sanear datos numéricos (alternativa/complemento al JSONProvider) ---
def sanitize_numeric_values(data):
    """Recursively sanitize numeric values in a dictionary or list, replacing NaN/Inf."""
    if isinstance(data, dict):
        return {key: sanitize_numeric_values(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_numeric_values(item) for item in data]
    elif isinstance(data, float):
        if math.isinf(data):
            return 9999.0 if data > 0 else -9999.0
        elif math.isnan(data):
            return 0.0
        return data
    else:
        return data

# --- Función PSI segura (si la usas directamente en app.py, si no, debe estar en monitoring.py) ---
def calculate_psi_safe(baseline_dist, current_dist, epsilon=1e-8):
    """
    Calcular PSI (Population Stability Index) de forma segura, evitando valores infinitos.
    NOTA: Lo ideal es que esta función sea parte de 'monitoring.py' y sea utilizada por ModelMonitor.
    """
    import numpy as np

    baseline_dist = np.array(baseline_dist) + epsilon
    current_dist = np.array(current_dist) + epsilon

    baseline_dist = baseline_dist / np.sum(baseline_dist)
    current_dist = current_dist / np.sum(current_dist)

    baseline_dist = np.nan_to_num(baseline_dist, nan=epsilon, posinf=epsilon, neginf=epsilon)
    current_dist = np.nan_to_num(current_dist, nan=epsilon, posinf=epsilon, neginf=epsilon)

    psi = np.sum((current_dist - baseline_dist) * np.log(current_dist / baseline_dist))

    # Limitar PSI para evitar valores extremos
    max_psi = 10.0
    if np.isnan(psi) or np.isinf(psi):
        psi = max_psi

    return float(psi)


class HousingPredictor:
    """Clase para manejar predicciones de precios de vivienda"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.power_transformer = None
        self.imputer = None
        self.metadata = {}
        self.feature_names = [] # Nombres de características finales después del preprocesamiento
        self.ready = False
        
    def load_trained_models(self):
        """Cargar todos los modelos entrenados y preprocesadores"""
        try:
            dual_logger.info("🔄 Cargando modelos entrenados...")
            
            models_dir = 'models'
            if not os.path.exists(models_dir):
                dual_logger.error(f"❌ Directorio {models_dir} no encontrado. Asegúrate de que tus modelos y preprocesadores estén en esta carpeta.")
                return False
            
            # Cargar metadata (debe contener 'final_feature_names' y umbrales/cuantiles si se usaron)
            metadata_path = os.path.join(models_dir, 'metadata.pkl') 
            if os.path.exists(metadata_path):
                self.metadata = joblib.load(metadata_path)
                # Asegúrate de que la clave aquí coincida con cómo guardas la lista final de features en tu entrenamiento
                self.feature_names = self.metadata.get('final_feature_names', []) 
                if not self.feature_names:
                    dual_logger.warning("⚠️ 'final_feature_names' no encontrado en metadata. Esto puede causar errores de características.")
                
                dual_logger.info(f"✅ Metadata cargada: {len(self.feature_names)} características finales esperadas.")
            else:
                dual_logger.warning("⚠️ Archivo metadata.pkl no encontrado. El preprocesamiento puede fallar si no hay nombres de características ni umbrales.")
            
            # Cargar transformadores
            transformers = {
                'scaler.pkl': 'scaler',
                'feature_selector.pkl': 'feature_selector', 
                'power_transformer.pkl': 'power_transformer',
                'imputer.pkl': 'imputer'
            }
            
            for file_name, attr_name in transformers.items():
                file_path = os.path.join(models_dir, file_name)
                if os.path.exists(file_path):
                    setattr(self, attr_name, joblib.load(file_path))
                    dual_logger.info(f"✅ {attr_name} cargado")
                else:
                    dual_logger.warning(f"⚠️ {file_name} no encontrado. El preprocesamiento podría estar incompleto.")
            
            # Cargar modelos
            model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
            
            for model_file in model_files:
                try:
                    model_path = os.path.join(models_dir, model_file)
                    model = joblib.load(model_path)
                    
                    model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
                    self.models[model_name] = model
                    dual_logger.info(f"✅ Modelo cargado: {model_name}")
                    
                except Exception as e:
                    dual_logger.error(f"❌ Error cargando {model_file}: {str(e)}", exc_info=True)
            
            # Cargar ensemble si existe
            ensemble_path = os.path.join(models_dir, 'voting_ensemble.pkl')
            if os.path.exists(ensemble_path):
                self.models['Voting Ensemble'] = joblib.load(ensemble_path)
                dual_logger.info("✅ Voting Ensemble cargado")
            
            if self.models:
                self.ready = True
                dual_logger.info(f"🎉 {len(self.models)} modelos cargados exitosamente")
                return True
            else:
                dual_logger.error("❌ No se cargaron modelos. La aplicación no podrá hacer predicciones.")
                return False
                
        except Exception as e:
            dual_logger.error(f"❌ Error general al cargar modelos: {str(e)}", exc_info=True)
            return False
    
    def preprocess_features(self, df):
        """Aplicar el mismo preprocesamiento que durante el entrenamiento"""
        try:
            df_processed = df.copy()
            dual_logger.debug(f"Columnas iniciales en preprocess_features: {df_processed.columns.tolist()}")
            
            # 1. Imputación si es necesaria
            if self.imputer:
                df_processed = self.imputer.transform(df_processed) 
                dual_logger.debug(f"Columnas después de imputer: {df_processed.columns.tolist()}")
            
            # 2. Ingeniería de características (aplicar las mismas transformaciones)
            df_enhanced = self._create_advanced_features(df_processed)
            dual_logger.debug(f"Columnas después de _create_advanced_features: {df_enhanced.columns.tolist()}")

            # 3. Transformación de variables sesgadas (si aplica)
            if self.power_transformer and self.metadata.get('skewed_columns'):
                skewed_cols = [col for col in self.metadata['skewed_columns'] if col in df_enhanced.columns]
                if skewed_cols:
                    df_enhanced[skewed_cols] = self.power_transformer.transform(df_enhanced[skewed_cols])
                    dual_logger.debug(f"Columnas después de power_transformer: {df_enhanced.columns.tolist()}")
            
            # 4. Escalado
            if self.scaler:
                df_scaled = self.scaler.transform(df_enhanced)
                dual_logger.debug(f"Columnas después de scaler: {df_scaled.columns.tolist()}")
            else:
                df_scaled = df_enhanced
            
            # 5. Selección de características
            if self.feature_selector:
                df_selected = self.feature_selector.transform(df_scaled)
                df_final = df_selected 
                dual_logger.debug(f"Columnas después de feature_selector: {df_final.columns.tolist()}")
            else:
                df_final = df_scaled
            
            # --- CRUCIAL: REINDEXAR PARA ASEGURAR EL ORDEN Y LA PRESENCIA DE CARACTERÍSTICAS ---
            if self.feature_names:
                # Asegura que df_final contenga todas las columnas de self.feature_names,
                # rellenando con 0 si faltan y descartando las que no están en self.feature_names.
                df_final = df_final.reindex(columns=self.feature_names, fill_value=0)
                dual_logger.debug(f"Columnas finales después de reindexado: {df_final.columns.tolist()}")
            else:
                dual_logger.warning("No se proporcionaron nombres de características finales en la metadata. La predicción puede ser inconsistente.")
            
            return df_final
            
        except Exception as e:
            dual_logger.error(f"Error en preprocesamiento: {str(e)}", exc_info=True)
            raise e
    
    def _create_advanced_features(self, df):
        """Aplicar la misma ingeniería de características que en el entrenamiento"""
        df_enhanced = df.copy()
        
        try:
            # 1. Características polinómicas clave
            key_features = ['rm', 'lstat', 'ptratio', 'dis', 'tax', 'crim', 'age'] 
            for feature in key_features:
                if feature in df_enhanced.columns:
                    df_enhanced[f'{feature}_squared'] = df_enhanced[feature] ** 2
                    # Manejo de valores no positivos para sqrt y log
                    df_enhanced[f'{feature}_sqrt'] = np.sqrt(np.abs(df_enhanced[feature]))
                    df_enhanced[f'{feature}_log'] = np.log1p(np.abs(df_enhanced[feature]))
                    # Evitar división por cero
                    df_enhanced[f'{feature}_inv'] = 1 / (df_enhanced[feature] + 1e-8)
            
            # 2. Interacciones importantes
            interactions = [
                ('rm', 'lstat'), ('crim', 'dis'), ('tax', 'ptratio'), 
                ('nox', 'dis'), ('age', 'rm')
            ]
            
            for feat1, feat2 in interactions:
                if all(col in df_enhanced.columns for col in [feat1, feat2]):
                    df_enhanced[f'{feat1}_{feat2}_interaction'] = df_enhanced[feat1] * df_enhanced[feat2]
            
            # 3. Ratios significativos
            ratio_combinations = [
                ('rm', 'age'), ('tax', 'rm'), ('crim', 'dis'), 
                ('lstat', 'rm'), ('ptratio', 'rm')
            ]
            
            for num, den in ratio_combinations:
                if all(col in df_enhanced.columns for col in [num, den]):
                    # Evitar división por cero
                    df_enhanced[f'{num}_per_{den}'] = df_enhanced[num] / (df_enhanced[den] + 1e-8)
            
            # 4. Características categóricas binarias (USAR UMBRALES DE METADATA)
            # Asegúrate que metadata contenga los umbrales específicos calculados durante el entrenamiento.
            if 'crim' in df_enhanced.columns and 'feature_engineering_thresholds' in self.metadata and 'crim_quantile_75' in self.metadata['feature_engineering_thresholds']:
                df_enhanced['high_crime'] = (df_enhanced['crim'] > self.metadata['feature_engineering_thresholds']['crim_quantile_75']).astype(int)
            else: # Fallback si no hay umbral en metadata para 'crim'
                 if 'crim' in df_enhanced.columns:
                     dual_logger.warning("Umbral 'crim_quantile_75' no encontrado en metadata, usando cálculo dinámico (solo para desarrollo/pruebas).")
                     df_enhanced['high_crime'] = (df_enhanced['crim'] > df_enhanced['crim'].quantile(0.75)).astype(int)
            
            if 'rm' in df_enhanced.columns and 'feature_engineering_thresholds' in self.metadata and 'rm_large_threshold' in self.metadata['feature_engineering_thresholds']: 
                df_enhanced['large_rooms'] = (df_enhanced['rm'] > self.metadata['feature_engineering_thresholds']['rm_large_threshold']).astype(int)
            else: # Fallback si no hay umbral en metadata para 'rm'
                 if 'rm' in df_enhanced.columns:
                     dual_logger.warning("Umbral 'rm_large_threshold' no encontrado en metadata, usando valor por defecto 7 (solo para desarrollo/pruebas).")
                     df_enhanced['large_rooms'] = (df_enhanced['rm'] > 7).astype(int) 

            if 'lstat' in df_enhanced.columns and 'feature_engineering_thresholds' in self.metadata and 'lstat_quantile_75' in self.metadata['feature_engineering_thresholds']:
                df_enhanced['low_status'] = (df_enhanced['lstat'] > self.metadata['feature_engineering_thresholds']['lstat_quantile_75']).astype(int)
            else: # Fallback si no hay umbral en metadata para 'lstat'
                 if 'lstat' in df_enhanced.columns:
                     dual_logger.warning("Umbral 'lstat_quantile_75' no encontrado en metadata, usando cálculo dinámico (solo para desarrollo/pruebas).")
                     df_enhanced['low_status'] = (df_enhanced['lstat'] > df_enhanced['lstat'].quantile(0.75)).astype(int)
            
            # 5. Índices compuestos
            if all(col in df_enhanced.columns for col in ['rm', 'lstat', 'ptratio']):
                df_enhanced['livability_index'] = (
                    df_enhanced['rm'] * 0.4 - 
                    df_enhanced['lstat'] * 0.3 - 
                    df_enhanced['ptratio'] * 0.3
                )
            
            return df_enhanced
            
        except Exception as e:
            dual_logger.error(f"Error en ingeniería de características: {str(e)}", exc_info=True)
            raise e 
    
    def predict(self, input_data, model_name):
        """Hacer predicción con el modelo especificado"""
        try:
            if not self.is_ready():
                raise Exception("Predictor no está listo")
            
            if model_name not in self.models:
                raise Exception(f"Modelo {model_name} no disponible")
            
            # Preprocesar datos
            processed_data = self.preprocess_features(input_data)
            dual_logger.debug(f"Datos preprocesados para predicción: Columnas: {processed_data.columns.tolist()}, Shape: {processed_data.shape}")
            
            # Hacer predicción
            model = self.models[model_name]
            prediction = model.predict(processed_data)
            
            return prediction[0] if len(prediction) == 1 else prediction
            
        except Exception as e:
            dual_logger.error(f"Error en predicción: {str(e)}", exc_info=True)
            raise e
    
    def get_available_models(self):
        """Obtener lista de modelos disponibles"""
        return list(self.models.keys())
    
    def get_feature_names(self):
        """Obtener nombres de características originales para el formulario"""
        # Devolver características originales del dataset Boston Housing
        original_features = [
            'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 
            'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat'
        ]
        return original_features # Estas son las características de entrada RAW (del formulario)
    
    def get_model_metrics(self):
        """Obtener métricas de rendimiento de los modelos"""
        if 'metrics' in self.metadata:
            return self.metadata['metrics']
        return {}
    
    def generate_comparison_plot(self):
        """Generar gráfico de comparación de modelos"""
        try:
            metrics = self.get_model_metrics()
            if not metrics:
                dual_logger.warning("No hay métricas disponibles para generar el gráfico.")
                return None
            
            # Preparar datos para el gráfico
            model_names = list(metrics.keys())
            r2_scores = [metrics[model].get('r2_test', 0) for model in model_names] 
            rmse_scores = [metrics[model].get('rmse', 0) for model in model_names]
            
            # Crear figura con subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Gráfico de R² Score
            bars1 = ax1.bar(model_names, r2_scores, color='skyblue', alpha=0.8)
            ax1.set_title('Comparación de R² Score por Modelo', fontsize=14, fontweight='bold')
            ax1.set_ylabel('R² Score')
            ax1.set_ylim(0, 1)
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            
            # Añadir valores en las barras
            for bar, score in zip(bars1, r2_scores):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                 f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Gráfico de RMSE
            bars2 = ax2.bar(model_names, rmse_scores, color='lightcoral', alpha=0.8)
            ax2.set_title('Comparación de RMSE por Modelo', fontsize=14, fontweight='bold')
            ax2.set_ylabel('RMSE')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            # Añadir valores en las barras
            for bar, score in zip(bars2, rmse_scores):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                 f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Convertir a base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
            
        except Exception as e:
            dual_logger.error(f"Error generando gráfico: {str(e)}", exc_info=True)
            return None
    
    def is_ready(self):
        """Verificar si el predictor está listo para usar"""
        return self.ready and len(self.models) > 0

# Inicializar el predictor globalmente
predictor = HousingPredictor()
app_ready = False

# Inicializar el chatbot globalmente
chatbot = FAQChatbot()

def initialize_app():
    """Inicializar la aplicación cargando los modelos y configurando monitoreo"""
    global app_ready, monitor # Asegúrate de que 'monitor' sea global si se modifica
    try:
        dual_logger.info("🚀 Inicializando aplicación Housing Predictor...")
        dual_logger.info("Iniciando carga de modelos...")
        
        success = predictor.load_trained_models()
        if success and predictor.is_ready():
            app_ready = True
            
            # === CONFIGURAR MONITOREO CON DATOS DE REFERENCIA ===
            try:
                reference_data_path = 'models/reference_data.pkl'
                baseline_metrics_path = 'models/baseline_metrics.pkl'
                
                if os.path.exists(reference_data_path) and os.path.exists(baseline_metrics_path):
                    reference_data = joblib.load(reference_data_path)
                    baseline_metrics = joblib.load(baseline_metrics_path)
                    monitor.set_reference_data(reference_data, baseline_metrics)
                    dual_logger.info("✅ Sistema de monitoreo configurado con datos de referencia.")
                else:
                    dual_logger.warning("⚠️ Datos de referencia (reference_data.pkl o baseline_metrics.pkl) no encontrados. El monitoreo funcionará sin una línea base completa.")
                    
            except Exception as monitor_error:
                dual_logger.error(f"❌ Error configurando monitoreo: {str(monitor_error)}", exc_info=True)
                dual_logger.warning("⚠️ Monitoreo iniciado sin configuración completa. Verifique los archivos de referencia.")
            
            dual_logger.info("✅ Aplicación lista!")
            dual_logger.info(f"📊 Modelos cargados: {predictor.get_available_models()}")
            dual_logger.info("Aplicación inicializada correctamente.")
        else:
            dual_logger.error("❌ Error: No se pudieron cargar los modelos.")
            dual_logger.error("Fallo en la carga de modelos.")
            app_ready = False
    except Exception as e:
        dual_logger.error(f"❌ Error inicializando aplicación: {str(e)}", exc_info=True)
        app_ready = False

# Función para ejecutar check_logs_and_send_email periódicamente
def periodic_log_check():
    """Ejecuta el check de logs cada 10 minutos"""
    try:
        dual_logger.info("Ejecutando check periódico de logs...")
        check_logs_and_send_email()
        dual_logger.info("Check periódico de logs completado")
    except Exception as e:
        dual_logger.error(f"Error en check periódico de logs: {str(e)}", exc_info=True)
    
    # Programar la próxima ejecución (10 minutos)
    # Se usa 600.0 segundos = 10 minutos
    timer = Timer(600.0, periodic_log_check)  
    timer.daemon = True # Permite que el programa principal se cierre aunque este hilo esté activo
    timer.start()

# Función para generar logs de prueba
def generate_test_logs():
    """Genera algunos logs de prueba para verificar el sistema"""
    dual_logger.info("=== SISTEMA DE LOGGING INICIALIZADO ===")
    dual_logger.info("Aplicación Flask iniciada correctamente")
    dual_logger.info("Sistema de monitoreo de logs configurado")
    dual_logger.warning("Este es un mensaje de warning de prueba.")
    dual_logger.error("Este es un mensaje de ERROR de prueba (no es un error real).", exc_info=False) 
    dual_logger.info("Logs de prueba generados - El sistema está funcionando.")


# Inicializar al importar el módulo (cuando Gunicorn carga app.py)
initialize_app()

# =============================================================================
# MANEJO DE ERRORES
# =============================================================================

@app.errorhandler(404)
def not_found_error(error):
    dual_logger.warning(f"Error 404: Endpoint no encontrado - {request.path}")
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    dual_logger.error(f"Error interno del servidor (500): {str(error)}", exc_info=True)
    return jsonify({'error': 'Error interno del servidor'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    dual_logger.error(f"Excepción no manejada: {str(e)}", exc_info=True)
    return jsonify({'error': 'Error inesperado en el servidor'}), 500


# =============================================================================
# RUTAS DE LA APLICACIÓN
# =============================================================================

@app.route('/')
def index():
    """Página principal de bienvenida"""
    try:
        if not app_ready:
            dual_logger.warning("Intento de acceso a '/' con modelos no cargados")
            return render_template('error.html', 
                                   error="Los modelos de predicción no están disponibles. Por favor, contacte al administrador."), 503
        
        # Información para mostrar en la página de bienvenida
        context = {
            'models_count': len(predictor.models),
            'available_models': predictor.get_available_models(),
            'features_count': len(predictor.get_feature_names()),
            'app_ready': True
        }
        dual_logger.info("Página principal accedida")
        return render_template('index.html', **context)
        
    except Exception as e:
        dual_logger.error(f"Error en página principal ('/'): {str(e)}", exc_info=True)
        return render_template('error.html', 
                               error=f"Error al cargar la aplicación: {str(e)}"), 500

@app.route('/prediccion')
def prediccion_page():
    """Página del formulario de predicción"""
    try:
        if not app_ready:
            dual_logger.warning("Intento de acceso a '/prediccion' con modelos no cargados")
            return render_template('error.html', 
                                   error="Los modelos de predicción no están disponibles. Por favor, contacte al administrador."), 503
        
        # Obtener información necesaria para el formulario
        feature_names = predictor.get_feature_names() # Estas son las características RAW esperadas del input
        available_models = predictor.get_available_models()
        
        if not feature_names:
            dual_logger.error("No se pudieron obtener los nombres de las características para el formulario de predicción (lista vacía).")
            raise Exception("No se pudieron obtener los nombres de las características para el formulario de predicción")
        
        # Información de características para ayudar al usuario
        feature_info = {
            'crim': {'label': 'Tasa de Criminalidad', 'unit': 'per cápita por ciudad', 'range': '0.006-89'},
            'zn': {'label': 'Zonificación Residencial (%)', 'unit': '% lotes > 25,000 sq ft', 'range': '0-100'},
            'indus': {'label': 'Proporción Industrial (%)', 'unit': '% acres industriales', 'range': '0-28'},
            'chas': {'label': 'Cerca del Río Charles', 'unit': '1=Sí, 0=No', 'range': '0-1'},
            'nox': {'label': 'Concentración de Óxidos Nítricos', 'unit': 'partes por 10 millones', 'range': '0.38-0.87'},
            'rm': {'label': 'Promedio de Habitaciones', 'unit': 'habitaciones por vivienda', 'range': '3-9'},
            'age': {'label': 'Proporción de Unidades Viejas', 'unit': '% construidas antes 1940', 'range': '2-100'},
            'dis': {'label': 'Distancia a Centros de Empleo', 'unit': 'distancia ponderada', 'range': '1-12'},
            'rad': {'label': 'Índice de Accesibilidad', 'unit': 'índice carreteras radiales', 'range': '1-24'},
            'tax': {'label': 'Tasa de Impuesto', 'unit': 'por $10,000', 'range': '187-711'},
            'ptratio': {'label': 'Ratio Alumno-Profesor', 'unit': 'por ciudad', 'range': '12-22'},
            'b': {'label': 'Proporción de Población Negra', 'unit': 'índice calculado', 'range': '0.32-396.9'},
            'lstat': {'label': '% Población de Estatus Bajo', 'unit': 'porcentaje', 'range': '1.73-37.97'}
        }
        dual_logger.info("Página de predicción accedida")
        return render_template('prediccion.html', **context)
        
    except Exception as e:
        dual_logger.error(f"Error en página de predicción ('/prediccion'): {str(e)}", exc_info=True)
        return render_template('error.html', 
                               error=f"Error al cargar la página de predicción: {str(e)}"), 500

@app.route('/monitoreo')
def monitoreo_page():
    """Página del dashboard de monitoreo"""
    try:
        if not app_ready:
            dual_logger.warning("Intento de acceso a '/monitoreo' con modelos no cargados")
            return render_template('error.html', 
                                   error="Los modelos de predicción no están disponibles."), 503
        
        # Obtener estadísticas de monitoreo básicas
        try:
            recent_predictions = len(monitor.get_recent_predictions(7))
            drift_alerts = len(monitor.get_drift_alerts(7))
            feedback_summary = monitor.get_user_feedback_summary(30)
        except Exception as e:
            dual_logger.warning(f"Error obteniendo estadísticas iniciales de monitoreo: {str(e)}")
            recent_predictions = 0
            drift_alerts = 0
            feedback_summary = {
                'total_feedback': 0,
                'avg_rating': 0,
                'positive_feedback': 0,
                'satisfaction_rate': 0
            }
        
        context = {
            'recent_predictions': recent_predictions,
            'drift_alerts': drift_alerts,
            'feedback_summary': feedback_summary,
            'app_ready': app_ready,
            'current_year': datetime.now().year,
            'models_count': len(predictor.models) if app_ready else 0
        }
        dual_logger.info("Página de monitoreo accedida")
        return render_template('monitoreo.html', **context)
        
    except Exception as e:
        dual_logger.error(f"Error en página de monitoreo: {str(e)}", exc_info=True)
        return render_template('error.html', 
                               error=f"Error al cargar monitoreo: {str(e)}"), 500

@app.route('/reporte')
def reporte_page():
    """Página de reportes detallados"""
    try:
        if not app_ready:
            dual_logger.warning("Intento de acceso a '/reporte' con modelos no cargados")
            return render_template('error.html', 
                                   error="Los modelos de predicción no están disponibles."), 503
        
        report = monitor.generate_monitoring_report()
        
        if not isinstance(report, dict):
            dual_logger.error("❌ Reporte no tiene formato válido (se esperaba un dict)")
            return render_template('error.html', 
                                   error="El reporte no está disponible en este momento."), 500

        # Verifica que todas las claves necesarias estén presentes
        for clave in ['resumen', 'drift', 'feedback']:
            if clave not in report:
                dual_logger.error(f"❌ El reporte no contiene la sección requerida: {clave}")
                return render_template('error.html', 
                                       error=f"El reporte está incompleto: falta {clave}."), 500
        
        context = {
            'report': report,
            'app_ready': True
        }
        dual_logger.info("Página de reportes accedida")
        return render_template('reporte.html', **context)
        
    except Exception as e:
        dual_logger.error(f"Error en página de reportes: {str(e)}", exc_info=True)
        return render_template('error.html', 
                               error=f"Error al cargar reportes: {str(e)}"), 500

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint para hacer predicciones (versión con monitoreo)"""
    if not app_ready:
        dual_logger.warning("Intento de acceso a '/api/predict' con modelos no cargados")
        return jsonify({
            'success': False,
            'error': 'Modelos no disponibles. El servidor no está completamente inicializado.'
        }), 503
    
    try:
        data = request.json
        if not data:
            dual_logger.warning("Solicitud de predicción sin datos JSON.")
            return jsonify({
                'success': False,
                'error': 'No se recibieron datos'
            }), 400
        
        features_input = data.get('features') # 'features' es una lista de valores
        model_type = data.get('model')
        
        if not features_input or not model_type:
            dual_logger.warning(f"Faltan parámetros requeridos: features={features_input is None}, model={model_type is None}")
            return jsonify({
                'success': False,
                'error': 'Faltan parámetros requeridos (features, model)'
            }), 400
        
        # Verificar que el modelo esté disponible
        available_models = predictor.get_available_models()
        if model_type not in available_models:
            dual_logger.warning(f"Modelo solicitado '{model_type}' no disponible. Modelos disponibles: {available_models}")
            return jsonify({
                'success': False,
                'error': f'Modelo {model_type} no disponible. Modelos disponibles: {available_models}'
            }), 400
        
        # Validar características (usando los nombres de características RAW esperados)
        expected_raw_features = predictor.get_feature_names() # Estas son las 13 características RAW
        if len(features_input) != len(expected_raw_features):
            dual_logger.error(f"Conteo de características de entrada incorrecto. Esperadas: {len(expected_raw_features)}, Recibidas: {len(features_input)}")
            return jsonify({
                'success': False,
                'error': f'Se esperan {len(expected_raw_features)} características, se recibieron {len(features_input)}'
            }), 400
        
        # Convertir la lista de valores de características a un DataFrame de pandas con los nombres de características RAW
        feature_df = pd.DataFrame([features_input], columns=expected_raw_features)
        dual_logger.debug(f"DataFrame de entrada RAW creado para predicción: {feature_df.columns.tolist()}")

        # Hacer predicción
        prediction = predictor.predict(feature_df, model_type) # Ahora input_data es un DataFrame
        
        # Formatear precio (convertir de miles a dólares)
        formatted_price = f"${prediction * 1000:,.2f}"
        
        # Calcular intervalo de confianza aproximado (±10%)
        confidence_lower = prediction * 0.9 * 1000
        confidence_upper = prediction * 1.1 * 1000
        confidence_interval = {
            'lower': f"${confidence_lower:,.2f}",
            'upper': f"${confidence_upper:,.2f}"
        }
        
        # === REGISTRAR PREDICCIÓN PARA MONITOREO ===
        try:
            # Crear diccionario de features para el log a partir del DataFrame de entrada RAW
            features_dict = feature_df.iloc[0].to_dict()
            
            prediction_log = PredictionLog(
                timestamp=datetime.now(),
                features=features_dict, # Registra las características RAW de entrada
                prediction=float(prediction),
                model_used=model_type,
                confidence_interval=confidence_interval # Puedes almacenar el diccionario completo si el tipo de datos lo permite en tu DB de monitoreo
            )
            
            prediction_id = monitor.log_prediction(prediction_log)
            dual_logger.info(f"Predicción registrada para monitoreo con ID: {prediction_id}")
            
        except Exception as monitor_error:
            dual_logger.warning(f"Error registrando predicción para monitoreo: {str(monitor_error)}", exc_info=True)
            prediction_id = None
        
        response_data = {
            'success': True,
            'prediction': float(prediction),
            'prediction_thousands': f"{prediction:.2f}k",
            'formatted_price': formatted_price,
            'confidence_interval': confidence_interval,
            'model_used': model_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # Agregar ID de predicción si el monitoreo funcionó
        if prediction_id:
            response_data['prediction_id'] = prediction_id
        
        dual_logger.info(f"Predicción completada exitosamente. Precio: {formatted_price}")
        return jsonify(response_data)
        
    except Exception as e:
        dual_logger.error(f"Error general en /api/predict: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Error en predicción: {str(e)}'
        }), 400


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """API endpoint para interactuar con el chatbot"""
    try:
        data = request.json
        user_message = data.get('message')

        if not user_message:
            dual_logger.warning("Solicitud de chat sin mensaje.")
            return jsonify({'success': False, 'response': 'No se recibió ningún mensaje.'}), 400

        chatbot_response = chatbot.get_response(user_message)
        dual_logger.info("Respuesta del chatbot generada.")
        return jsonify({'success': True, 'response': chatbot_response})

    except Exception as e:
        dual_logger.error(f"Error en la API del chatbot: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'response': 'Ocurrió un error interno al procesar tu mensaje.'}), 500


@app.route('/api/models')
def api_models():
    """API endpoint para obtener modelos disponibles"""
    try:
        if not app_ready:
            dual_logger.warning("Intento de acceso a '/api/models' con modelos no cargados")
            return jsonify({
                'success': False,
                'error': 'Modelos no disponibles. El servidor no está completamente inicializado.'
            }), 503
        
        models = predictor.get_available_models()
        dual_logger.info(f"Modelos disponibles solicitados. Cantidad: {len(models)}")
        return jsonify({
            'success': True,
            'models': models,
            'count': len(models)
        })
        
    except Exception as e:
        dual_logger.error(f"Error obteniendo modelos en '/api/models': {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/features')
def api_features():
    """API endpoint para obtener características del modelo"""
    try:
        if not app_ready:
            dual_logger.warning("Intento de acceso a '/api/features' con modelos no cargados")
            return jsonify({
                'success': False,
                'error': 'Modelos no disponibles. El servidor no está completamente inicializado.'
            }), 503
        
        features = predictor.get_feature_names() # Características RAW esperadas
        dual_logger.info(f"Características solicitadas. Cantidad: {len(features)}")
        return jsonify({
            'success': True,
            'features': features,
            'count': len(features)
        })
        
    except Exception as e:
        dual_logger.error(f"Error obteniendo características en '/api/features': {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/metrics')
def api_metrics():
    """API endpoint para obtener métricas de rendimiento"""
    if not app_ready:
        dual_logger.warning("Intento de acceso a '/api/metrics' con modelos no cargados")
        return jsonify({
            'success': False,
            'error': 'Modelos no disponibles. El servidor no está completamente inicializado.'
        }), 503
    
    try:
        dual_logger.info("Obteniendo métricas de rendimiento del modelo...")
        
        if not predictor.is_ready():
            dual_logger.warning("Predictor no listo al intentar obtener métricas.")
            return jsonify({
                'success': False,
                'error': 'El predictor no está listo. Verifique que los modelos estén entrenados.'
            }), 500
        
        metrics = predictor.get_model_metrics()
        
        if not metrics:
            dual_logger.error("get_model_metrics() devolvió None o vacío. No hay métricas para mostrar.")
            return jsonify({
                'success': False,
                'error': 'No se pudieron obtener las métricas del modelo. Verifique que existan datos de evaluación.'
            }), 500
        
        # Validar estructura de métricas
        if not isinstance(metrics, dict):
            dual_logger.error(f"Las métricas tienen formato incorrecto: {type(metrics)}")
            return jsonify({
                'success': False,
                'error': 'Las métricas tienen un formato incorrecto'
            }), 500
        
        dual_logger.info(f"Métricas obtenidas correctamente: {list(metrics.keys())}")
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'models_count': len(metrics)
        })
        
    except Exception as e:
        dual_logger.error(f"Error obteniendo métricas en '/api/metrics': {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Error interno al obtener métricas: {str(e)}'
        }), 500

@app.route('/api/plots')
def api_plots():
    """API endpoint para generar gráficos de comparación"""
    if not app_ready:
        dual_logger.warning("Intento de acceso a '/api/plots' con modelos no cargados")
        return jsonify({
            'success': False,
            'error': 'Modelos no disponibles. El servidor no está completamente inicializado.'
        }), 503
    
    try:
        dual_logger.info("Generando gráfico de comparación...")
        
        if not predictor.is_ready():
            dual_logger.warning("Predictor no listo al intentar generar gráficos.")
            return jsonify({
                'success': False,
                'error': 'El predictor no está listo. Verifique que los modelos estén entrenados.'
            }), 500
        
        img_base64 = predictor.generate_comparison_plot()
        
        if not img_base64:
            dual_logger.error("generate_comparison_plot() devolvió None o vacío. No se pudo generar el gráfico.")
            return jsonify({
                'success': False,
                'error': 'No se pudo generar el gráfico. Verifique que existan datos de entrenamiento y evaluación.'
            }), 500
        
        if not isinstance(img_base64, str):
            dual_logger.error(f"El gráfico tiene formato incorrecto: {type(img_base64)}")
            return jsonify({
                'success': False,
                'error': 'El gráfico generado tiene un formato incorrecto'
            }), 500
        
        dual_logger.info("Gráfico generado exitosamente")
        
        return jsonify({
            'success': True,
            'plot': img_base64,
            'plots': [img_base64],  # Array para consistencia
            'plot_count': 1,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        dual_logger.error(f"Error generando gráficos en '/api/plots': {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Error interno al generar gráficos: {str(e)}'
        }), 500

@app.route('/api/health')
def api_health():
    """Endpoint de verificación de salud"""
    try:
        models_count = len(predictor.models) if predictor.models else 0
        dual_logger.info("Health check solicitado.")
        return jsonify({
            'status': 'healthy' if app_ready else 'unhealthy',
            'models_loaded': models_count,
            'ready': app_ready,
            'models_available': predictor.get_available_models(),
            'timestamp': datetime.now().isoformat(),
            'predictor_ready': predictor.is_ready()
        })
    except Exception as e:
        dual_logger.error(f"Error en health check: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/monitor/prediction', methods=['POST'])
def api_monitor_prediction():
    """API para registrar una predicción para monitoreo"""
    try:
        data = request.json
        
        # Crear log de predicción
        prediction_log = PredictionLog(
            timestamp=datetime.now(),
            features=data.get('features', {}),
            prediction=data.get('prediction', 0.0),
            model_used=data.get('model_used', 'Unknown'),
            confidence_interval=data.get('confidence_interval'),
            actual_value=data.get('actual_value')
        )
        
        # Registrar en el monitor
        prediction_id = monitor.log_prediction(prediction_log)
        dual_logger.info(f"API: Predicción registrada para monitoreo con ID: {prediction_id}")
        
        return jsonify({
            'success': True,
            'prediction_id': prediction_id,
            'message': 'Predicción registrada para monitoreo'
        })
        
    except Exception as e:
        dual_logger.error(f"Error registrando predicción en API de monitoreo: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/monitor/feedback', methods=['POST'])
def api_monitor_feedback():
    """API para registrar feedback del usuario"""
    try:
        data = request.json
        
        prediction_id = data.get('prediction_id')
        rating = data.get('rating')
        feedback_text = data.get('feedback_text')
        actual_price = data.get('actual_price')
        
        if not prediction_id or rating is None:
            dual_logger.warning("Feedback API: prediction_id y rating son requeridos.")
            return jsonify({
                'success': False,
                'error': 'prediction_id y rating son requeridos'
            }), 400
        
        # Registrar feedback
        monitor.log_user_feedback(prediction_id, rating, feedback_text, actual_price)
        dual_logger.info(f"API: Feedback registrado exitosamente para prediction_id: {prediction_id}")
        
        return jsonify({
            'success': True,
            'message': 'Feedback registrado exitosamente'
        })
        
    except Exception as e:
        dual_logger.error(f"Error registrando feedback en API de monitoreo: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/monitor/drift-check', methods=['POST'])
def api_drift_check():
    """API para verificar drift en nuevos datos"""
    try:
        data = request.json
        
        # Convertir datos a DataFrame
        if 'data' not in data:
            dual_logger.warning("Drift check API: Se requieren datos para verificar drift.")
            return jsonify({
                'success': False,
                'error': 'Se requieren datos para verificar drift'
            }), 400
        
        new_data = pd.DataFrame(data['data'])
        
        # Detectar drift de datos
        data_alerts = monitor.detect_data_drift(new_data)
        
        # Si hay un modelo disponible, también verificar drift de rendimiento
        performance_alerts = []
        if 'targets' in data and len(predictor.models) > 0:
            model = list(predictor.models.values())[0]  # Usar primer modelo disponible
            targets = np.array(data['targets'])
            performance_alerts = monitor.detect_performance_drift(model, new_data, targets)
        
        total_alerts = data_alerts + performance_alerts
        dual_logger.info(f"Drift check API: {len(total_alerts)} alertas de drift detectadas.")
        
        return jsonify({
            'success': True,
            'drift_detected': len(total_alerts) > 0,
            'alerts_count': len(total_alerts),
            'data_alerts': len(data_alerts),
            'performance_alerts': len(performance_alerts),
            'severity_breakdown': {
                'high': len([a for a in total_alerts if a.severity == 'high']),
                'medium': len([a for a in total_alerts if a.severity == 'medium']),
                'low': len([a for a in total_alerts if a.severity == 'low'])
            }
        })
        
    except Exception as e:
        dual_logger.error(f"Error verificando drift en API de monitoreo: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/monitor/stats')
def api_monitor_stats():
    """API para obtener estadísticas de monitoreo"""
    try:
        recent_predictions = monitor.get_recent_predictions(7)
        performance_history = monitor.get_performance_history(30)
        drift_alerts = monitor.get_drift_alerts(7)
        feedback_summary = monitor.get_user_feedback_summary(30)

        dual_logger.info("API: Recolección de datos de monitoreo completada.")

        # Convertir a dict si es un DataFrame o truncar para JSON serializable
        if hasattr(drift_alerts, 'to_dict'): # Si es un DataFrame
            drift_alerts_dict = drift_alerts.head(3).to_dict('records') if not drift_alerts.empty else []
        else: # Si ya es una lista de objetos DriftAlert
            drift_alerts_dict = [alert.__dict__ for alert in drift_alerts[:3]] if drift_alerts else []


        stats = {
            'recent_predictions': {
                'count': len(recent_predictions),
                'last_prediction': recent_predictions.iloc[0]['timestamp'].isoformat() if not recent_predictions.empty else None
            },
            'performance_metrics': {
                'metrics_count': len(performance_history),
                'latest_metrics': performance_history.head(5).to_dict('records') if not performance_history.empty else []
            },
            'drift_status': {
                'total_alerts': len(drift_alerts),
                'critical_alerts': len(drift_alerts[drift_alerts['severity'] == 'high']) if isinstance(drift_alerts, pd.DataFrame) and not drift_alerts.empty else len([a for a in drift_alerts if a.severity == 'high']),
                'recent_alerts': drift_alerts_dict
            },
            'user_feedback': feedback_summary
        }

        dual_logger.debug(f"API: Estadísticas recopiladas (antes de sanear): {stats}")

        sanitized_stats = sanitize_numeric_values(stats)
        dual_logger.info("API: Estadísticas de monitoreo saneadas y listas para enviar.")

        return jsonify({
            'success': True,
            'stats': sanitized_stats,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        dual_logger.error(f"Error en /api/monitor/stats: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/monitor/plots')
def api_monitor_plots():
    """API para generar gráficos de monitoreo"""
    try:
        plots = monitor.generate_monitoring_plots()
        dual_logger.info(f"API: Generados {len(plots)} gráficos de monitoreo.")
        
        return jsonify({
            'success': True,
            'plots': plots,
            'plot_count': len(plots),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        dual_logger.error(f"Error generando gráficos de monitoreo: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/monitor/report')
def api_monitor_report():
    """API para generar reporte completo"""
    try:
        report = monitor.generate_monitoring_report()
        dual_logger.info("API: Reporte de monitoreo completo generado.")
        
        return jsonify({
            'success': True,
            'report': report,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        dual_logger.error(f"Error generando reporte completo: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# =============================================================================
# FUNCIONES DE UTILIDAD (Context Processor)
# =============================================================================

@app.context_processor
def utility_processor():
    """Funciones de utilidad disponibles en todas las plantillas"""
    # logger.debug("Utility processor accedido") # No debería ir aquí, es muy frecuente
    return dict(
        app_ready=app_ready,
        models_count=len(predictor.models) if app_ready else 0,
        current_year=datetime.now().year,
        monitor_available=True # Indicar que el monitoreo está disponible
    )


if __name__ == '__main__':
    dual_logger.info("=== INICIANDO APLICACIÓN FLASK ===")
    dual_logger.info(f"Timestamp de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Generar logs de prueba al iniciar
    generate_test_logs()
        
    # Iniciar el monitoreo periódico de logs
    dual_logger.info("Iniciando monitoreo periódico de logs (cada 10 minutos)...")
    periodic_log_check() # La primera llamada inicia el ciclo

    # Ejecutar la aplicación Flask
    # En un entorno de producción, 'debug=True' debe ser 'debug=False'
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv("PORT", 5000)))

