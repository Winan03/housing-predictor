from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import logging
import base64
import io
from datetime import datetime
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class HousingPredictor:
    """Clase para manejar predicciones de precios de vivienda"""
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.power_transformer = None
        self.imputer = None
        self.metadata = {}
        self.feature_names = []
        self.ready = False
        
    def load_trained_models(self):
        """Cargar todos los modelos entrenados"""
        try:
            logger.info("🔄 Cargando modelos entrenados...")
            
            models_dir = 'models'
            if not os.path.exists(models_dir):
                logger.error(f"❌ Directorio {models_dir} no encontrado")
                return False
            
            # Cargar metadata
            metadata_path = os.path.join(models_dir, 'metadata.pkl')
            if os.path.exists(metadata_path):
                self.metadata = joblib.load(metadata_path)
                self.feature_names = self.metadata.get('feature_names', [])
                logger.info(f"✅ Metadata cargada: {len(self.feature_names)} características")
            else:
                logger.warning("⚠️ Archivo metadata.pkl no encontrado")
            
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
                    logger.info(f"✅ {attr_name} cargado")
                else:
                    logger.warning(f"⚠️ {file_name} no encontrado")
            
            # Cargar modelos
            model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
            
            for model_file in model_files:
                try:
                    model_path = os.path.join(models_dir, model_file)
                    model = joblib.load(model_path)
                    
                    # Extraer nombre del modelo
                    model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
                    self.models[model_name] = model
                    logger.info(f"✅ Modelo cargado: {model_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Error cargando {model_file}: {str(e)}")
            
            # Cargar ensemble si existe
            ensemble_path = os.path.join(models_dir, 'voting_ensemble.pkl')
            if os.path.exists(ensemble_path):
                self.models['Voting Ensemble'] = joblib.load(ensemble_path)
                logger.info("✅ Voting Ensemble cargado")
            
            if self.models:
                self.ready = True
                logger.info(f"🎉 {len(self.models)} modelos cargados exitosamente")
                return True
            else:
                logger.error("❌ No se cargaron modelos")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {str(e)}")
            return False
    
    def preprocess_features(self, df):
        """Aplicar el mismo preprocesamiento que durante el entrenamiento"""
        try:
            df_processed = df.copy()
            
            # 1. Imputación si es necesaria
            if self.imputer and df_processed.isnull().sum().sum() > 0:
                df_processed = pd.DataFrame(
                    self.imputer.transform(df_processed), 
                    columns=df_processed.columns
                )
            
            # 2. Ingeniería de características (aplicar las mismas transformaciones)
            df_enhanced = self._create_advanced_features(df_processed)
            
            # 3. Transformación de variables sesgadas
            if self.power_transformer and self.metadata.get('skewed_columns'):
                skewed_cols = [col for col in self.metadata['skewed_columns'] 
                             if col in df_enhanced.columns]
                if skewed_cols:
                    df_enhanced[skewed_cols] = self.power_transformer.transform(df_enhanced[skewed_cols])
            
            # 4. Escalado
            if self.scaler:
                df_scaled = pd.DataFrame(
                    self.scaler.transform(df_enhanced),
                    columns=df_enhanced.columns
                )
            else:
                df_scaled = df_enhanced
            
            # 5. Selección de características
            if self.feature_selector:
                df_selected = self.feature_selector.transform(df_scaled)
                # Convertir de nuevo a DataFrame con nombres de características seleccionadas
                if self.feature_names:
                    df_final = pd.DataFrame(df_selected, columns=self.feature_names)
                else:
                    df_final = pd.DataFrame(df_selected)
            else:
                df_final = df_scaled
            
            return df_final
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {str(e)}")
            raise e
    
    def _create_advanced_features(self, df):
        """Aplicar la misma ingeniería de características que en el entrenamiento"""
        df_enhanced = df.copy()
        
        try:
            # 1. Características polinómicas clave
            key_features = ['rm', 'lstat', 'ptratio', 'dis', 'tax', 'crim']
            for feature in key_features:
                if feature in df_enhanced.columns:
                    df_enhanced[f'{feature}_squared'] = df_enhanced[feature] ** 2
                    df_enhanced[f'{feature}_sqrt'] = np.sqrt(np.abs(df_enhanced[feature]))
                    df_enhanced[f'{feature}_log'] = np.log1p(np.abs(df_enhanced[feature]))
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
                    df_enhanced[f'{num}_per_{den}'] = df_enhanced[num] / (df_enhanced[den] + 1e-8)
            
            # 4. Características categóricas binarias
            if 'crim' in df_enhanced.columns:
                df_enhanced['high_crime'] = (df_enhanced['crim'] > df_enhanced['crim'].quantile(0.75)).astype(int)
            
            if 'rm' in df_enhanced.columns:
                df_enhanced['large_rooms'] = (df_enhanced['rm'] > 7).astype(int)
            
            if 'lstat' in df_enhanced.columns:
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
            logger.error(f"Error en ingeniería de características: {str(e)}")
            return df_enhanced
    
    def predict(self, input_data, model_name):
        """Hacer predicción con el modelo especificado"""
        try:
            if not self.is_ready():
                raise Exception("Predictor no está listo")
            
            if model_name not in self.models:
                raise Exception(f"Modelo {model_name} no disponible")
            
            # Preprocesar datos
            processed_data = self.preprocess_features(input_data)
            
            # Hacer predicción
            model = self.models[model_name]
            prediction = model.predict(processed_data)
            
            return prediction[0] if len(prediction) == 1 else prediction
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
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
        return original_features
    
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
                return None
            
            # Preparar datos para el gráfico
            model_names = list(metrics.keys())
            r2_scores = [metrics[model]['r2_test'] for model in model_names]
            rmse_scores = [metrics[model]['rmse'] for model in model_names]
            
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
            logger.error(f"Error generando gráfico: {str(e)}")
            return None
    
    def is_ready(self):
        """Verificar si el predictor está listo para usar"""
        return self.ready and len(self.models) > 0

# Inicializar el predictor globalmente
predictor = HousingPredictor()
app_ready = False

def initialize_app():
    """Inicializar la aplicación cargando los modelos"""
    global app_ready
    try:
        print("🚀 Inicializando aplicación Housing Predictor...")
        logger.info("Iniciando carga de modelos...")
        
        success = predictor.load_trained_models()
        if success and predictor.is_ready():
            app_ready = True
            print("✅ Aplicación lista!")
            print(f"📊 Modelos cargados: {predictor.get_available_models()}")
            logger.info("Aplicación inicializada correctamente")
        else:
            print("❌ Error: No se pudieron cargar los modelos")
            logger.error("Fallo en la carga de modelos")
            app_ready = False
    except Exception as e:
        print(f"❌ Error inicializando aplicación: {str(e)}")
        logger.error(f"Error en inicialización: {str(e)}")
        app_ready = False

# Inicializar al importar
initialize_app()

# =============================================================================
# RUTAS DE LA APLICACIÓN
# =============================================================================

@app.route('/')
def index():
    """Página principal de bienvenida"""
    try:
        if not app_ready:
            logger.warning("Intento de acceso con modelos no cargados")
            return render_template('error.html', 
                                 error="Los modelos de predicción no están disponibles. Por favor, contacte al administrador."), 503
        
        # Información para mostrar en la página de bienvenida
        context = {
            'models_count': len(predictor.models),
            'available_models': predictor.get_available_models(),
            'features_count': len(predictor.get_feature_names()),
            'app_ready': True
        }
        
        return render_template('index.html', **context)
        
    except Exception as e:
        logger.error(f"Error en página principal: {str(e)}")
        return render_template('error.html', 
                             error=f"Error al cargar la aplicación: {str(e)}"), 500

@app.route('/prediccion')
def prediccion_page():
    """Página del formulario de predicción"""
    try:
        if not app_ready:
            logger.warning("Intento de acceso a predicción con modelos no cargados")
            return render_template('error.html', 
                                 error="Los modelos de predicción no están disponibles. Por favor, contacte al administrador."), 503
        
        # Obtener información necesaria para el formulario
        feature_names = predictor.get_feature_names()
        available_models = predictor.get_available_models()
        
        if not feature_names:
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
        
        context = {
            'features': feature_names,
            'feature_info': feature_info,
            'available_models': available_models,
            'app_ready': True,
            'default_model': available_models[0] if available_models else None
        }
        
        return render_template('prediccion.html', **context)
        
    except Exception as e:
        logger.error(f"Error en página de predicción: {str(e)}")
        return render_template('error.html', 
                             error=f"Error al cargar la página de predicción: {str(e)}"), 500

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint para hacer predicciones"""
    if not app_ready:
        return jsonify({
            'success': False,
            'error': 'Modelos no disponibles'
        }), 503
    
    try:
        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No se recibieron datos'
            }), 400
        
        features = data.get('features')
        model_type = data.get('model')
        
        if not features or not model_type:
            return jsonify({
                'success': False,
                'error': 'Faltan parámetros requeridos (features, model)'
            }), 400
        
        # Verificar que el modelo esté disponible
        available_models = predictor.get_available_models()
        if model_type not in available_models:
            return jsonify({
                'success': False,
                'error': f'Modelo {model_type} no disponible. Modelos disponibles: {available_models}'
            }), 400
        
        # Validar características
        expected_features = predictor.get_feature_names()
        if len(features) != len(expected_features):
            return jsonify({
                'success': False,
                'error': f'Se esperan {len(expected_features)} características, se recibieron {len(features)}'
            }), 400
        
        # Convertir a DataFrame
        feature_df = pd.DataFrame([features], columns=expected_features)
        
        # Hacer predicción
        prediction = predictor.predict(feature_df, model_type)
        
        # Formatear precio (convertir de miles a dólares)
        formatted_price = f"${prediction * 1000:,.2f}"
        
        # Calcular intervalo de confianza aproximado (±10%)
        confidence_lower = prediction * 0.9 * 1000
        confidence_upper = prediction * 1.1 * 1000
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'prediction_thousands': f"{prediction:.2f}k",
            'formatted_price': formatted_price,
            'confidence_interval': {
                'lower': f"${confidence_lower:,.2f}",
                'upper': f"${confidence_upper:,.2f}"
            },
            'model_used': model_type,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error en predicción: {str(e)}'
        }), 400

@app.route('/api/models')
def api_models():
    """API endpoint para obtener modelos disponibles"""
    try:
        if not app_ready:
            return jsonify({
                'success': False,
                'error': 'Modelos no disponibles'
            }), 503
        
        models = predictor.get_available_models()
        return jsonify({
            'success': True,
            'models': models,
            'count': len(models)
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo modelos: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/features')
def api_features():
    """API endpoint para obtener características del modelo"""
    try:
        if not app_ready:
            return jsonify({
                'success': False,
                'error': 'Modelos no disponibles'
            }), 503
        
        features = predictor.get_feature_names()
        return jsonify({
            'success': True,
            'features': features,
            'count': len(features)
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo características: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/metrics')
def api_metrics():
    """API endpoint para obtener métricas de rendimiento"""
    if not app_ready:
        return jsonify({
            'success': False,
            'error': 'Modelos no disponibles. El servidor no está completamente inicializado.'
        }), 503
    
    try:
        logger.info("Obteniendo métricas de rendimiento del modelo...")
        
        if not predictor.is_ready():
            return jsonify({
                'success': False,
                'error': 'El predictor no está listo. Verifique que los modelos estén entrenados.'
            }), 500
        
        metrics = predictor.get_model_metrics()
        
        if not metrics:
            logger.error("get_model_metrics() devolvió None o vacío")
            return jsonify({
                'success': False,
                'error': 'No se pudieron obtener las métricas del modelo. Verifique que existan datos de evaluación.'
            }), 500
        
        # Validar estructura de métricas
        if not isinstance(metrics, dict):
            logger.error(f"Las métricas tienen formato incorrecto: {type(metrics)}")
            return jsonify({
                'success': False,
                'error': 'Las métricas tienen un formato incorrecto'
            }), 500
        
        logger.info(f"Métricas obtenidas correctamente: {list(metrics.keys())}")
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'models_count': len(metrics)
        })
        
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error interno al obtener métricas: {str(e)}'
        }), 500

@app.route('/api/plots')
def api_plots():
    """API endpoint para generar gráficos de comparación"""
    if not app_ready:
        return jsonify({
            'success': False,
            'error': 'Modelos no disponibles. El servidor no está completamente inicializado.'
        }), 503
    
    try:
        logger.info("Generando gráfico de comparación...")
        
        if not predictor.is_ready():
            return jsonify({
                'success': False,
                'error': 'El predictor no está listo. Verifique que los modelos estén entrenados.'
            }), 500
        
        img_base64 = predictor.generate_comparison_plot()
        
        if not img_base64:
            logger.error("generate_comparison_plot() devolvió None o vacío")
            return jsonify({
                'success': False,
                'error': 'No se pudo generar el gráfico. Verifique que existan datos de entrenamiento y evaluación.'
            }), 500
        
        if not isinstance(img_base64, str):
            logger.error(f"El gráfico tiene formato incorrecto: {type(img_base64)}")
            return jsonify({
                'success': False,
                'error': 'El gráfico generado tiene un formato incorrecto'
            }), 500
        
        logger.info("Gráfico generado exitosamente")
        
        return jsonify({
            'success': True,
            'plot': img_base64,
            'plots': [img_base64],  # Array para consistencia
            'plot_count': 1,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generando gráficos: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error interno al generar gráficos: {str(e)}'
        }), 500

@app.route('/api/health')
def api_health():
    """Endpoint de verificación de salud"""
    try:
        models_count = len(predictor.models) if predictor.models else 0
        return jsonify({
            'status': 'healthy' if app_ready else 'unhealthy',
            'models_loaded': models_count,
            'ready': app_ready,
            'models_available': predictor.get_available_models(),
            'timestamp': datetime.now().isoformat(),
            'predictor_ready': predictor.is_ready()
        })
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

# =============================================================================
# MANEJO DE ERRORES
# =============================================================================

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Error interno del servidor: {str(error)}")
    return jsonify({'error': 'Error interno del servidor'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Excepción no manejada: {str(e)}")
    return jsonify({'error': 'Error inesperado en el servidor'}), 500

# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

@app.context_processor
def utility_processor():
    """Funciones de utilidad disponibles en todas las plantillas"""
    return dict(
        app_ready=app_ready,
        models_count=len(predictor.models) if app_ready else 0,
        current_year=datetime.now().year
    )

if __name__ == "__main__":
    print("🏠 Housing Price Predictor")
    print("=" * 40)
    print(f"📊 Estado de la aplicación: {'✅ Lista' if app_ready else '❌ No lista'}")
    if app_ready:
        print(f"🤖 Modelos cargados: {len(predictor.models)}")
        print(f"📋 Modelos disponibles: {', '.join(predictor.get_available_models())}")
    print("=" * 40)
    
    app.run(debug=True, host='0.0.0.0', port=5000)