from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from model import HousingPredictor
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Para usar matplotlib sin GUI
import matplotlib.pyplot as plt
import os
import logging

import sklearn
import numpy
print(f"scikit-learn version: {sklearn.__version__}")
print(f"numpy version: {numpy.__version__}")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Inicializar el predictor globalmente
predictor = HousingPredictor()
app_ready = False

def initialize_app():
    """Inicializar la aplicación cargando los modelos"""
    global app_ready
    try:
        print("🚀 Inicializando aplicación...")
        logger.info("Iniciando carga de modelos...")
        
        success = predictor.load_trained_models()
        if success and predictor.is_ready():
            app_ready = True
            print("✅ Aplicación lista!")
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

@app.route('/')
def index():
    """Página principal"""
    try:
        if not app_ready:
            logger.warning("Intento de acceso con modelos no cargados")
            return render_template('error.html', 
                                error="Los modelos de predicción no están disponibles. Por favor, contacte al administrador."), 503
        
        feature_names = predictor.get_feature_names()
        if not feature_names:
            raise Exception("No se pudieron obtener los nombres de las características")
            
        return render_template('index.html', features=feature_names)
        
    except Exception as e:
        logger.error(f"Error en página principal: {str(e)}")
        return render_template('error.html', 
                             error=f"Error al cargar la aplicación: {str(e)}"), 500

@app.route('/health')
def health_check():
    """Endpoint de verificación de salud"""
    try:
        models_count = len(predictor.models) if predictor.models else 0
        return jsonify({
            'status': 'healthy' if app_ready else 'unhealthy',
            'models_loaded': models_count,
            'ready': app_ready,
            'models_available': list(predictor.models.keys()) if predictor.models else []
        })
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para hacer predicciones"""
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
        if model_type not in predictor.models:
            available_models = list(predictor.models.keys())
            return jsonify({
                'success': False,
                'error': f'Modelo {model_type} no disponible. Modelos disponibles: {available_models}'
            }), 400
        
        # Convertir a DataFrame
        feature_df = pd.DataFrame([features])
        
        # Hacer predicción
        prediction = predictor.predict(feature_df, model_type)
        
        # Formatear precio
        formatted_price = f"${prediction * 1000:,.2f}"
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'formatted_price': formatted_price,
            'model_used': model_type
        })
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error en predicción: {str(e)}'
        }), 400

@app.route('/model_performance')
def model_performance():
    """Endpoint para obtener métricas de rendimiento"""
    if not app_ready:
        return jsonify({
            'success': False,
            'error': 'Modelos no disponibles'
        }), 503
    
    try:
        metrics = predictor.get_model_metrics()
        if not metrics:
            return jsonify({
                'success': False,
                'error': 'No se pudieron obtener las métricas'
            }), 500
            
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error obteniendo métricas: {str(e)}'
        }), 400

@app.route('/plots')
def generate_plots():
    """Endpoint para generar gráficos de comparación"""
    if not app_ready:
        return jsonify({
            'success': False,
            'error': 'Modelos no disponibles'
        }), 503
    
    try:
        # Generar gráfico
        img_base64 = predictor.generate_comparison_plot()
        
        if not img_base64:
            return jsonify({
                'success': False,
                'error': 'No se pudo generar el gráfico'
            }), 500
        
        return jsonify({
            'success': True,
            'plot': img_base64
        })
        
    except Exception as e:
        logger.error(f"Error generando gráficos: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Error generando gráficos: {str(e)}'
        }), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Error interno del servidor: {str(error)}")
    return jsonify({'error': 'Error interno del servidor'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Excepción no manejada: {str(e)}")
    return jsonify({'error': 'Error inesperado en el servidor'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode, host='0.0.0.0', port=port)