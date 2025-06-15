import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import sqlite3
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionLog:
    """Estructura para almacenar logs de predicciones"""
    timestamp: datetime
    features: Dict
    prediction: float
    actual_value: Optional[float] = None
    model_used: str = "Unknown"
    confidence_interval: Optional[Dict] = None
    user_feedback: Optional[Dict] = None

def calculate_psi_safe_percentiles(expected: np.array, actual: np.array, buckets: int = 10) -> float:
        """
        Calcular el PSI (Population Stability Index) de forma segura usando percentiles.
        """
        try:
            if len(expected) == 0 or len(actual) == 0:
                return 0.0

            breakpoints = np.linspace(0, 100, buckets + 1)
            bins = np.percentile(expected, breakpoints)
            bins[0] = -np.inf
            bins[-1] = np.inf

            expected_bins = pd.cut(expected, bins, duplicates='drop')
            actual_bins = pd.cut(actual, bins, duplicates='drop')

            expected_dist = expected_bins.value_counts().sort_index()
            actual_dist = actual_bins.value_counts().sort_index()

            expected_dist = expected_dist / expected_dist.sum()
            actual_dist = actual_dist / actual_dist.sum()

            expected_dist = np.array(expected_dist) + 1e-6
            actual_dist = np.array(actual_dist) + 1e-6

            expected_dist = np.nan_to_num(expected_dist, nan=1e-6, posinf=1e-6, neginf=1e-6)
            actual_dist = np.nan_to_num(actual_dist, nan=1e-6, posinf=1e-6, neginf=1e-6)

            psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))

            if np.isnan(psi) or np.isinf(psi):
                psi = 10.0

            return float(psi)

        except Exception as e:
            print(f"⚠️ Error calculando PSI: {str(e)}")
            return 10.0

def detect_data_drift(self, new_data):
    """Detect data drift using various statistical tests"""
    alerts = []
    
    if self.reference_data is None:
        logger.warning("No reference data available for drift detection")
        return alerts
    
    try:
        for column in new_data.columns:
            if column in self.reference_data.columns:
                # Get reference and new data for this column
                ref_values = self.reference_data[column].dropna()
                new_values = new_data[column].dropna()
                
                if len(ref_values) == 0 or len(new_values) == 0:
                    continue
                
                # === PSI TEST (FIXED) ===
                try:
                    # Create bins based on reference data
                    bins = np.histogram_bin_edges(ref_values, bins=10)
                    
                    # Calculate distributions
                    ref_hist, _ = np.histogram(ref_values, bins=bins, density=True)
                    new_hist, _ = np.histogram(new_values, bins=bins, density=True)
                    
                    # Convert to probabilities (normalize)
                    ref_prob = ref_hist / np.sum(ref_hist) if np.sum(ref_hist) > 0 else np.ones_like(ref_hist) / len(ref_hist)
                    new_prob = new_hist / np.sum(new_hist) if np.sum(new_hist) > 0 else np.ones_like(new_hist) / len(new_hist)
                    
                    # Calculate PSI safely
                    psi_value = self.calculate_psi_safe_percentiles(ref_values.to_numpy(), new_values.to_numpy())
                    
                    # Check threshold
                    psi_threshold = 0.2
                    if psi_value > psi_threshold:
                        severity = 'high' if psi_value > 0.5 else 'medium'
                        
                        alert = DriftAlert(
                            timestamp=datetime.now(),
                            drift_type='data',
                            metric_name=f'{column}_psi',
                            current_value=float(psi_value),  # Ensure it's a regular float
                            baseline_value=0.0,
                            threshold_value=psi_threshold,
                            severity=severity,
                            description=f'Data drift detectado en {column}: PSI = {psi_value:.4f}'
                        )
                        alerts.append(alert)
                        
                except Exception as psi_error:
                    logger.warning(f"Error calculating PSI for {column}: {str(psi_error)}")
                
                # === KS TEST ===
                try:
                    from scipy import stats
                    ks_stat, p_value = stats.ks_2samp(ref_values, new_values)
                    
                    # Ensure finite values
                    p_value = float(p_value) if not math.isnan(p_value) and not math.isinf(p_value) else 1.0
                    
                    if p_value < 0.05:  # Significant difference
                        severity = 'high' if p_value < 0.01 else 'medium'
                        
                        alert = DriftAlert(
                            timestamp=datetime.now(),
                            drift_type='data',
                            metric_name=f'{column}_ks_test',
                            current_value=float(1.0 - p_value),  # Convert to drift score
                            baseline_value=0.0,
                            threshold_value=0.05,
                            severity=severity,
                            description=f'Data drift detectado en {column}: KS p-value = {p_value:.4f}'
                        )
                        alerts.append(alert)
                        
                except Exception as ks_error:
                    logger.warning(f"Error calculating KS test for {column}: {str(ks_error)}")
    
    except Exception as e:
        logger.error(f"Error in drift detection: {str(e)}")
    
    return alerts
    

@dataclass
class DriftAlert:
    """Estructura para alertas de drift"""
    timestamp: datetime
    drift_type: str  # 'data', 'concept', 'performance'
    severity: str   # 'low', 'medium', 'high'
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    description: str

class ModelMonitor:
    """Sistema de monitoreo para detectar drift y gestionar rendimiento"""
    
    def __init__(self, db_path: str = "monitoring.db"):
        self.db_path = db_path
        self.reference_data = None
        self.baseline_metrics = {}
        self.drift_thresholds = {
            'ks_test': 0.05,          # Kolmogorov-Smirnov test
            'psi_threshold': 0.2,      # Population Stability Index
            'performance_degradation': 0.1,  # 10% degradación
            'concept_drift': 0.15      # 15% cambio en relaciones
        }
        self.initialize_database()
        
    def initialize_database(self):
        """Inicializar base de datos SQLite para logging"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabla para logs de predicciones
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    features TEXT NOT NULL,
                    prediction REAL NOT NULL,
                    actual_value REAL,
                    model_used TEXT,
                    confidence_interval TEXT,
                    user_feedback TEXT
                )
            ''')
            
            # Tabla para métricas de rendimiento
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    model_name TEXT,
                    data_window TEXT
                )
            ''')
            
            # Tabla para alertas de drift
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drift_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    drift_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    baseline_value REAL NOT NULL,
                    threshold_value REAL NOT NULL,
                    description TEXT
                )
            ''')
            
            # Tabla para feedback de usuarios
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    prediction_id INTEGER,
                    rating INTEGER,
                    feedback_text TEXT,
                    actual_price REAL,
                    FOREIGN KEY (prediction_id) REFERENCES prediction_logs (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Base de datos de monitoreo inicializada")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando base de datos: {str(e)}")
            raise e
    
    def set_reference_data(self, reference_df: pd.DataFrame, baseline_metrics: Dict):
        """Establecer datos de referencia para comparación de drift"""
        self.reference_data = reference_df.copy()
        self.baseline_metrics = baseline_metrics.copy()
        logger.info(f"✅ Datos de referencia establecidos: {len(reference_df)} muestras")
    
    def log_prediction(self, prediction_log: PredictionLog) -> int:
        """Registrar una predicción en la base de datos"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO prediction_logs 
                (timestamp, features, prediction, actual_value, model_used, confidence_interval, user_feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_log.timestamp.isoformat(),
                json.dumps(prediction_log.features),
                prediction_log.prediction,
                prediction_log.actual_value,
                prediction_log.model_used,
                json.dumps(prediction_log.confidence_interval) if prediction_log.confidence_interval else None,
                json.dumps(prediction_log.user_feedback) if prediction_log.user_feedback else None
            ))
            
            prediction_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return prediction_id
            
        except Exception as e:
            logger.error(f"❌ Error logging predicción: {str(e)}")
            raise e
    
    def log_performance_metric(self, metric_name: str, metric_value: float, 
                             model_name: str = None, data_window: str = None):
        """Registrar métrica de rendimiento"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics (timestamp, metric_name, metric_value, model_name, data_window)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                metric_name,
                metric_value,
                model_name,
                data_window
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error logging métrica: {str(e)}")
    
    def log_drift_alert(self, alert: DriftAlert):
        """Registrar alerta de drift"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO drift_alerts 
                (timestamp, drift_type, severity, metric_name, current_value, baseline_value, threshold_value, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.timestamp.isoformat(),
                alert.drift_type,
                alert.severity,
                alert.metric_name,
                alert.current_value,
                alert.baseline_value,
                alert.threshold,
                alert.description
            ))
            
            conn.commit()
            conn.close()
            logger.warning(f"🚨 DRIFT ALERT: {alert.description}")
            
        except Exception as e:
            logger.error(f"❌ Error logging alerta: {str(e)}")
    
    def calculate_psi(self, expected: np.array, actual: np.array, buckets: int = 10) -> float:
        """Calcular Population Stability Index (PSI)"""
        try:
            # Crear bins basados en los datos esperados
            breakpoints = np.linspace(0, 100, buckets + 1)
            breakpoints = np.percentile(expected, breakpoints)
            breakpoints[0] = -float('inf')
            breakpoints[-1] = float('inf')
            
            # Calcular distribuciones
            expected_dist = pd.cut(expected, breakpoints, duplicates='drop').value_counts().sort_index()
            actual_dist = pd.cut(actual, breakpoints, duplicates='drop').value_counts().sort_index()
            
            # Normalizar
            expected_dist = expected_dist / len(expected)
            actual_dist = actual_dist / len(actual)
            
            # Calcular PSI
            # Evitar log(0)
            psi = np.sum((actual_dist - expected_dist) * np.log((actual_dist + 1e-10) / (expected_dist + 1e-10)))
            return psi
            
        except Exception as e:
            logger.error(f"Error calculando PSI: {str(e)}")
            return float('inf')
    
    def detect_performance_drift(self, model, test_data: pd.DataFrame, 
                               test_targets: np.array) -> List[DriftAlert]:
        """Detectar drift en el rendimiento del modelo"""
        alerts = []
        
        try:
            # Hacer predicciones
            predictions = model.predict(test_data)
            
            # Calcular métricas actuales
            current_r2 = r2_score(test_targets, predictions)
            current_rmse = np.sqrt(mean_squared_error(test_targets, predictions))
            current_mae = mean_absolute_error(test_targets, predictions)
            
            # Comparar con baseline
            baseline_r2 = self.baseline_metrics.get('r2_test', 0)
            baseline_rmse = self.baseline_metrics.get('rmse', float('inf'))
            baseline_mae = self.baseline_metrics.get('mae', float('inf'))
            
            # Detectar degradación
            # Asegúrate de que baseline_r2 no sea cero para evitar ZeroDivisionError
            r2_degradation = (baseline_r2 - current_r2) / baseline_r2 if baseline_r2 > 1e-6 else 0
            rmse_increase = (current_rmse - baseline_rmse) / baseline_rmse if baseline_rmse > 1e-6 else 0
            
            threshold = self.drift_thresholds['performance_degradation']
            
            if r2_degradation > threshold:
                severity = 'high' if r2_degradation > 0.2 else 'medium'
                alert = DriftAlert(
                    timestamp=datetime.now(),
                    drift_type='performance',
                    severity=severity,
                    metric_name='r2_degradation',
                    current_value=current_r2,
                    baseline_value=baseline_r2,
                    threshold=threshold,
                    description=f"Degradación de R² detectada: {r2_degradation:.2%}"
                )
                alerts.append(alert)
                self.log_drift_alert(alert)
            
            if rmse_increase > threshold:
                severity = 'high' if rmse_increase > 0.2 else 'medium'
                alert = DriftAlert(
                    timestamp=datetime.now(),
                    drift_type='performance',
                    severity=severity,
                    metric_name='rmse_increase',
                    current_value=current_rmse,
                    baseline_value=baseline_rmse,
                    threshold=threshold,
                    description=f"Aumento de RMSE detectado: {rmse_increase:.2%}"
                )
                alerts.append(alert)
                self.log_drift_alert(alert)
            
            # Log métricas actuales
            self.log_performance_metric('r2_score', current_r2, model_name="Current Model")
            self.log_performance_metric('rmse', current_rmse, model_name="Current Model")
            self.log_performance_metric('mae', current_mae, model_name="Current Model")
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Error detectando performance drift: {str(e)}")
            return alerts
    
    def log_user_feedback(self, prediction_id: int, rating: int, 
                         feedback_text: str = None, actual_price: float = None):
        """Registrar feedback del usuario"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_feedback (timestamp, prediction_id, rating, feedback_text, actual_price)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                prediction_id,
                rating,
                feedback_text,
                actual_price
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Feedback registrado para predicción {prediction_id}")
            
        except Exception as e:
            logger.error(f"❌ Error registrando feedback: {str(e)}")
    
    def get_recent_predictions(self, days: int = 7) -> pd.DataFrame:
        """Obtener predicciones recientes"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT * FROM prediction_logs 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo predicciones: {str(e)}")
            return pd.DataFrame()
    
    def get_performance_history(self, days: int = 30) -> pd.DataFrame:
        """Obtener historial de rendimiento"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT * FROM performance_metrics 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo historial: {str(e)}")
            return pd.DataFrame()
    
    def get_drift_alerts(self, days: int = 7) -> pd.DataFrame:
        """Obtener alertas de drift recientes"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT * FROM drift_alerts 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo alertas: {str(e)}")
            return pd.DataFrame()
    
    def get_user_feedback_summary(self, days: int = 30) -> Dict:
        """Obtener resumen de feedback de usuarios"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT 
                    COUNT(*) as total_feedback,
                    AVG(rating) as avg_rating,
                    COUNT(CASE WHEN rating >= 4 THEN 1 END) as positive_feedback,
                    COUNT(CASE WHEN rating <= 2 THEN 1 END) as negative_feedback,
                    COUNT(CASE WHEN actual_price IS NOT NULL THEN 1 END) as with_actual_price
                FROM user_feedback 
                WHERE timestamp >= datetime('now', '-{} days')
            '''.format(days)
            
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchone()
            conn.close()
            
            return {
                'total_feedback': result[0] or 0,
                'avg_rating': result[1] or 0,
                'positive_feedback': result[2] or 0,
                'negative_feedback': result[3] or 0,
                'with_actual_price': result[4] or 0,
                'satisfaction_rate': (result[2] or 0) / max(result[0] or 1, 1) * 100
            }
            
        except Exception as e:
            logger.error(f"❌ Error obteniendo resumen de feedback: {str(e)}")
            return {}
    
    def generate_monitoring_report(self):
        try:
            recent_preds = self.get_recent_predictions(30)
            feedback = self.get_user_feedback_summary(30)
            drift_alerts = self.get_drift_alerts(30)

            resumen = {
                'total_predicciones': len(recent_preds),
                'promedio_prediccion': round(recent_preds['prediction'].mean(), 2) if not recent_preds.empty else 0.0,
                'modelos_usados': dict(recent_preds['model_used'].value_counts()) if 'model_used' in recent_preds else {}
            }

            drift = {
                'total_alertas': len(drift_alerts),
                'criticas': len(drift_alerts[drift_alerts['severity'] == 'high']) if not drift_alerts.empty else 0,
                'metricas_afectadas': drift_alerts['metric_name'].unique().tolist() if not drift_alerts.empty else []
            }

            feedback_section = {
                'total_feedback': feedback.get('total_feedback', 0),
                'satisfaccion': feedback.get('satisfaction_rate', 0.0),
                'comentarios': []  # Aquí puedes incluir feedback detallado si lo tienes almacenado
            }

            return {
                'resumen': resumen,
                'drift': drift,
                'feedback': feedback_section
            }

        except Exception as e:
            print(f"Error generando el reporte: {str(e)}")
            return {'error': str(e)}

    
    def generate_monitoring_plots(self) -> Dict[str, str]:
        """Generar gráficos de monitoreo con manejo de errores y valores vacíos"""
        plots = {}

        try:
            # === GRÁFICO DE RENDIMIENTO ===
            perf_df = self.get_performance_history(30)

            if not perf_df.empty and perf_df['metric_value'].notnull().any():
                fig, ax = plt.subplots(figsize=(12, 6))
                perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])

                for metric in perf_df['metric_name'].unique():
                    metric_data = perf_df[perf_df['metric_name'] == metric]
                    ax.plot(metric_data['timestamp'], metric_data['metric_value'], 
                            label=metric, marker='o')

                ax.set_title('Evolución de Métricas de Rendimiento')
                ax.set_xlabel('Fecha')
                ax.set_ylabel('Valor de Métrica')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()

                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                plots['performance_history'] = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close()
            else:
                # Gráfico vacío por defecto
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.text(0.5, 0.5, 'Sin métricas de rendimiento disponibles.', fontsize=14,
                        ha='center', va='center')
                ax.axis('off')
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300)
                img_buffer.seek(0)
                plots['performance_history'] = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close()

            # === GRÁFICO DE FEEDBACK DE USUARIO ===
            feedback_summary = self.get_user_feedback_summary(30)

            if feedback_summary.get('total_feedback', 0) > 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Distribución de ratings
                conn = sqlite3.connect(self.db_path)
                rating_query = '''
                    SELECT rating, COUNT(*) as count 
                    FROM user_feedback 
                    WHERE timestamp >= datetime('now', '-30 days')
                    GROUP BY rating
                '''
                rating_df = pd.read_sql_query(rating_query, conn)
                conn.close()

                if not rating_df.empty:
                    ax1.bar(rating_df['rating'], rating_df['count'], color='skyblue')
                    ax1.set_title('Distribución de Ratings de Usuario')
                    ax1.set_xlabel('Rating')
                    ax1.set_ylabel('Cantidad')
                else:
                    ax1.text(0.5, 0.5, 'Sin datos de ratings.', ha='center', va='center')
                    ax1.axis('off')

                # Métricas resumen
                metrics_names = list(feedback_summary.keys())[:4]
                metrics_values = [feedback_summary[key] for key in metrics_names]

                ax2.bar(metrics_names, metrics_values, color='lightcoral')
                ax2.set_title('Resumen de Feedback')
                plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

                plt.tight_layout()

                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                img_buffer.seek(0)
                plots['user_feedback'] = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close()
            else:
                # Gráfico de fallback si no hay feedback
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.text(0.5, 0.5, 'Sin feedback disponible aún.', fontsize=14,
                        ha='center', va='center')
                ax.axis('off')
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=300)
                img_buffer.seek(0)
                plots['user_feedback'] = base64.b64encode(img_buffer.getvalue()).decode()
                plt.close()

            print("📊 Gráficos generados:", list(plots.keys()))
            return plots

        except Exception as e:
            logger.error(f"❌ Error generando gráficos de monitoreo: {str(e)}")
            return plots

# Función utilitaria para crear instancia global
def create_monitor(db_path: str = "monitoring.db") -> ModelMonitor:
    """Crear instancia del monitor"""
    return ModelMonitor(db_path)