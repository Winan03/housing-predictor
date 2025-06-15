import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import time
from dotenv import load_dotenv
import requests
from datetime import datetime, timedelta
import json

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# --- Configuración de logging para log_monitoring.py ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración para los logs de la aplicación ---
APP_LOG_FILE = 'app_logs.log'
LAST_CHECK_FILE = 'last_check.txt'

def send_email(subject, body):
    """
    Envía un correo electrónico con el asunto y cuerpo dados.
    """
    sender_email = os.getenv('EMAIL_SENDER')
    password = os.getenv('EMAIL_PASSWORD')
    receiver_email = os.getenv('EMAIL_RECEIVER')

    print(f"Debug - Configuración de correo:")
    print(f"  Sender: {sender_email}")
    print(f"  Receiver: {receiver_email}")
    print(f"  Password configured: {'Yes' if password else 'No'}")

    if not all([sender_email, password, receiver_email]):
        error_msg = f"Variables de entorno faltantes - Sender: {bool(sender_email)}, Password: {bool(password)}, Receiver: {bool(receiver_email)}"
        logger.error(error_msg)
        print(f"Error: {error_msg}")
        return False

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        print(f"Intentando enviar correo...")
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            text = msg.as_string()
            server.sendmail(sender_email, receiver_email, text)
            print(f"✓ Correo enviado exitosamente!")
            logger.info(f"Correo '{subject}' enviado exitosamente.")
            return True
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"Error de autenticación SMTP: {str(e)}"
        logger.error(error_msg)
        print(f"✗ Error de autenticación: {error_msg}")
        return False
    except Exception as e:
        error_msg = f"Error al enviar el correo: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"✗ Error general: {error_msg}")
        return False

def get_last_check_time():
    """
    Obtiene la última vez que se verificaron los logs.
    """
    try:
        if os.path.exists(LAST_CHECK_FILE):
            with open(LAST_CHECK_FILE, 'r') as f:
                timestamp = f.read().strip()
                return datetime.fromisoformat(timestamp)
        else:
            # Si no existe el archivo, usar una hora atrás como referencia
            return datetime.now() - timedelta(hours=1)
    except Exception as e:
        logger.error(f"Error al leer último check: {e}")
        return datetime.now() - timedelta(hours=1)

def save_last_check_time():
    """
    Guarda el tiempo actual como última verificación.
    """
    try:
        with open(LAST_CHECK_FILE, 'w') as f:
            f.write(datetime.now().isoformat())
    except Exception as e:
        logger.error(f"Error al guardar último check: {e}")

def generate_log_entry(message, level="INFO"):
    """
    Genera una entrada de log que será enviada por correo.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return f"[{timestamp}] {level}: {message}"

def check_logs_and_send_email():
    """
    Revisa si hay logs nuevos y los envía por correo.
    Para Render, esto verificará el archivo de logs si existe,
    o enviará un resumen de la actividad de la aplicación.
    """
    try:
        print("=== VERIFICANDO LOGS ===")
        
        logs_to_send = []
        current_time = datetime.now()
        
        # Intentar leer el archivo de logs local primero
        if os.path.exists(APP_LOG_FILE):
            print(f"Archivo de logs encontrado: {APP_LOG_FILE}")
            with open(APP_LOG_FILE, 'r+', encoding='utf-8') as file:
                logs = file.read()
                if logs.strip():
                    logs_to_send.append(logs)
                    # Limpiar el archivo después de leer
                    file.seek(0)
                    file.truncate()
        else:
            print(f"Archivo de logs no encontrado: {APP_LOG_FILE}")
            # Si no hay archivo de logs, crear un log de estado
            logs_to_send.append(generate_log_entry("Sistema de monitoreo activo - Verificando logs de Render"))
        
        # Agregar información del sistema
        logs_to_send.append(generate_log_entry(f"Check de logs ejecutado en: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"))
        logs_to_send.append(generate_log_entry(f"Directorio de trabajo: {os.getcwd()}"))
        logs_to_send.append(generate_log_entry(f"Archivos en directorio: {', '.join(os.listdir('.'))}"))
        
        # Verificar variables de entorno (sin mostrar valores sensibles)
        env_vars = ['EMAIL_SENDER', 'EMAIL_RECEIVER', 'EMAIL_PASSWORD']
        for var in env_vars:
            value = os.getenv(var)
            status = "✓ Configurada" if value else "✗ Faltante"
            logs_to_send.append(generate_log_entry(f"Variable {var}: {status}"))
        
        # Si hay logs para enviar
        if logs_to_send:
            print("Preparando correo con logs...")
            subject = f"Reporte de Logs - {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
            body = "=== REPORTE DE MONITOREO DE LOGS ===\n\n"
            body += "\n".join(logs_to_send)
            body += f"\n\n=== FIN DEL REPORTE ===\n"
            body += f"Generado el: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
            
            success = send_email(subject, body)
            if success:
                save_last_check_time()
                print("✓ Logs enviados exitosamente")
            else:
                print("✗ Error al enviar logs")
        else:
            print("No hay logs para enviar")
            
    except Exception as e:
        error_msg = f"Error durante el procesamiento de logs: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"✗ Error: {error_msg}")

def test_email_configuration():
    """
    Función para probar la configuración de correo.
    """
    print("=== PROBANDO CONFIGURACIÓN DE CORREO ===")
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    subject = f"Prueba de Configuración - Log Monitoring [{current_time}]"
    body = f"""
=== PRUEBA DE CONFIGURACIÓN DE CORREO ===

Este es un correo de prueba para verificar que el sistema de monitoreo de logs funciona correctamente.

Información del sistema:
- Timestamp: {current_time}
- Directorio: {os.getcwd()}
- Variables de entorno configuradas:
  * EMAIL_SENDER: {'✓' if os.getenv('EMAIL_SENDER') else '✗'}
  * EMAIL_PASSWORD: {'✓' if os.getenv('EMAIL_PASSWORD') else '✗'}
  * EMAIL_RECEIVER: {'✓' if os.getenv('EMAIL_RECEIVER') else '✗'}

Si recibes este correo, ¡la configuración está funcionando correctamente!

=== FIN DE LA PRUEBA ===
    """
    success = send_email(subject, body)
    return success

if __name__ == "__main__":
    print("=== INICIANDO SISTEMA DE MONITOREO DE LOGS ===")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Directorio de trabajo: {os.getcwd()}")
    
    # Mostrar variables de entorno (sin valores sensibles)
    print("\nVariables de entorno:")
    for var in ['EMAIL_SENDER', 'EMAIL_RECEIVER', 'EMAIL_PASSWORD']:
        value = os.getenv(var)
        print(f"  {var}: {'✓ Configurada' if value else '✗ Faltante'}")
    
    # Probar configuración de correo
    print("\n1. Probando configuración de correo...")
    if test_email_configuration():
        print("✓ Configuración de correo OK")
    else:
        print("✗ Error en configuración de correo")
        # Aún así, intentar verificar logs
    
    # Verificar y enviar logs
    print("\n2. Verificando logs de la aplicación...")
    check_logs_and_send_email()
    
    print("\n=== MONITOREO DE LOGS COMPLETADO ===")