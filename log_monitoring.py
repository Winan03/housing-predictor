import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import time
from dotenv import load_dotenv  # Añadir esta importación

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# --- Configuración de logging para log_monitoring.py ---
logging.basicConfig(filename='log_monitoring_errors.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración para los logs de la aplicación (app_logs.log) ---
APP_LOG_FILE = 'app_logs.log'

def send_email(subject, body):
    """
    Envía un correo electrónico con el asunto y cuerpo dados.
    Obtiene el remitente, la contraseña y el correo electrónico del receptor
    de las variables de entorno.
    Registra cualquier error encontrado durante el envío del correo.
    """
    sender_email = os.getenv('EMAIL_SENDER')
    password = os.getenv('EMAIL_PASSWORD')
    receiver_email = os.getenv('EMAIL_RECEIVER')

    # Debug: Verificar que las variables se están cargando
    print(f"Debug - Sender: {sender_email}")
    print(f"Debug - Receiver: {receiver_email}")
    print(f"Debug - Password configured: {'Yes' if password else 'No'}")

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
        print(f"Intentando enviar correo a {receiver_email}...")
        # Usando SMTP_SSL para una conexión segura en el puerto 465
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            text = msg.as_string()
            server.sendmail(sender_email, receiver_email, text)
            print(f"Correo '{subject}' enviado exitosamente a {receiver_email}!")
            logger.info(f"Correo '{subject}' enviado exitosamente.")
            return True
    except smtplib.SMTPAuthenticationError as e:
        error_msg = f"Error de autenticación SMTP: {str(e)}. Verifica tu email y contraseña de aplicación."
        logger.error(error_msg)
        print(f"Error de autenticación: {error_msg}")
        return False
    except Exception as e:
        error_msg = f"Error al enviar el correo '{subject}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"Error general: {error_msg}")
        return False

def check_logs_and_send_email():
    """
    Lee el archivo de logs de la aplicación (app_logs.log) y, si hay nuevas entradas de log
    desde la última verificación, envía un correo electrónico con el nuevo contenido.
    También borra el archivo de logs después de enviarlo para evitar notificaciones duplicadas.
    """
    try:
        print(f"Verificando archivo de logs: {APP_LOG_FILE}")
        
        if not os.path.exists(APP_LOG_FILE):
            warning_msg = f"Archivo de logs de la aplicación '{APP_LOG_FILE}' no encontrado. Se omite la verificación de logs."
            logger.warning(warning_msg)
            print(f"Warning: {warning_msg}")
            return

        # Verificar el tamaño del archivo
        file_size = os.path.getsize(APP_LOG_FILE)
        print(f"Tamaño del archivo de logs: {file_size} bytes")

        with open(APP_LOG_FILE, 'r+', encoding='utf-8') as file:
            logs = file.read()
            print(f"Contenido leído del archivo (primeros 200 chars): {logs[:200]}...")
            
            # Verificar si hay contenido para enviar
            if logs.strip():
                print("Enviando logs por correo...")
                subject = f"Alerta de Monitoreo de Logs - {time.strftime('%Y-%m-%d %H:%M:%S')}"
                success = send_email(subject, f"Nueva actividad en los logs de la aplicación:\n\n{logs}")
                
                if success:
                    # Borrar el archivo de logs después de enviar el correo exitosamente
                    file.seek(0)
                    file.truncate()
                    print("Archivo de logs limpiado después del envío exitoso.")
                else:
                    print("No se limpió el archivo de logs debido a error en el envío.")
            else:
                info_msg = f"No hay nuevas entradas de log en '{APP_LOG_FILE}' para enviar."
                logger.info(info_msg)
                print(info_msg)
                
    except Exception as e:
        error_msg = f"Error durante el procesamiento del archivo de logs: {str(e)}"
        logger.error(error_msg, exc_info=True)
        print(f"Error: {error_msg}")

def test_email_configuration():
    """
    Función para probar la configuración de correo.
    """
    print("=== PROBANDO CONFIGURACIÓN DE CORREO ===")
    subject = "Prueba de Configuración - Log Monitoring"
    body = "Este es un correo de prueba para verificar que la configuración de log monitoring funciona correctamente."
    success = send_email(subject, body)
    return success

if __name__ == "__main__":
    print("=== INICIANDO WORKER DE MONITOREO DE LOGS ===")
    logger.info("Worker de monitoreo de logs iniciado.")
    
    # Primero probar la configuración de correo
    print("\n1. Probando configuración de correo...")
    if test_email_configuration():
        print("✓ Configuración de correo OK")
    else:
        print("✗ Error en configuración de correo")
    
    # Luego verificar logs
    print("\n2. Verificando logs de la aplicación...")
    check_logs_and_send_email()
    
    print("\n=== WORKER DE MONITOREO FINALIZADO ===")
    logger.info("Worker de monitoreo de logs finalizado.")