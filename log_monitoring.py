import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import time

# --- Configuración de logging para log_monitoring.py ---
# Esto asegura que cualquier error dentro de log_monitoring.py también sea registrado.
# Es una buena práctica tener loggers separados para diferentes módulos si es necesario,
# pero por simplicidad, usaremos la configuración básica para los propios logs del script.
logging.basicConfig(filename='log_monitoring_errors.log', level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración para los logs de la aplicación (app_logs.log) ---
# Este es el archivo de logs que será monitoreado para enviar alertas por correo.
# Es crucial que tu aplicación Flask escriba en este archivo.
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

    if not all([sender_email, password, receiver_email]):
        logger.error("Las variables de entorno del correo (EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER) no están configuradas.")
        return

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Usando SMTP_SSL para una conexión segura en el puerto 465
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            text = msg.as_string()
            server.sendmail(sender_email, receiver_email, text)
            print(f"Correo '{subject}' enviado exitosamente a {receiver_email}!")
            logger.info(f"Correo '{subject}' enviado exitosamente.")
    except Exception as e:
        logger.error(f"Error al enviar el correo '{subject}': {str(e)}", exc_info=True)

def check_logs_and_send_email():
    """
    Lee el archivo de logs de la aplicación (app_logs.log) y, si hay nuevas entradas de log
    desde la última verificación, envía un correo electrónico con el nuevo contenido.
    También borra el archivo de logs después de enviarlo para evitar notificaciones duplicadas.
    """
    try:
        if not os.path.exists(APP_LOG_FILE):
            logger.warning(f"Archivo de logs de la aplicación '{APP_LOG_FILE}' no encontrado. Se omite la verificación de logs.")
            return

        with open(APP_LOG_FILE, 'r+') as file: # Usar 'r+' para leer y luego truncar
            logs = file.read()
            # Verificar si hay contenido para enviar
            if logs.strip(): # Verificar si logs no está vacío o solo contiene espacios en blanco
                subject = f"Alerta de Monitoreo de Logs - {time.strftime('%Y-%m-%d %H:%M:%S')}"
                send_email(subject, f"Nueva actividad en los logs de la aplicación:\n\n{logs}")
                # Borrar el archivo de logs después de enviar el correo para evitar reenviar los mismos logs
                file.seek(0)  # Ir al principio del archivo
                file.truncate() # Borrar su contenido
            else:
                logger.info(f"No hay nuevas entradas de log en '{APP_LOG_FILE}' para enviar.")
    except Exception as e:
        logger.error(f"Error durante el procesamiento del archivo de logs: {str(e)}", exc_info=True)

if __name__ == "__main__":
    # Este bloque se ejecutará cuando `python log_monitoring.py` se ejecute directamente,
    # por ejemplo, por un proceso worker de Render.
    logger.info("Worker de monitoreo de logs iniciado.")
    # Puedes añadir un bucle aquí para verificar los logs periódicamente si este es un worker dedicado
    # que debe ejecutarse continuamente y enviar correos periódicamente.
    # Por ahora, solo verificará una vez y saldrá, según tu Procfile.
    # Si deseas un monitoreo continuo por este worker, deberías añadir un bucle 'while True':
    # while True:
    #    check_logs_and_send_email()
    #    time.sleep(300) # Verificar cada 5 minutos (300 segundos)

    # Para tu actual Procfile 'worker: python log_monitoring.py',
    # este script se ejecutará una vez y luego terminará.
    # Si deseas un monitoreo continuo, necesitas modificar el Procfile
    # o añadir un bucle aquí y asegurar que el worker se ejecute continuamente.
    
    # Por ahora, lo mantendremos simple y solo lo ejecutaremos una vez cuando sea invocado,
    # asumiendo que el worker de Render lo reinicia o que lo programarás.
    check_logs_and_send_email()
    logger.info("Worker de monitoreo de logs finalizado.")
