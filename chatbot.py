import re

class FAQChatbot:
    """
    Clase para el chatbot de Preguntas Frecuentes de Predicción de Precios de Viviendas.
    Contiene un diccionario de intenciones y respuestas para manejar las consultas de los usuarios.
    """

    def __init__(self):
        """
        Inicializa el chatbot con el diccionario de preguntas frecuentes.
        Cada clave del diccionario es una tupla de palabras clave (o patrones de frases)
        que un usuario podría usar para hacer una pregunta, y el valor es la respuesta correspondiente.
        """
        self.faq_responses = {
            # --- Saludos y Despedidas ---
            ("hola", "saludos", "que tal", "buenos dias", "buenas tardes", "buenas noches", "que onda", "hey", "que hay", "holis", "un saludo", "como andas"):
                "¡Hola! ¡Qué alegría verte por aquí! Soy el asistente virtual de Predictor de Viviendas IA, tu compañero inteligente en el fascinante mundo del mercado inmobiliario. Estoy listo para resolver cualquier duda que tengas. ¿En qué puedo echarte una mano hoy?",
            ("adios", "chao", "hasta luego", "nos vemos", "bye", "despedida", "hasta pronto", "me despido", "un gusto", "fue un placer", "gracias por todo"):
                "¡Hasta pronto! Ha sido un verdadero placer charlar contigo y asistirte con tus inquietudes. ¡No dudes en volver cuando lo necesites! ¡Gracias por elegir Predictor de Viviendas IA para tus necesidades inmobiliarias, te esperamos!",
            ("gracias", "muchas gracias", "te lo agradezco", "agradecido", "mil gracias", "muy amable", "agradecimiento", "te doy las gracias", "gracias por la info"):
                "¡De nada! Es un placer inmenso poder ayudarte. Estoy aquí justamente para despejar todas tus dudas y brindarte la mejor información sobre la predicción de precios de viviendas y la increíble tecnología de IA que hemos desarrollado. ¡Por favor, no te quedes con ninguna pregunta, estoy a tu disposición!",
            ("cómo estás", "como estas", "que tal estas", "todo bien", "como te va", "que haces"):
                "¡Excelente! Estoy funcionando a la perfección, totalmente optimizado y con todas mis capacidades al máximo para responder tus preguntas de la manera más eficiente y precisa posible. ¿En qué maravilla puedo serte útil en este momento?",
            ("quien eres", "eres humano", "eres un bot", "que haces tu", "dime quien eres", "quien te creo", "eres real"):
                "¡Qué buena pregunta! Soy un avanzado chatbot de Inteligencia Artificial, diseñado con mucho esfuerzo y dedicación para ofrecerte información instantánea y detallada sobre el Predictor de Viviendas IA. No soy humano, pero mi objetivo es ser tu mejor aliado digital en este tema. ¿Tienes alguna otra curiosidad sobre mí o la plataforma?",

            # --- Información General sobre el Sitio ---
            ("que es", "que hacen", "para que sirve", "ayuda", "informacion", "acerca de", "sobre la plataforma", "descripción", "introduccion", "objetivo", "que ofrece"):
                "¡Claro! El Predictor de Viviendas IA es una plataforma de vanguardia que revoluciona la forma de entender el mercado inmobiliario. Utilizamos Inteligencia Artificial de última generación para predecir precios de viviendas con una asombrosa precisión del 92%. Nuestro gran objetivo es empoderar a compradores, vendedores e inversores, proporcionándoles datos súper confiables y análisis profundos para que puedan tomar decisiones inmobiliarias inteligentes, estratégicas y con total confianza. Es como tener un experto a tu lado las 24 horas del día.",
            ("precio", "cuanto cuesta", "costo", "gratis", "tarifa", "es de pago", "subscripcion", "plan", "es caro", "hay que pagar"):
                "¡Tengo una excelente noticia para ti! ¡Nuestro servicio de predicción de precios de viviendas es 100% gratuito, totalmente sin costo alguno! No hay sorpresas, ni tarifas ocultas, ni planes de suscripción complicados, ni trucos. Puedes usarlo sin límites, las veces que necesites, y obtener resultados inmediatos sin ningún tipo de compromiso. ¡Queremos que la información valiosa sea accesible para todos!",
            ("registro", "cuenta", "crear cuenta", "iniciar sesion", "log in", "logearse", "registrarse", "necesito registrarme", "hay que crear cuenta"):
                "Para garantizar tu privacidad y hacer tu experiencia lo más cómoda y fluida posible, no necesitas registrarte, ni crear ninguna cuenta, ni rellenar formularios. ¡Es así de simple! Nuestro predictor es completamente gratuito, de acceso inmediato para todos los usuarios, y no hay procesos engorrosos de por medio. ¡Llegas, predices y listo!",
            ("como empezar", "comenzar", "iniciar", "donde predigo", "listo para comenzar", "quiero predecir", "primeros pasos", "como funciona la prediccion"):
                "¡Es más fácil de lo que imaginas comenzar a predecir! Para dar tus primeros pasos, simplemente busca el botón grande y llamativo que dice 'Comenzar Predicción' o 'INICIAR PREDICCIÓN AHORA'. Lo hemos colocado de manera prominente en nuestra página principal para que lo encuentres sin dificultad. Haz clic ahí y serás guiado directamente a nuestra intuitiva herramienta de predicción. ¡Verás lo sencillo que es!",
            ("futuro inmobiliario", "bienvenido al futuro", "vision", "que es el futuro inmobiliario"):
                "En Predictor de Viviendas IA, no solo pensamos en el presente, sino que visionamos y construimos el 'Futuro Inmobiliario'. Para nosotros, este futuro es uno donde la Inteligencia Artificial no es solo una moda, sino una herramienta indispensable que te proporciona información invaluable, análisis profundos y perspectivas claras. Queremos que navegues el mercado con una confianza inquebrantable y tomes decisiones que sean verdaderamente informadas, estratégicas y que te impulsen al éxito. ¡Es el futuro al alcance de tu mano!",
            ("beneficios", "ventajas", "por que elegirnos", "que me ofrece", "cual es la ventaja", "porque usarlos"):
                "¡Las ventajas de elegirnos son muchas y muy claras! Te ofrecemos el poder de la IA avanzada con una altísima precisión (¡un impresionante 92%!), resultados instantáneos que obtienes en menos de 2 segundos, y un análisis verdaderamente completo basado en 13 variables críticas de las propiedades. Además, somos una herramienta totalmente gratuita, sin necesidad de registro, y estamos potenciados por tecnología de vanguardia. Es decir, eficiencia, precisión y comodidad, todo en uno. ¿Qué más se puede pedir?",

            # --- Características del Modelo de IA ---
            ("precision", "exactitud", "fiabilidad", "que tan preciso", "porcentaje de precision", "error", "margen de error", "confianza"):
                "¡Absolutamente! Nuestro modelo de IA ha demostrado una precisión realmente impresionante: un 92.05% en la predicción de precios de viviendas. Y para que tengas total confianza, esto se respalda con métricas robustas como un RMSE de 2.60 y un MAE de 1.93. Estos números son sólidos indicadores de la altísima fiabilidad y exactitud de nuestras estimaciones. ¡Hemos entrenado rigurosamente nuestros modelos con miles y miles de propiedades reales para garantizarlo!",
            ("algoritmos", "ia", "inteligencia artificial", "modelo", "como funciona", "que modelos usan", "algoritmos de ml", "tipos de ia", "base del modelo"):
                "¡Es una excelente pregunta! Detrás de cada predicción, utilizamos un sistema de IA súper avanzado que no se limita a un solo enfoque. En realidad, combina y orquesta inteligentemente varios algoritmos de Machine Learning de vanguardia. Esto nos permite asegurar la máxima robustez, adaptabilidad y fiabilidad. Específicamente, empleamos joyas como Random Forest, XGBoost, SVR y un Voting Ensemble, entre otros. Esta combinación sinérgica es la que nos permite obtener predicciones ultra-precisas. ¡Es una verdadera orquesta de algoritmos trabajando para ti!",
            ("variables", "factores", "que considera", "parametros", "analisis", "datos analizados", "caracteristicas", "atributos", "que influye", "como calcula"):
                "Nuestro análisis es meticuloso y verdaderamente exhaustivo, ¡no dejamos ningún detalle al azar! Nos basamos en la evaluación de nada menos que 13 variables críticas de las propiedades. Esto incluye desde la ubicación exacta y la seguridad de la zona, hasta la disponibilidad y cercanía de servicios esenciales (como escuelas, hospitales y transporte público). Y por supuesto, consideramos las características físicas detalladas de la vivienda: su tamaño en metros cuadrados, el número de habitaciones, baños, la antigüedad, y muchos otros elementos que influyen en el valor. ¡Así garantizamos una predicción lo más completa posible!",
            ("modelos ia", "cuantos modelos", "numero de modelos", "modelos utilizados", "cuantos algoritmos"):
                "Nuestro 'Sistema Inteligente' es una verdadera maravilla de la ingeniería de IA. No utiliza uno ni dos, sino que integra y utiliza un total de 10 modelos de IA diferentes. Todos ellos trabajan en perfecta armonía y en conjunto, para garantizar la máxima precisión, una fiabilidad inigualable y una predicción robusta en cada una de tus estimaciones de precio. ¡Es un cerebro colectivo de IA al servicio de tu inversión!",
            ("analisis completo", "analisis 360", "que es un analisis 360"):
                "Cuando hablamos de un 'Análisis Completo' o 'Análisis 360°' en nuestra plataforma, nos referimos a una evaluación exhaustiva y sin precedentes de 13 variables críticas por cada propiedad. Es un enfoque holístico que nos permite ofrecerte una visión completa y profundamente informada del valor potencial de la vivienda. Consideramos absolutamente todos los aspectos relevantes, desde la estructura y el entorno, hasta los detalles más pequeños que puedan influir en el precio. ¡No se nos escapa nada!",
            ("random forest", "xgboost", "svr", "voting ensemble", "bayesian ridge", "extra trees", "gradient boosting", "knn", "lightgbm", "ridge", "hablame de modelos"):
                "¡Ah, me encanta que preguntes sobre los detalles técnicos! Cada uno de estos son algoritmos de Machine Learning de élite que contribuyen a la inteligencia de nuestro sistema. Si hablamos de cuál se destaca por su precisión, el **Support Vector Regression (SVR)** es el campeón, con un impresionante R2 Score de 0.9205. Pero no subestimemos a los demás; otros modelos importantes como Gradient Boosting (con un R2 Score de 0.9190), XGBoost (0.9145), y el Voting Ensemble (0.9128) también rinden a un nivel excepcional. Cada uno aporta su 'grano de arena' para la predicción final. ¡Es un equipo de campeones!",
            ("r2 score", "rmse", "mae", "metricas", "que significan metricas", "que es r2", "que es rmse", "que es mae"):
                "¡Entiendo tu interés en las métricas! El R2 Score (o coeficiente de determinación) es como un termómetro que mide qué tan bien nuestro modelo logra predecir la variabilidad de los precios; un valor de 0.9205 es simplemente excelente, ¡casi perfecto! Por otro lado, el RMSE (Root Mean Squared Error) y el MAE (Mean Absolute Error) son indicadores que nos dicen cuán cerca están nuestras predicciones del valor real. Un RMSE de 2.60 y un MAE de 1.93 son cifras fantásticas que confirman que nuestras predicciones están muy, muy cerca del precio de mercado real. ¡Son la prueba de nuestra precisión!",
            ("mejor modelo", "modelo mas preciso", "cual es el mejor algoritmo", "cual me recomiendas", "que modelo me recomiendas"):
                "¡Qué gran pregunta! Si tengo que destacar uno, según todas nuestras métricas rigurosas y resultados de pruebas, el mejor modelo individual para la predicción de precios de viviendas es sin duda el **Support Vector Regression (SVR)**. Este modelo alcanza un asombroso **R2 Score de 0.9205**, con un RMSE de solo 2.60 y un MAE de 1.93. Por estas razones, es nuestro modelo de referencia y la base de su excepcional precisión. Sin embargo, nuestro sistema final utiliza una combinación de varios para una robustez aún mayor, pero SVR es el líder en rendimiento. ¡Es el que te da la mayor confianza!",
            ("comparacion modelos", "modelos y precision", "rendimiento de modelos", "que tan buenos son sus modelos"):
                "¡Con gusto te doy un desglose de la precisión de algunos de nuestros modelos principales! Esto te dará una idea clara de la calidad de nuestras herramientas:\n"
                "- **Support Vector Regression (SVR):** R2 Score 0.9205, RMSE 2.60, MAE 1.93 (¡Nuestro campeón, el más preciso!)\n"
                "- **Gradient Boosting:** R2 Score 0.9190, RMSE 2.62, MAE 1.98\n"
                "- **XGBoost:** R2 Score 0.9145, RMSE 2.69, MAE 2.00\n"
                "- **Voting Ensemble:** R2 Score 0.9128, RMSE 2.72, MAE 1.95 (Este combina la fuerza de varios modelos)\n"
                "- **LightGBM:** R2 Score 0.9041, RMSE 2.85, MAE 2.06\n"
                "- **Extra Trees:** R2 Score 0.9011, RMSE 2.89, MAE 2.01\n"
                "Como puedes ver, la gran mayoría de nuestros modelos operan con una altísima precisión, lo que garantiza la fiabilidad de nuestros resultados. SVR es el más destacado, pero todos contribuyen a un sistema robusto. ¡La calidad es nuestra prioridad!",


            # --- Rendimiento y Tecnología ---
            ("tiempo de respuesta", "rapidez", "segundos", "cuanto tarda la prediccion", "tiempo real", "ultra rapido", "velocidad", "que tan rapido es"):
                "¡Aquí la velocidad es clave! Nuestro sistema está diseñado para ser increíblemente rápido. Prepárate para obtener tus predicciones precisas en un abrir y cerrar de ojos, ¡literalmente en menos de 2 segundos! Esto te ofrece una experiencia de usuario 'Ultra Rápida' y en 'Tiempo Real', perfecta para tomar decisiones ágiles y no perder ni un minuto en el dinámico mercado inmobiliario. ¡La información que necesitas, al instante!",
            ("tecnologia", "backend", "con que esta hecho", "plataforma", "stack tecnologico", "herramientas", "infraestructura", "tecnologia usada", "que usan"):
                "Detrás de la magia de nuestras predicciones, se esconde una robusta y sofisticada tecnología de vanguardia. El 'backend', que es el cerebro de toda nuestra operación, está sólidamente construido con el potente lenguaje de programación Python, utilizando el ágil y eficiente framework Flask. Esta combinación nos garantiza no solo una escalabilidad impresionante, sino también una eficiencia y un rendimiento que te sorprenderán. ¡Estamos construidos para el futuro!",
            ("python", "flask", "scikit-learn", "xgboost", "pandas", "numpy", "para que usan python", "librerias", "frameworks"):
                "¡Excelente pregunta sobre nuestros cimientos tecnológicos! Sí, utilizamos Python como el lenguaje principal para todo nuestro desarrollo, desde la lógica del negocio hasta los intrincados algoritmos de IA. Flask es el framework web ligero y potente que da vida a nuestro backend, gestionando las solicitudes y respuestas. Y para el corazón de nuestro Machine Learning, nos apoyamos en librerías de clase mundial: Scikit-learn (una navaja suiza para una amplia gama de algoritmos), XGBoost (perfecto para modelos de gradient boosting avanzados), y Pandas junto con NumPy (esenciales para el manejo, procesamiento y análisis eficiente de grandes volúmenes de datos, vitales para el entrenamiento de nuestros modelos). ¡Cada herramienta cumple un rol crucial en la precisión que te ofrecemos!",

            # --- Estadísticas Clave ---
            ("propiedades analizadas", "cuantas propiedades", "base de datos", "datos de entrenamiento", "tamaño base de datos", "datos usados"):
                "Hemos nutrido y entrenado a nuestros modelos de IA con una base de datos realmente extensísima, ¡que abarca más de 10,000 propiedades reales! Cada una de estas propiedades aporta valiosa información que permite a nuestra IA aprender patrones complejos y refinar sus predicciones. Este vasto conjunto de datos es absolutamente crucial para la solidez, la confiabilidad y la altísima precisión de nuestras predicciones. ¡Cuantos más datos, más inteligente se vuelve nuestro sistema!",
            ("predicciones realizadas", "cuantas predicciones hemos hecho", "uso", "cuanta gente lo usa", "historial de predicciones"):
                "¡Nos enorgullece mucho poder decirte que hasta la fecha, hemos realizado más de 1500 predicciones de precios para nuestros valiosos usuarios! Y este número sigue creciendo cada día, lo cual es una clara señal de la confianza que la gente deposita en nosotros y el inmenso valor que nuestra tecnología de IA aporta a la comunidad inmobiliaria. ¡Estamos felices de ayudar a tantas personas a tomar decisiones informadas!",
            ("resultados comprobados", "numeros que hablan", "estadisticas clave", "pruebas de rendimiento", "confirma tus resultados"):
                "¡Nuestros resultados no solo son buenos, son tangibles y completamente comprobados! Aquí te dejo las estadísticas clave que hablan por sí solas sobre nuestra efectividad y fiabilidad:\n"
                "- Una precisión del 92% en nuestro modelo (con SVR como el mejor)\n"
                "- Un tiempo de respuesta increíblemente rápido de menos de 2 segundos\n"
                "- Más de 10,000 propiedades analizadas y utilizadas para el entrenamiento de nuestros modelos\n"
                "- Más de 1500 predicciones ya realizadas y exitosas para nuestros usuarios\n"
                "¡Estos números demuestran nuestro compromiso con la excelencia y la confianza que puedes depositar en Predictor de Viviendas IA!",

            # --- Consultas Específicas / Soporte ---
            ("contacto", "soporte", "ayuda personal", "hablar con alguien", "dudas", "quien es jefer", "donde puedo contactarlos", "hay atencion al cliente"):
                "Soy un chatbot de preguntas frecuentes, diseñado para darte respuestas rápidas e inmediatas a tus dudas más comunes. Sin embargo, si necesitas asistencia personalizada, tienes consultas muy específicas que no puedo resolver o quieres comunicarte directamente con nuestro equipo de soporte, te sugiero encarecidamente que busques nuestra sección de 'Contacto' o 'Soporte' en la página principal de la plataforma. Allí encontrarás todas las formas de comunicarte con nosotros. ¡Y sí, aprovecho para contarte que 'Jefer' es el talentoso y dedicado desarrollador detrás de esta innovadora y útil plataforma! ¡Un verdadero genio!",
            ("problema", "error", "algo no funciona", "fallo", "bug", "no me predice", "hay un error", "la pagina falla"):
                "Lamento muchísimo escuchar que estás experimentando un problema. Entiendo lo frustrante que puede ser. Para que pueda ayudarte de la manera más efectiva y resolver esto rápidamente, por favor, descríbeme con el mayor detalle posible qué está sucediendo, qué acción realizaste y si ves algún mensaje de error en pantalla. Si es algo relacionado con la predicción, por favor, asegúrate de haber revisado cuidadosamente los datos que ingresaste, ya que a veces un pequeño detalle puede generar un inconveniente. ¡Estoy aquí para ayudarte a solucionarlo!",
            ("seguridad", "mis datos", "es seguro", "privacidad", "manejo de datos", "confidencialidad", "proteccion de datos", "mis datos estan seguros"):
                "¡Absolutamente! Nos tomamos la seguridad y la privacidad de tus datos con la máxima seriedad, es una prioridad fundamental para nosotros. La plataforma está diseñada para ser 100% gratuita y, lo más importante, **no requiere ningún tipo de registro ni ingreso de datos personales sensibles**. Esto implica que NO almacenamos ninguna información tuya, ni personal, ni de tus búsquedas. Tus predicciones son inmediatas, completamente privadas y se procesan al instante, sin dejar rastro de información. ¡Tu tranquilidad es nuestra garantía!",
            ("que es el pulse animation", "animacion pulse", "efecto pulse", "que significa pulse"):
                "¡Excelente observación! El 'pulse-animation' es un detalle visual sutil pero muy efectivo que aplicamos a ciertos elementos clave de nuestra interfaz, como el botón 'Comenzar Predicción'. Su función es crear una suave y rítmica pulsación que lo hace más atractivo, captando tu atención de manera agradable y resaltando la invitación a interactuar con él. Es un toque de diseño que mejora la experiencia visual y la usabilidad.",
            ("gradient-text", "que es gradient text", "texto degradado", "efecto de texto"):
                "El 'gradient-text' es un estilo visual de tipografía que le da un toque moderno y dinámico a nuestros encabezados. Utiliza una transición fluida y armoniosa entre dos o más colores, creando un efecto degradado directamente en el texto. Esto no solo lo hace estéticamente agradable, sino que también le otorga un aspecto vibrante, contemporáneo y muy distintivo. ¡Es una manera de hacer que el texto cobre vida!",
            ("glass-card", "que es glass card", "efecto vidrio", "tarjeta de vidrio"):
                "Una 'glass-card' es un componente de diseño de interfaz de usuario que, como su nombre lo indica, simula un sofisticado efecto de vidrio esmerilado o traslúcido. ¿Cómo se logra? Principalmente con un desenfoque sutil (blur) aplicado al fondo y una leve transparencia en el elemento. Esto no solo le da un aspecto elegante y limpio, sino que también añade una sensación de profundidad y un efecto tridimensional a los paneles o tarjetas informativas. ¡Es un toque moderno y minimalista!",
            ("requisitos", "necesito algo", "compatible", "que necesito para usarlo", "software"):
                "¡La buena noticia es que casi no necesitas nada! Nuestra plataforma es una aplicación web. Esto significa que funciona directamente desde tu navegador de internet favorito (como Chrome, Firefox, Safari, Edge, etc.) sin necesidad de instalar ningún software adicional. Lo único que realmente necesitas es una conexión a internet estable para acceder al predictor. ¡Así de simple!",
            ("personalizar", "mis datos", "guardar predicciones", "historial", "se guardan mis predicciones"):
                "Actualmente, nuestro sistema está diseñado pensando en la simplicidad y el uso inmediato. Para mantener la privacidad y la rapidez, no ofrecemos la opción de guardar predicciones personalizadas ni historiales de tus consultas. Cada predicción es una consulta en tiempo real, procesada al instante y sin almacenar la información de forma persistente. ¡Es un servicio directo y al punto!",
            ("novedades", "actualizaciones", "mejoras", "que hay de nuevo", "hay planes de futuro"):
                "¡Estamos en constante evolución! Nos dedicamos continuamente a trabajar en mejorar la plataforma y a perfeccionar nuestros modelos de IA. Nuestro objetivo es siempre ofrecerte la experiencia más precisa, útil y fluida posible en la predicción de precios de viviendas. ¡Siempre estamos buscando formas de innovar y añadir más valor para ti!",
            ("diferencia con otros", "que los hace unicos", "competencia"):
                "Lo que nos hace verdaderamente únicos en el mercado es nuestra combinación de una **precisión excepcional (92.05%)**, la **velocidad ultra-rápida** de nuestras predicciones (menos de 2 segundos), y el hecho de que ofrecemos este **servicio de vanguardia de forma 100% gratuita y sin necesidad de registro**. Además, la integración de 10 algoritmos de IA diferentes en un 'Voting Ensemble' asegura una robustez que pocos pueden igualar. No solo te damos un número, te damos una herramienta poderosa para tomar decisiones inteligentes con confianza.",
            ("como funciona la inteligencia artificial", "que es la ia", "ia en bienes raices"):
                "La Inteligencia Artificial, en nuestro caso, se refiere a programas de computadora que aprenden y mejoran con la experiencia, similar a como lo hacemos los humanos. En el contexto inmobiliario, le 'alimentamos' a nuestros modelos de IA con una enorme cantidad de datos de propiedades (ubicación, tamaño, número de habitaciones, etc.) y sus precios reales. La IA analiza estos datos, encuentra patrones complejos y relaciones que son difíciles de detectar para un ojo humano, y luego usa esos patrones para predecir el precio de una propiedad nueva. Es como tener un experto matemático y de bienes raíces superdotado trabajando 24/7 para ti. En esencia, aprende de lo que ya ha ocurrido en el mercado para predecir lo que probablemente valdrá tu propiedad.",
            ("casos de uso", "para quien es util", "quien puede usarlo"):
                "¡Nuestro Predictor de Viviendas IA es increíblemente útil para una amplia gama de personas! Es ideal para: **compradores** que quieren saber si están pagando un precio justo; **vendedores** que buscan fijar un precio competitivo y atractivo; **inversionistas** que necesitan evaluar rápidamente el potencial de retorno de una propiedad; **agentes inmobiliarios** que quieren ofrecer a sus clientes una herramienta de valoración adicional y basada en IA; y **cualquier persona** interesada en el valor de una propiedad, ya sea por curiosidad o para planificar su futuro. ¡Es una herramienta versátil para el mercado inmobiliario!",
            ("entrenamiento del modelo", "como se entrena la ia", "datos de entrenamiento"):
                "El entrenamiento de nuestros modelos de IA es un proceso crucial y muy riguroso. Consiste en 'enseñarles' a nuestros algoritmos a reconocer patrones y relaciones en los precios de las propiedades. Lo hacemos presentándoles una enorme cantidad de datos históricos de propiedades, cada uno con sus características (número de habitaciones, ubicación, etc.) y su precio de venta real. El modelo ajusta sus 'conocimientos' internamente una y otra vez, minimizando los errores en sus predicciones hasta que alcanza una alta precisión. Es un ciclo de aprendizaje continuo, alimentado por miles de propiedades reales, para que cada predicción sea lo más cercana posible a la realidad del mercado.",
        }

        # Una respuesta por defecto si no se encuentra ninguna coincidencia clara
        self.default_response = (
            "¡Uhm, disculpa! Parece que no logré comprender tu pregunta con la información que tengo a mano en este momento. "
            "Para poder ayudarte mejor, ¿podrías intentar reformularla con otras palabras o ser un poco más específico? "
            "No te preocupes, estoy aquí para asistirte. Generalmente, puedo responder preguntas sobre la **precisión de nuestro modelo** (por ejemplo, '¿cuál es la precisión?' o '¿cuál es el mejor modelo?'), "
            "la **tecnología que utilizamos** (como '¿qué algoritmos usan?' o '¿con qué está hecho?'), "
            "las **características de la predicción** (por ejemplo, '¿qué variables considera?'), "
            "o **información general sobre la plataforma** (como '¿es gratis?' o '¿cómo puedo empezar?'). "
            "¡Anímate a intentarlo de nuevo, estoy seguro de que encontraremos la respuesta juntos!"
        )

    def _preprocess_text(self, text):
        """
        Preprocesa el texto de entrada del usuario:
        - Convierte a minúsculas.
        - Elimina signos de puntuación y acentos.
        - Elimina espacios extra.
        """
        text = text.lower()
        # Eliminar acentos y diéresis
        text = text.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
        text = text.replace('ü', 'u')
        # Eliminar signos de puntuación (mantener solo letras, números y espacios)
        text = re.sub(r'[^\w\s]', '', text)
        # Eliminar espacios extra y dejar solo uno entre palabras
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_response(self, user_input):
        """
        Obtiene una respuesta para la entrada del usuario.
        Busca coincidencias con las palabras clave definidas en faq_responses.
        Prioriza coincidencias más específicas o que contengan más palabras clave.
        """
        processed_input = self._preprocess_text(user_input)

        best_match_response = self.default_response
        max_matches = 0
        
        # Prioridad 1: Coincidencia exacta de frase
        for keywords, response in self.faq_responses.items():
            for kw in keywords:
                if self._preprocess_text(kw) == processed_input:
                    return response

        # Prioridad 2: Mejor coincidencia basada en el número de palabras clave coincidentes
        for keywords, response in self.faq_responses.items():
            current_matches = 0
            processed_keywords_for_pattern = [self._preprocess_text(kw) for kw in keywords]
            
            for kw_in_pattern in processed_keywords_for_pattern:
                if kw_in_pattern in processed_input:
                    current_matches += 1
            
            if current_matches > max_matches:
                max_matches = current_matches
                best_match_response = response
            elif current_matches == max_matches and current_matches > 0:
                # Compara la longitud de la cadena combinada de palabras clave para preferir patrones más largos
                current_pattern_length = sum(len(kw) for kw in processed_keywords_for_pattern)
                best_match_pattern_length = sum(len(self._preprocess_text(k)) for k in self._get_keywords_for_response(best_match_response))
                if current_pattern_length > best_match_pattern_length:
                    best_match_response = response
        
        # Si la entrada del usuario es de una sola palabra y no se encontró una coincidencia por frase exacta,
        # intentamos una coincidencia de palabra clave única.
        if max_matches == 0 and len(processed_input.split()) == 1:
            for keywords, response in self.faq_responses.items():
                if processed_input in [self._preprocess_text(k) for k in keywords]:
                    return response

        return best_match_response

    def _get_keywords_for_response(self, response_text):
        """
        Función auxiliar para obtener las palabras clave asociadas a una respuesta.
        Usada para comparar la longitud de los patrones de clave en caso de empate de coincidencias.
        """
        for keywords, response in self.faq_responses.items():
            if response == response_text:
                return keywords
        return () # Retorna una tupla vacía si no se encuentra (no debería ocurrir con un uso normal)


# Ejemplo de uso (esto simula la interacción del backend)
if __name__ == "__main__":
    chatbot = FAQChatbot()

    print("Bienvenido al Chatbot de Predictor de Viviendas IA. ¡Estoy aquí para ayudarte!")
    print("Puedes preguntar sobre la precisión, tecnología, cómo funciona, o cualquier otra cosa.")
    print("Escribe 'salir' para terminar la conversación.")

    while True:
        user_message = input("Tú: ")
        if user_message.lower() == "salir":
            print("Chatbot: ¡Hasta luego! Ha sido un placer charlar contigo. ¡Que tengas un excelente día!")
            break
        
        response = chatbot.get_response(user_message)
        print(f"Chatbot: {response}")