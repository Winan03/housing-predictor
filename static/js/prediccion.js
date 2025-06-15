// ============================================================================
// HOUSING PREDICTOR - PREDICCION.JS (CORREGIDO)
// Sistema de predicción de precios de vivienda
// ============================================================================

// Variables globales
let currentModel = null;
let predictionHistory = [];

// ============================================================================
// INICIALIZACIÓN
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('🏠 Housing Predictor cargado');
    initializeApp();
});

function initializeApp() {
    try {
        // Inicializar secciones colapsables
        initializeSections();
        
        // Configurar el selector de modelo
        initializeModelSelector();
        
        // Validar formulario en tiempo real
        initializeFormValidation();
        
        // Configurar eventos
        initializeEventListeners();
        
        console.log('✅ Aplicación inicializada correctamente');
    } catch (error) {
        console.error('❌ Error inicializando aplicación:', error);
        showError('Error al inicializar la aplicación');
    }
}

// ============================================================================
// INICIALIZACIÓN DE COMPONENTES
// ============================================================================

function initializeSections() {
    const sections = document.querySelectorAll('.form-section');
    
    sections.forEach(section => {
        const header = section.querySelector('.section-header');
        const content = section.querySelector('.section-content');
        const chevron = header.querySelector('.fas.fa-chevron-down');
        
        if (!header || !content) return;
        
        header.addEventListener('click', () => {
            const isActive = content.classList.contains('active');
            
            // Cerrar todas las secciones
            sections.forEach(s => {
                const sContent = s.querySelector('.section-content');
                const sChevron = s.querySelector('.fas.fa-chevron-down');
                if (sContent) sContent.classList.remove('active');
                if (sChevron) sChevron.style.transform = 'rotate(0deg)';
            });
            
            // Abrir la sección clickeada si no estaba activa
            if (!isActive) {
                content.classList.add('active');
                if (chevron) chevron.style.transform = 'rotate(180deg)';
            }
        });
    });
    
    // Activar la primera sección por defecto
    const firstSection = document.querySelector('.form-section .section-content');
    if (firstSection) {
        firstSection.classList.add('active');
        const firstChevron = document.querySelector('.form-section .fas.fa-chevron-down');
        if (firstChevron) {
            firstChevron.style.transform = 'rotate(180deg)';
        }
    }
    
    console.log('✅ Secciones colapsables inicializadas');
}

function initializeModelSelector() {
    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect) {
        currentModel = modelSelect.value;
        modelSelect.addEventListener('change', (e) => {
            currentModel = e.target.value;
            console.log(`📊 Modelo seleccionado: ${currentModel}`);
        });
    }
}

function initializeFormValidation() {
    const inputs = document.querySelectorAll('input[type="number"]');
    
    inputs.forEach(input => {
        input.addEventListener('input', validateInput);
        input.addEventListener('blur', validateInput);
    });
    
    console.log('✅ Validación de formulario inicializada');
}

function initializeEventListeners() {
    // Prevenir envío del formulario con Enter
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', (e) => {
            e.preventDefault();
            predictPrice();
        });
    }
    
    // Atajos de teclado
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            predictPrice();
        } else if (e.ctrlKey && e.key === 'r') {
            e.preventDefault();
            clearForm();
        }
    });
    
    console.log('✅ Event listeners configurados');
}

// ============================================================================
// VALIDACIÓN DE FORMULARIO
// ============================================================================

function validateInput(event) {
    const input = event.target;
    const value = parseFloat(input.value);
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    
    // Remover clases de validación previas
    input.classList.remove('valid', 'invalid');
    
    if (isNaN(value)) {
        input.classList.add('invalid');
        return false;
    }
    
    if ((min !== null && value < min) || (max !== null && value > max)) {
        input.classList.add('invalid');
        showInputError(input, `Valor debe estar entre ${min} y ${max}`);
        return false;
    }
    
    input.classList.add('valid');
    hideInputError(input);
    return true;
}

function showInputError(input, message) {
    // Remover error previo
    hideInputError(input);
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'input-error';
    errorDiv.textContent = message;
    errorDiv.style.cssText = `
        color: #e53e3e;
        font-size: 12px;
        margin-top: 5px;
        padding: 5px;
        background: #fed7d7;
        border-radius: 4px;
    `;
    
    input.parentNode.appendChild(errorDiv);
}

function hideInputError(input) {
    const errorDiv = input.parentNode.querySelector('.input-error');
    if (errorDiv) {
        errorDiv.remove();
    }
}

function validateForm() {
    const requiredInputs = document.querySelectorAll('input[required], select[required]');
    let isValid = true;
    const errors = [];
    
    requiredInputs.forEach(input => {
        if (!input.value.trim()) {
            isValid = false;
            errors.push(`${input.labels[0]?.textContent || input.name} es requerido`);
            input.classList.add('invalid');
        } else if (input.type === 'number') {
            const value = parseFloat(input.value);
            const min = parseFloat(input.min);
            const max = parseFloat(input.max);
            
            if (isNaN(value)) {
                isValid = false;
                errors.push(`${input.labels[0]?.textContent || input.name} debe ser un número válido`);
                input.classList.add('invalid');
            } else if ((min !== null && value < min) || (max !== null && value > max)) {
                isValid = false;
                errors.push(`${input.labels[0]?.textContent || input.name} debe estar entre ${min} y ${max}`);
                input.classList.add('invalid');
            }
        }
    });
    
    if (!isValid) {
        showError(`Errores de validación:\n${errors.join('\n')}`);
    }
    
    return isValid;
}

// ============================================================================
// PREDICCIÓN DE PRECIOS
// ============================================================================

async function predictPrice() {
    try {
        console.log('🔄 Iniciando predicción...');
        
        // Validar formulario
        if (!validateForm()) {
            return;
        }
        
        // Obtener datos del formulario
        const formData = getFormData();
        if (!formData) {
            showError('Error al obtener datos del formulario');
            return;
        }
        
        // Mostrar loading
        showLoading(true);
        hideError();
        hideResult();
        
        // Hacer predicción
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                // IMPORTANTE: formData.features ahora es un objeto, no un array
                features: formData.features, 
                model: formData.model
            })
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || `Error ${response.status}: ${response.statusText}`);
        }
        
        if (!result.success) {
            throw new Error(result.error || 'Error en la predicción');
        }
        
        // Mostrar resultado
        displayPredictionResult(result);
        
        // Guardar en historial
        savePredictionToHistory(formData, result);
        
        console.log('✅ Predicción completada exitosamente');
        
    } catch (error) {
        console.error('❌ Error en predicción:', error);
        showError(`Error al realizar la predicción: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

function getFormData() {
    try {
        const form = document.getElementById('predictionForm');
        const modelSelect = document.getElementById('modelSelect');
        
        if (!form || !modelSelect) {
            throw new Error('Elementos del formulario no encontrados');
        }
        
        // Lista completa de características (ahora usaremos esto para las claves del objeto)
        // Asegúrate de que esta lista de featureNames coincida exactamente con las características
        // que tu modelo espera en el orden correcto en el backend, incluyendo 'neighborhood_cluster'.
        // Si 'neighborhood_cluster' es una característica que se genera en el backend a partir de otras,
        // entonces no debería estar aquí o su valor debe ser '0' como placeholder si no está en el formulario.
        const featureNames = [
            'crim', 'zn', 'indus', 'chas', 'nox', 'rm', 
            'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat',
            'neighborhood_cluster' 
        ];
        
        const features = {}; // <--- CAMBIO CLAVE: Inicializar como un OBJETO vacío
        
        for (const featureName of featureNames) {
            const input = form.querySelector(`[name="${featureName}"]`);
            
            if (!input) {
                // Si no existe el campo en el HTML, usar valor por defecto
                const defaultValue = getDefaultValueForFeature(featureName);
                features[featureName] = defaultValue; // Asignar al OBJETO usando el nombre como clave
                console.warn(`⚠️ Campo ${featureName} no encontrado en el formulario, usando valor por defecto: ${defaultValue}`);
                continue;
            }
            
            const value = parseFloat(input.value);
            if (isNaN(value)) {
                // Si el valor no es un número válido, puedes decidir cómo manejarlo.
                // Aquí lanzamos un error como lo hacías.
                throw new Error(`Valor inválido para ${featureName}: ${input.value}`);
            }
            
            features[featureName] = value; // Asignar al OBJETO usando el nombre como clave
        }
        
        console.log('📊 Características obtenidas:', {
            total: Object.keys(features).length, // Contar claves del objeto
            features: features // El objeto de características tal como se enviará
        });
        
        return {
            features: features, // Ahora features es un objeto (diccionario)
            model: modelSelect.value,
            timestamp: new Date().toISOString()
        };
        
    } catch (error) {
        console.error('Error obteniendo datos del formulario:', error);
        showError(`Error en datos del formulario: ${error.message}`);
        return null;
    }
}

function getDefaultValueForFeature(featureName) {
    // Valores por defecto para características que pueden faltar
    const defaults = {
        'crim': 0.00632,
        'zn': 18.0,
        'indus': 2.31,
        'chas': 0,
        'nox': 0.538,
        'rm': 6.575,
        'age': 65.2,
        'dis': 4.09,
        'rad': 1,
        'tax': 296,
        'ptratio': 15.3,
        'b': 396.9,
        'lstat': 4.98,
        'neighborhood_cluster': 0 // Valor por defecto para cluster de vecindario
    };
    
    return defaults[featureName] || 0;
}

function displayPredictionResult(result) {
    try {
        // Actualizar elementos del resultado
        const priceValue = document.getElementById('priceValue');
        const confidenceRange = document.getElementById('confidenceRange');
        const modelUsed = document.getElementById('modelUsed');
        const timestamp = document.getElementById('timestamp');
        const resultDiv = document.getElementById('predictionResult');
        
        if (!priceValue || !confidenceRange || !modelUsed || !timestamp || !resultDiv) {
            throw new Error('Elementos de resultado no encontrados');
        }
        
        // Formatear y mostrar precio
        const price = result.prediction;
        const formattedPrice = result.formatted_price || `$${(price * 1000).toLocaleString('es-ES')}`;
        priceValue.textContent = formattedPrice;
        
        // Mostrar intervalo de confianza
        if (result.confidence_interval) {
            const lower = result.confidence_interval.lower;
            const upper = result.confidence_interval.upper;
            confidenceRange.textContent = `${lower} - ${upper}`;
        } else {
            const lower = (price * 0.9 * 1000).toLocaleString('es-ES');
            const upper = (price * 1.1 * 1000).toLocaleString('es-ES');
            confidenceRange.textContent = `$${lower} - $${upper}`;
        }
        
        // Mostrar modelo usado
        modelUsed.textContent = result.model_used || currentModel || 'N/A';
        
        // Mostrar timestamp
        const date = new Date(result.timestamp || Date.now());
        timestamp.textContent = date.toLocaleString('es-ES');
        
        // Mostrar resultado con animación
        resultDiv.classList.add('show');
        
        // Scroll suave al resultado
        setTimeout(() => {
            resultDiv.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'center' 
            });
        }, 300);
        
        console.log('✅ Resultado mostrado correctamente');
        
    } catch (error) {
        console.error('Error mostrando resultado:', error);
        showError(`Error al mostrar el resultado: ${error.message}`);
    }
}

function savePredictionToHistory(formData, result) {
    try {
        const prediction = {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            features: formData.features,
            model: formData.model,
            prediction: result.prediction,
            formatted_price: result.formatted_price,
            confidence_interval: result.confidence_interval
        };
        
        predictionHistory.unshift(prediction);
        
        // Mantener solo las últimas 10 predicciones
        if (predictionHistory.length > 10) {
            predictionHistory = predictionHistory.slice(0, 10);
        }
        
        console.log(`📚 Predicción guardada en historial (${predictionHistory.length} total)`);
        
    } catch (error) {
        console.error('Error guardando en historial:', error);
    }
}

// ============================================================================
// MÉTRICAS Y GRÁFICOS
// ============================================================================

async function showMetrics() {
    try {
        console.log('📊 Obteniendo métricas de modelos...');
        showLoading(true, 'Cargando métricas...');
        
        const response = await fetch('/api/metrics');
        
        if (!response.ok) {
            throw new Error(`Error ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Error obteniendo métricas');
        }
        
        displayMetrics(result.metrics);
        console.log('✅ Métricas cargadas exitosamente');
        
    } catch (error) {
        console.error('❌ Error obteniendo métricas:', error);
        showError(`Error al cargar métricas: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

async function showCharts() {
    try {
        console.log('📈 Generando gráficos de comparación...');
        showLoading(true, 'Generando gráficos...');
        
        const response = await fetch('/api/plots');
        
        if (!response.ok) {
            throw new Error(`Error ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (!result.success) {
            throw new Error(result.error || 'Error generando gráficos');
        }
        
        displayCharts(result.plot);
        console.log('✅ Gráficos generados exitosamente');
        
    } catch (error) {
        console.error('❌ Error generando gráficos:', error);
        showError(`Error al generar gráficos: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

function displayMetrics(metrics) {
    // Crear ventana modal para mostrar métricas
    const modal = createModal('Métricas de Rendimiento de Modelos', createMetricsTable(metrics));
    document.body.appendChild(modal);
    
    // Mostrar modal
    setTimeout(() => modal.classList.add('show'), 10);
}

function displayCharts(plotBase64) {
    // Crear ventana modal para mostrar gráficos
    const chartImg = document.createElement('img');
    chartImg.src = `data:image/png;base64,${plotBase64}`;
    chartImg.style.cssText = 'max-width: 100%; height: auto; border-radius: 8px;';
    
    const modal = createModal('Comparación de Modelos', chartImg);
    document.body.appendChild(modal);
    
    // Mostrar modal
    setTimeout(() => modal.classList.add('show'), 10);
}

function createMetricsTable(metrics) {
    const table = document.createElement('table');
    table.style.cssText = `
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        font-size: 14px;
    `;
    
    // Encabezados
    const headers = ['Modelo', 'R² Score', 'RMSE', 'MAE'];
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        th.style.cssText = `
            padding: 12px;
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            text-align: left;
            font-weight: bold;
        `;
        headerRow.appendChild(th);
    });
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Cuerpo de la tabla
    const tbody = document.createElement('tbody');
    
    Object.entries(metrics).forEach(([modelName, modelMetrics]) => {
        const row = document.createElement('tr');
        
        const cells = [
            modelName,
            modelMetrics.r2_test?.toFixed(4) || 'N/A',
            modelMetrics.rmse?.toFixed(4) || 'N/A',
            modelMetrics.mae?.toFixed(4) || 'N/A'
        ];
        
        cells.forEach(cellText => {
            const td = document.createElement('td');
            td.textContent = cellText;
            td.style.cssText = `
                padding: 10px 12px;
                border: 1px solid #e2e8f0;
                `;
            row.appendChild(td);
        });
        
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    return table;
}

function createModal(title, content) {
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
        opacity: 0;
        transition: opacity 0.3s ease;
    `;
    
    const modalContent = document.createElement('div');
    modalContent.style.cssText = `
        background: white;
        border-radius: 15px;
        padding: 25px;
        max-width: 90%;
        max-height: 90%;
        overflow-y: auto;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        transform: scale(0.9);
        transition: transform 0.3s ease;
        position: relative;
    `;
    
    const titleElement = document.createElement('h3');
    titleElement.textContent = title;
    titleElement.style.cssText = `
        margin: 0 0 20px 0;
        color: #2d3748;
        font-size: 24px;
        text-align: center;
    `;
    
    const closeButton = document.createElement('button');
    closeButton.innerHTML = '&times;';
    closeButton.style.cssText = `
        position: absolute;
        top: 15px;
        right: 20px;
        background: none;
        border: none;
        font-size: 30px;
        cursor: pointer;
        color: #999;
        line-height: 1;
    `;
    
    closeButton.addEventListener('click', () => {
        modal.style.opacity = '0';
        modalContent.style.transform = 'scale(0.9)';
        setTimeout(() => modal.remove(), 300);
    });
    
    // Cerrar con Escape
    const closeOnEscape = (e) => {
        if (e.key === 'Escape') {
            modal.style.opacity = '0';
            modalContent.style.transform = 'scale(0.9)';
            setTimeout(() => modal.remove(), 300);
            document.removeEventListener('keydown', closeOnEscape);
        }
    };
    document.addEventListener('keydown', closeOnEscape);
    
    // Cerrar al hacer clic fuera
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.style.opacity = '0';
            modalContent.style.transform = 'scale(0.9)';
            setTimeout(() => modal.remove(), 300);
        }
    });
    
    modalContent.appendChild(titleElement);
    modalContent.appendChild(closeButton);
    modalContent.appendChild(content);
    modal.appendChild(modalContent);
    
    // Implementar función show personalizada
    modal.classList.add = function(className) {
        if (className === 'show') {
            this.style.opacity = '1';
            modalContent.style.transform = 'scale(1)';
        }
    };
    
    return modal;
}

// ============================================================================
// UTILIDADES
// ============================================================================

function clearForm() {
    try {
        const form = document.getElementById('predictionForm');
        if (!form) return;
        
        // Resetear inputs a valores por defecto
        const defaultValues = {
            'crim': '0.00632',
            'zn': '18.0',
            'indus': '2.31',
            'chas': '0',
            'nox': '0.538',
            'rm': '6.575',
            'age': '65.2',
            'dis': '4.09',
            'rad': '1',
            'tax': '296',
            'ptratio': '15.3',
            'b': '396.9',
            'lstat': '4.98',
            'neighborhood_cluster': '0' // Valor por defecto para el cluster
        };
        
        Object.entries(defaultValues).forEach(([name, value]) => {
            const input = form.querySelector(`[name="${name}"]`);
            if (input) {
                input.value = value;
                input.classList.remove('valid', 'invalid');
                hideInputError(input);
            }
        });
        
        // Ocultar resultado
        hideResult();
        hideError();
        
        console.log('🧹 Formulario limpiado');
        
    } catch (error) {
        console.error('Error limpiando formulario:', error);
        showError('Error al limpiar el formulario');
    }
}

function showLoading(show, message = 'Procesando predicción...') {
    const loadingDiv = document.getElementById('loading');
    if (!loadingDiv) return;
    
    if (show) {
        const loadingText = loadingDiv.querySelector('p');
        if (loadingText) loadingText.textContent = message;
        loadingDiv.classList.add('show');
    } else {
        loadingDiv.classList.remove('show');
    }
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    
    if (errorDiv && errorText) {
        errorText.textContent = message;
        errorDiv.classList.add('show');
        
        // Auto-ocultar después de 15 segundos para errores largos
        setTimeout(() => {
            if (errorDiv.classList.contains('show')) {
                errorDiv.classList.remove('show');
            }
        }, 15000);
    }
    
    console.error('🚨 Error mostrado:', message);
}

function hideError() {
    const errorDiv = document.getElementById('errorMessage');
    if (errorDiv) {
        errorDiv.classList.remove('show');
    }
}

function hideResult() {
    const resultDiv = document.getElementById('predictionResult');
    if (resultDiv) {
        resultDiv.classList.remove('show');
    }
}

// ============================================================================
// FUNCIONES DE EXPORTACIÓN
// ============================================================================

function exportPredictionHistory() {
    try {
        if (predictionHistory.length === 0) {
            showError('No hay predicciones en el historial para exportar');
            return;
        }
        
        const dataStr = JSON.stringify(predictionHistory, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `predicciones_${new Date().toISOString().split('T')[0]}.json`;
        link.click();
        
        console.log('📁 Historial exportado exitosamente');
        
    } catch (error) {
        console.error('Error exportando historial:', error);
        showError('Error al exportar el historial');
    }
}

// ============================================================================
// EVENTOS GLOBALES
// ============================================================================

// Manejar errores globales
window.addEventListener('error', (event) => {
    console.error('❌ Error global:', event.error);
    showError('Se produjo un error inesperado en la aplicación');
});

// Manejar errores de promesas no capturadas
window.addEventListener('unhandledrejection', (event) => {
    console.error('❌ Promesa rechazada no manejada:', event.reason);
    showError('Error de comunicación con el servidor');
    event.preventDefault();
});

// Indicador de conexión
window.addEventListener('online', () => {
    console.log('🌐 Conexión restaurada');
    hideError();
});

window.addEventListener('offline', () => {
    console.log('🌐 Conexión perdida');
    showError('Sin conexión a internet. Algunas funciones pueden no estar disponibles.');
});

// ============================================================================
// CONSOLA DE DEBUG (solo en desarrollo)
// ============================================================================

if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    window.HousingPredictor = {
        predictor: {
            history: predictionHistory,
            clearHistory: () => { predictionHistory = []; },
            exportHistory: exportPredictionHistory,
            currentModel: () => currentModel,
            getFormData: getFormData,
            predictPrice: predictPrice
        },
        utils: {
            showMetrics,
            showCharts,
            clearForm,
            validateForm,
            getDefaultValueForFeature
        },
        debug: {
            testPrediction: () => {
                console.log('🧪 Ejecutando predicción de prueba...');
                const testData = getFormData();
                console.log('Datos de prueba:', testData);
                return testData;
            }
        }
    };
    
    console.log('🔧 Modo debug activado. Usa window.HousingPredictor para acceder a funciones de debug.');
}