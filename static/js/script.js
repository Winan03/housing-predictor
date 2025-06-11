// Variables globales
let currentPrediction = null;
let currentMetrics = null;
let currentPlots = null;

// Configuración de la API
const API_BASE_URL = window.location.origin;

// Inicialización cuando el DOM está listo
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    loadDefaultValues();
});

// Configurar todos los event listeners
function initializeEventListeners() {
    // Botón de predicción
    document.getElementById('predictBtn').addEventListener('click', makePrediction);
    
    // Botón de métricas
    document.getElementById('showMetricsBtn').addEventListener('click', showModelMetrics);
    
    // Botón de gráficos
    document.getElementById('showPlotsBtn').addEventListener('click', showPlots);
    
    // Botón de limpiar formulario
    document.getElementById('clearFormBtn').addEventListener('click', clearForm);
    
    // Botón de exportar (cuando esté visible)
    document.getElementById('exportBtn').addEventListener('click', exportMetrics);
    
    // Validación en tiempo real de campos numéricos
    setupInputValidation();
}

// Cargar valores por defecto del dataset Boston Housing
function loadDefaultValues() {
    const defaultValues = {
        'crim': 0.00632,
        'zn': 18.0,
        'indus': 2.31,
        'chas': 0,
        'nox': 0.538,
        'rm': 6.575,
        'age': 65.2,
        'dis': 4.0900,
        'rad': 1,
        'tax': 296,
        'ptratio': 15.3,
        'b': 396.90,
        'lstat': 4.98
    };
    
    // Solo cargar si los campos están vacíos
    Object.keys(defaultValues).forEach(key => {
        const input = document.getElementById(key);
        if (input && !input.value) {
            input.value = defaultValues[key];
        }
    });
}

// Configurar validación de inputs
function setupInputValidation() {
    const inputs = document.querySelectorAll('#predictionForm input[type="number"]');
    inputs.forEach(input => {
        input.addEventListener('input', function() {
            validateInput(this);
        });
        
        input.addEventListener('blur', function() {
            validateInput(this);
        });
    });
}

// Validar input individual
function validateInput(input) {
    const value = parseFloat(input.value);
    const isValid = !isNaN(value) && value >= 0;
    
    if (isValid) {
        input.classList.remove('is-invalid');
        input.classList.add('is-valid');
    } else {
        input.classList.remove('is-valid');
        input.classList.add('is-invalid');
    }
}

// Recopilar datos del formulario
function getFormData() {
    const formData = {};
    const inputs = document.querySelectorAll('#predictionForm input, #predictionForm select');
    
    inputs.forEach(input => {
        if (input.type === 'number') {
            formData[input.name] = parseFloat(input.value);
        } else {
            formData[input.name] = input.value;
        }
    });
    
    return formData;
}

// Validar todos los campos del formulario
function validateForm() {
    const requiredFields = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat'];
    let isValid = true;
    const errors = [];
    
    requiredFields.forEach(field => {
        const input = document.getElementById(field);
        if (!input.value || (input.type === 'number' && isNaN(parseFloat(input.value)))) {
            isValid = false;
            errors.push(`El campo ${field} es requerido`);
            input.classList.add('is-invalid');
        }
    });
    
    if (!isValid) {
        showError('Por favor, complete todos los campos requeridos:\n' + errors.join('\n'));
    }
    
    return isValid;
}

// Realizar predicción
async function makePrediction() {
    if (!validateForm()) {
        return;
    }
    
    const formData = getFormData();
    const selectedModel = document.getElementById('modelSelect').value;
    
    showLoadingSpinner(true);
    hideAllResults();
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: selectedModel,
                features: formData
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            currentPrediction = result;
            displayPredictionResult(result);
        } else {
            throw new Error(result.error || 'Error en la predicción');
        }
        
    } catch (error) {
        console.error('Error en la predicción:', error);
        showError('Error al realizar la predicción: ' + error.message);
    } finally {
        showLoadingSpinner(false);
    }
}

// Mostrar resultado de predicción
function displayPredictionResult(result) {
    const resultCard = document.getElementById('predictionResult');
    const priceElement = document.getElementById('predictedPrice');
    const modelElement = document.getElementById('modelUsed');
    
    // Formatear precio (asumiendo que está en miles de dólares)
    const formattedPrice = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(result.prediction * 1000);
    
    priceElement.textContent = formattedPrice;
    modelElement.textContent = `Modelo utilizado: ${result.model}`;
    
    // Mostrar información adicional si está disponible
    if (result.confidence) {
        const confidenceElement = document.createElement('p');
        confidenceElement.className = 'text-muted';
        confidenceElement.textContent = `Confianza: ${(result.confidence * 100).toFixed(1)}%`;
        modelElement.parentNode.appendChild(confidenceElement);
    }
    
    resultCard.style.display = 'block';
    resultCard.classList.add('fade-in-up');
    
    // Scroll suave hacia el resultado
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Mostrar métricas de modelos
async function showModelMetrics() {
    showLoadingSpinner(true);
    hideAllResults();
    
    try {
        const response = await fetch(`${API_BASE_URL}/metrics`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            currentMetrics = result;
            displayMetricsResult(result);
        } else {
            throw new Error(result.error || 'Error al obtener métricas');
        }
        
    } catch (error) {
        console.error('Error al obtener métricas:', error);
        showError('Error al obtener métricas: ' + error.message);
    } finally {
        showLoadingSpinner(false);
    }
}

// Mostrar resultado de métricas
function displayMetricsResult(result) {
    const resultCard = document.getElementById('metricsResult');
    const contentElement = document.getElementById('metricsContent');
    
    let htmlContent = '<div class="table-responsive">';
    htmlContent += '<table class="table table-striped table-hover metrics-table">';
    htmlContent += '<thead class="table-dark">';
    htmlContent += '<tr><th>Modelo</th><th>MAE</th><th>MSE</th><th>RMSE</th><th>R²</th></tr>';
    htmlContent += '</thead><tbody>';
    
    result.metrics.forEach(metric => {
        htmlContent += `<tr>
            <td><strong>${metric.model}</strong></td>
            <td>${metric.mae.toFixed(4)}</td>
            <td>${metric.mse.toFixed(4)}</td>
            <td>${metric.rmse.toFixed(4)}</td>
            <td>${metric.r2.toFixed(4)}</td>
        </tr>`;
    });
    
    htmlContent += '</tbody></table></div>';
    
    // Agregar interpretación de métricas
    htmlContent += '<div class="mt-4">';
    htmlContent += '<h6><i class="fas fa-info-circle me-2"></i>Interpretación de Métricas:</h6>';
    htmlContent += '<ul class="list-unstyled">';
    htmlContent += '<li><strong>MAE (Mean Absolute Error):</strong> Error promedio absoluto en miles de dólares</li>';
    htmlContent += '<li><strong>MSE (Mean Squared Error):</strong> Error cuadrático medio</li>';
    htmlContent += '<li><strong>RMSE (Root Mean Squared Error):</strong> Raíz del error cuadrático medio</li>';
    htmlContent += '<li><strong>R² (R-squared):</strong> Coeficiente de determinación (0-1, mayor es mejor)</li>';
    htmlContent += '</ul></div>';
    
    contentElement.innerHTML = htmlContent;
    resultCard.style.display = 'block';
    resultCard.classList.add('fade-in-up');
    
    // Mostrar botón de exportar
    document.getElementById('exportBtn').style.display = 'block';
    
    // Scroll suave hacia el resultado
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Mostrar gráficos de comparación
async function showPlots() {
    showLoadingSpinner(true);
    hideAllResults();
    
    try {
        const response = await fetch(`${API_BASE_URL}/plots`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            currentPlots = result;
            displayPlotsResult(result);
        } else {
            throw new Error(result.error || 'Error al obtener gráficos');
        }
        
    } catch (error) {
        console.error('Error al obtener gráficos:', error);
        showError('Error al obtener gráficos: ' + error.message);
    } finally {
        showLoadingSpinner(false);
    }
}

// Mostrar resultado de gráficos
function displayPlotsResult(result) {
    const resultCard = document.getElementById('plotsResult');
    const contentElement = document.getElementById('plotsContent');
    
    let htmlContent = '<div class="row">';
    
    result.plots.forEach((plot, index) => {
        htmlContent += `<div class="col-md-6 mb-4">
            <div class="plot-container">
                <h6>${plot.title}</h6>
                <img src="data:image/png;base64,${plot.image}" alt="${plot.title}" class="img-fluid">
            </div>
        </div>`;
    });
    
    htmlContent += '</div>';
    
    contentElement.innerHTML = htmlContent;
    resultCard.style.display = 'block';
    resultCard.classList.add('fade-in-up');
    
    // Scroll suave hacia el resultado
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Exportar métricas a CSV
function exportMetrics() {
    if (!currentMetrics) {
        showError('No hay métricas para exportar');
        return;
    }
    
    let csvContent = "Modelo,MAE,MSE,RMSE,R²\n";
    
    currentMetrics.metrics.forEach(metric => {
        csvContent += `${metric.model},${metric.mae},${metric.mse},${metric.rmse},${metric.r2}\n`;
    });
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('hidden', '');
    a.setAttribute('href', url);
    a.setAttribute('download', 'housing_metrics.csv');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Limpiar formulario
function clearForm() {
    const inputs = document.querySelectorAll('#predictionForm input, #predictionForm select');
    inputs.forEach(input => {
        if (input.type === 'number') {
            input.value = '';
        } else if (input.type === 'select-one') {
            input.selectedIndex = 0;
        }
        input.classList.remove('is-valid', 'is-invalid');
    });
    
    hideAllResults();
    loadDefaultValues();
}

// Mostrar/ocultar spinner de carga
function showLoadingSpinner(show) {
    const spinner = document.getElementById('loadingSpinner');
    if (show) {
        spinner.style.display = 'block';
    } else {
        spinner.style.display = 'none';
    }
}

// Ocultar todos los resultados
function hideAllResults() {
    document.getElementById('welcomeCard').style.display = 'none';
    document.getElementById('predictionResult').style.display = 'none';
    document.getElementById('metricsResult').style.display = 'none';
    document.getElementById('plotsResult').style.display = 'none';
    document.getElementById('exportBtn').style.display = 'none';
}

// Mostrar mensaje de error
function showError(message) {
    // Crear toast de error
    const toast = document.createElement('div');
    toast.className = 'toast align-items-center text-white bg-danger border-0 position-fixed';
    toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999;';
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas fa-exclamation-circle me-2"></i>
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    document.body.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    // Remover el toast después de que se oculte
    toast.addEventListener('hidden.bs.toast', function() {
        document.body.removeChild(toast);
    });
}

// Mostrar mensaje de éxito
function showSuccess(message) {
    const toast = document.createElement('div');
    toast.className = 'toast align-items-center text-white bg-success border-0 position-fixed';
    toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999;';
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                <i class="fas fa-check-circle me-2"></i>
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    document.body.appendChild(toast);
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    toast.addEventListener('hidden.bs.toast', function() {
        document.body.removeChild(toast);
    });
}

// Funciones auxiliares para llamadas desde HTML
window.makePrediction = makePrediction;
window.showModelMetrics = showModelMetrics;
window.showPlots = showPlots;
window.exportMetrics = exportMetrics;

// Función para mostrar métricas con manejo de errores mejorado
async function showModelMetrics() {
    showLoadingSpinner(true);
    hideAllResults();
    
    try {
        const response = await fetch(`${API_BASE_URL}/model_performance`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            let errorMessage;
            try {
                const errorData = await response.json();
                errorMessage = errorData.error || `Error HTTP ${response.status}: ${response.statusText}`;
            } catch {
                errorMessage = `Error HTTP ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        
        if (result.success && result.metrics) {
            currentMetrics = result;
            displayMetricsResult(result);
            showSuccess('Métricas cargadas correctamente');
        } else {
            throw new Error(result.error || 'No se pudieron obtener las métricas del modelo');
        }
        
    } catch (error) {
        console.error('Error al obtener métricas:', error);
        handleNetworkError(error);
    } finally {
        showLoadingSpinner(false);
    }
}

// Función para mostrar gráficos con validación mejorada
async function showPlots() {
    showLoadingSpinner(true);
    hideAllResults();
    
    try {
        const response = await fetch(`${API_BASE_URL}/plots`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            let errorMessage;
            try {
                const errorData = await response.json();
                errorMessage = errorData.error || `Error HTTP ${response.status}: ${response.statusText}`;
            } catch {
                errorMessage = `Error HTTP ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        
        // CORRECCIÓN: El backend devuelve 'plot' no 'plots'
        if (result.success && result.plot) {
            // Convertir a formato que espera displayPlotsResult
            const plotsResult = {
                success: true,
                plots: [result.plot] // Convertir el plot único en array
            };
            currentPlots = plotsResult;
            displayPlotsResult(plotsResult);
            showSuccess('Gráficos cargados correctamente');
        } else {
            throw new Error(result.error || 'No se pudieron obtener los gráficos del modelo');
        }
        
    } catch (error) {
        console.error('Error al obtener gráficos:', error);
        handleNetworkError(error);
    } finally {
        showLoadingSpinner(false);
    }
}

// Función mejorada para limpiar formulario con confirmación
function clearForm() {
    // Mostrar confirmación antes de limpiar
    if (confirm('¿Está seguro de que desea limpiar todos los campos del formulario?')) {
        const inputs = document.querySelectorAll('#predictionForm input, #predictionForm select');
        
        inputs.forEach(input => {
            if (input.type === 'number') {
                input.value = '';
            } else if (input.tagName === 'SELECT') {
                input.selectedIndex = 0;
            }
            // Remover clases de validación
            input.classList.remove('is-valid', 'is-invalid');
        });
        
        // Resetear variables globales
        currentPrediction = null;
        currentMetrics = null;
        currentPlots = null;
        
        // Ocultar todos los resultados y mostrar mensaje de bienvenida
        hideAllResults();
        const welcomeCard = document.getElementById('welcomeCard');
        if (welcomeCard) {
            welcomeCard.style.display = 'block';
        }
        
        // Recargar valores por defecto
        setTimeout(() => {
            if (typeof loadDefaultValues === 'function') {
                loadDefaultValues();
            }
            showSuccess('Formulario limpiado correctamente');
        }, 100);
        
        // Scroll hacia arriba
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

// Función para alternar visibilidad de resultados
function toggleResultsVisibility(targetId, hideOthers = true) {
    if (hideOthers) {
        hideAllResults();
    }
    
    const targetElement = document.getElementById(targetId);
    if (targetElement) {
        const isVisible = targetElement.style.display !== 'none';
        targetElement.style.display = isVisible ? 'none' : 'block';
        
        if (!isVisible) {
            targetElement.classList.add('fade-in-up');
            targetElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }
}

// Función para refrescar métricas (re-obtener del servidor)
async function refreshMetrics() {
    if (confirm('¿Desea actualizar las métricas desde el servidor?')) {
        currentMetrics = null;
        await showModelMetrics();
    }
}

// Función para refrescar gráficos
async function refreshPlots() {
    if (confirm('¿Desea actualizar los gráficos desde el servidor?')) {
        currentPlots = null;
        await showPlots();
    }
}

// Función para validar antes de mostrar métricas
function validateAndShowMetrics() {
    // Verificar si hay modelos disponibles
    const modelSelect = document.getElementById('modelSelect');
    if (!modelSelect || !modelSelect.value) {
        showError('Por favor seleccione un modelo antes de ver las métricas');
        return;
    }
    
    showModelMetrics();
}

// Función para validar antes de mostrar gráficos
function validateAndShowPlots() {
    // Verificar si hay modelos disponibles
    const modelSelect = document.getElementById('modelSelect');
    if (!modelSelect || !modelSelect.value) {
        showError('Por favor seleccione un modelo antes de ver los gráficos');
        return;
    }
    
    showPlots();
}

// Función para resetear completamente la aplicación
function resetApplication() {
    if (confirm('¿Está seguro de que desea resetear completamente la aplicación? Se perderán todos los datos cargados.')) {
        // Limpiar variables globales
        currentPrediction = null;
        currentMetrics = null;
        currentPlots = null;
        
        // Limpiar formulario
        clearForm();
        
        // Resetear selector de modelo
        const modelSelect = document.getElementById('modelSelect');
        if (modelSelect) {
            modelSelect.selectedIndex = 0;
        }
        
        // Ocultar spinner si está visible
        showLoadingSpinner(false);
        
        // Mostrar mensaje de bienvenida
        const welcomeCard = document.getElementById('welcomeCard');
        if (welcomeCard) {
            welcomeCard.style.display = 'block';
        }
        
        showSuccess('Aplicación reseteada correctamente');
    }
}

// Función mejorada para manejar errores de red específicamente
function handleNetworkError(error) {
    let errorMessage = 'Error desconocido';
    
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
        errorMessage = 'Error de conexión: No se puede conectar al servidor. Verifique:\n• Que el servidor Flask esté ejecutándose\n• Su conexión a internet\n• La URL del API';
    } else if (error.message.includes('404')) {
        errorMessage = 'Error 404: El endpoint solicitado no existe en el servidor. Verifique la implementación del backend.';
    } else if (error.message.includes('500')) {
        errorMessage = 'Error 500: Error interno del servidor. Revise los logs del servidor para más detalles.';
    } else if (error.message.includes('503')) {
        errorMessage = 'Error 503: Servicio no disponible. Los modelos pueden no estar cargados correctamente.';
    } else if (error.message.includes('400')) {
        errorMessage = `Error 400: Solicitud incorrecta - ${error.message}`;
    } else {
        errorMessage = `Error: ${error.message}`;
    }
    
    showError(errorMessage);
}

// Función mejorada para verificar el estado del servidor
// Función corregida para mostrar métricas
function displayMetricsResult(result) {
    try {
        const metricsContainer = document.getElementById('metricsResult');
        if (!metricsContainer) {
            console.error('Elemento metricsResult no encontrado');
            return;
        }

        // Verificar estructura de datos
        console.log('Estructura de métricas recibida:', result);
        
        if (!result.metrics) {
            throw new Error('No se encontraron métricas en la respuesta');
        }

        let metricsHTML = `
            <div class="card">
                <div class="card-header bg-info text-white">
                    <h5 class="mb-0">📊 Métricas de Rendimiento del Modelo</h5>
                </div>
                <div class="card-body">
        `;

        // Manejar diferentes estructuras de métricas
        if (typeof result.metrics === 'object' && !Array.isArray(result.metrics)) {
            // Si metrics es un objeto, convertirlo a formato manejable
            Object.keys(result.metrics).forEach(modelName => {
                const modelMetrics = result.metrics[modelName];
                metricsHTML += `
                    <div class="metric-section mb-4">
                        <h6 class="text-primary">${modelName}</h6>
                        <div class="row">
                `;
                
                if (typeof modelMetrics === 'object') {
                    Object.keys(modelMetrics).forEach(metricName => {
                        const metricValue = modelMetrics[metricName];
                        const formattedValue = typeof metricValue === 'number' ? 
                            metricValue.toFixed(4) : metricValue;
                        
                        metricsHTML += `
                            <div class="col-md-6 col-lg-3 mb-2">
                                <div class="bg-light p-2 rounded">
                                    <small class="text-muted">${metricName}</small>
                                    <div class="font-weight-bold">${formattedValue}</div>
                                </div>
                            </div>
                        `;
                    });
                }
                
                metricsHTML += `
                        </div>
                    </div>
                `;
            });
        } else if (Array.isArray(result.metrics)) {
            // Si metrics es un array
            result.metrics.forEach(metric => {
                metricsHTML += `
                    <div class="metric-item mb-3">
                        <div class="row">
                            <div class="col-md-4">
                                <strong>${metric.name || 'Métrica'}</strong>
                            </div>
                            <div class="col-md-8">
                                ${metric.value || 'N/A'}
                            </div>
                        </div>
                    </div>
                `;
            });
        } else {
            // Fallback para estructuras inesperadas
            metricsHTML += `
                <div class="alert alert-warning">
                    <h6>Datos de métricas:</h6>
                    <pre>${JSON.stringify(result.metrics, null, 2)}</pre>
                </div>
            `;
        }

        metricsHTML += `
                </div>
                <div class="card-footer">
                    <small class="text-muted">
                        Última actualización: ${new Date().toLocaleString()}
                    </small>
                </div>
            </div>
        `;

        metricsContainer.innerHTML = metricsHTML;
        metricsContainer.style.display = 'block';
        
        // Scroll suave al resultado
        metricsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
    } catch (error) {
        console.error('Error al mostrar métricas:', error);
        showError(`Error al mostrar métricas: ${error.message}`);
    }
}

// Función corregida para mostrar gráficos
function displayPlotsResult(result) {
    try {
        const plotsContainer = document.getElementById('plotsResult');
        if (!plotsContainer) {
            console.error('Elemento plotsResult no encontrado');
            return;
        }

        // Verificar estructura de datos
        console.log('Estructura de plots recibida:', result);
        
        if (!result.plots || (!Array.isArray(result.plots) && !result.plot)) {
            throw new Error('No se encontraron gráficos en la respuesta');
        }

        let plotsHTML = `
            <div class="card">
                <div class="card-header bg-success text-white">
                    <h5 class="mb-0">📈 Gráficos del Modelo</h5>
                </div>
                <div class="card-body">
        `;

        // Manejar diferentes estructuras de plots
        let plotsArray = [];
        
        if (result.plots && Array.isArray(result.plots)) {
            plotsArray = result.plots;
        } else if (result.plot) {
            plotsArray = [result.plot];
        }

        if (plotsArray.length === 0) {
            throw new Error('No hay gráficos disponibles para mostrar');
        }

        plotsArray.forEach((plot, index) => {
            if (plot) {
                plotsHTML += `
                    <div class="plot-container mb-4">
                        <h6>Gráfico ${index + 1}</h6>
                        <div class="text-center">
                            <img src="data:image/png;base64,${plot}" 
                                 class="img-fluid rounded shadow" 
                                 alt="Gráfico del modelo ${index + 1}"
                                 style="max-width: 100%; height: auto;">
                        </div>
                    </div>
                `;
            }
        });

        plotsHTML += `
                </div>
                <div class="card-footer">
                    <small class="text-muted">
                        ${plotsArray.length} gráfico(s) generado(s) - ${new Date().toLocaleString()}
                    </small>
                </div>
            </div>
        `;

        plotsContainer.innerHTML = plotsHTML;
        plotsContainer.style.display = 'block';
        
        // Scroll suave al resultado
        plotsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
    } catch (error) {
        console.error('Error al mostrar gráficos:', error);
        showError(`Error al mostrar gráficos: ${error.message}`);
    }
}

// Función auxiliar para validar la respuesta del servidor
function validateServerResponse(response, expectedFields) {
    if (!response.success) {
        throw new Error(response.error || 'El servidor reportó un error');
    }
    
    for (const field of expectedFields) {
        if (!(field in response)) {
            throw new Error(`Campo requerido '${field}' no encontrado en la respuesta`);
        }
    }
    
    return true;
}

// Función mejorada para mostrar métricas con mejor manejo de errores
async function showModelMetrics() {
    showLoadingSpinner(true);
    hideAllResults();
    
    try {
        console.log('Solicitando métricas del modelo...');
        
        const response = await fetch(`${API_BASE_URL}/model_performance`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            let errorMessage;
            try {
                const errorData = await response.json();
                errorMessage = errorData.error || `Error HTTP ${response.status}: ${response.statusText}`;
            } catch {
                errorMessage = `Error HTTP ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        console.log('Respuesta de métricas:', result);
        
        // Validar respuesta
        validateServerResponse(result, ['metrics']);
        
        currentMetrics = result;
        displayMetricsResult(result);
        showSuccess('Métricas cargadas correctamente');
        
    } catch (error) {
        console.error('Error detallado al obtener métricas:', error);
        showError(`Error al cargar métricas: ${error.message}`);
    } finally {
        showLoadingSpinner(false);
    }
}

// Función mejorada para mostrar gráficos con mejor manejo de errores
async function showPlots() {
    showLoadingSpinner(true);
    hideAllResults();
    
    try {
        console.log('Solicitando gráficos del modelo...');
        
        const response = await fetch(`${API_BASE_URL}/plots`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        if (!response.ok) {
            let errorMessage;
            try {
                const errorData = await response.json();
                errorMessage = errorData.error || `Error HTTP ${response.status}: ${response.statusText}`;
            } catch {
                errorMessage = `Error HTTP ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorMessage);
        }
        
        const result = await response.json();
        console.log('Respuesta de plots:', result);
        
        // Validar respuesta (aceptar tanto 'plots' como 'plot')
        if (!result.success) {
            throw new Error(result.error || 'El servidor reportó un error');
        }
        
        if (!result.plots && !result.plot) {
            throw new Error('No se encontraron gráficos en la respuesta del servidor');
        }
        
        // Normalizar la respuesta para compatibilidad
        if (result.plot && !result.plots) {
            result.plots = [result.plot];
        }
        
        currentPlots = result;
        displayPlotsResult(result);
        showSuccess('Gráficos cargados correctamente');
        
    } catch (error) {
        console.error('Error detallado al obtener gráficos:', error);
        showError(`Error al cargar gráficos: ${error.message}`);
    } finally {
        showLoadingSpinner(false);
    }
}