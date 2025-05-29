// Variables globales
let currentModelMetrics = null;

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    // Event listeners
    document.getElementById('predictBtn').addEventListener('click', makePrediction);
    document.getElementById('showMetricsBtn').addEventListener('click', showModelMetrics);
    document.getElementById('showPlotsBtn').addEventListener('click', showPlots);
    
    // Llenar formulario con valores por defecto
    fillDefaultValues();
});

// Valores por defecto para el formulario
function fillDefaultValues() {
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
        'black': 396.90,
        'lstat': 4.98
    };
    
    Object.keys(defaultValues).forEach(key => {
        const element = document.getElementById(key);
        if (element) {
            element.value = defaultValues[key];
        }
    });
}

// Función para hacer predicción
async function makePrediction() {
    try {
        showLoading(true);
        hideAllResults();
        
        // Obtener datos del formulario
        const formData = getFormData();
        const selectedModel = document.getElementById('modelSelect').value;
        
        // Validar datos
        if (!validateFormData(formData)) {
            showAlert('Por favor, complete todos los campos requeridos.', 'warning');
            return;
        }
        
        // Hacer petición
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                features: formData,
                model: selectedModel
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayPredictionResult(data);
        } else {
            showAlert('Error en la predicción: ' + data.error, 'danger');
        }
        
    } catch (error) {
        console.error('Error:', error);
        showAlert('Error de conexión. Por favor, intente nuevamente.', 'danger');
    } finally {
        showLoading(false);
    }
}

// Función para mostrar métricas de modelos
async function showModelMetrics() {
    try {
        showLoading(true);
        hideAllResults();
        
        const response = await fetch('/model_performance');
        const data = await response.json();
        
        if (data.success) {
            currentModelMetrics = data.metrics;
            displayMetrics(data.metrics);
        } else {
            showAlert('Error al obtener métricas: ' + data.error, 'danger');
        }
        
    } catch (error) {
        console.error('Error:', error);
        showAlert('Error de conexión. Por favor, intente nuevamente.', 'danger');
    } finally {
        showLoading(false);
    }
}

// Función para mostrar gráficos
async function showPlots() {
    try {
        showLoading(true);
        hideAllResults();
        
        const response = await fetch('/plots');
        const data = await response.json();
        
        if (data.success) {
            displayPlots(data.plot);
        } else {
            showAlert('Error al generar gráficos: ' + data.error, 'danger');
        }
        
    } catch (error) {
        console.error('Error:', error);
        showAlert('Error de conexión. Por favor, intente nuevamente.', 'danger');
    } finally {
        showLoading(false);
    }
}

// Obtener datos del formulario
function getFormData() {
    const formData = {};
    const form = document.getElementById('predictionForm');
    const inputs = form.querySelectorAll('input[type="number"]');
    
    inputs.forEach(input => {
        formData[input.name] = parseFloat(input.value) || 0;
    });
    
    return formData;
}

// Validar datos del formulario
function validateFormData(formData) {
    return Object.values(formData).every(value => !isNaN(value));
}

// Mostrar resultado de predicción
function displayPredictionResult(data) {
    const resultCard = document.getElementById('predictionResult');
    const priceElement = document.getElementById('predictedPrice');
    const modelElement = document.getElementById('modelUsed');
    
    priceElement.textContent = data.formatted_price;
    modelElement.textContent = `Predicción realizada con: ${data.model_used}`;
    
    resultCard.style.display = 'block';
    resultCard.classList.add('fade-in-up');
    
    // Scroll a resultado
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Mostrar métricas
function displayMetrics(metrics) {
    const metricsCard = document.getElementById('metricsResult');
    const metricsContent = document.getElementById('metricsContent');
    
    let html = `
        <div class="table-responsive">
            <table class="table metrics-table">
                <thead>
                    <tr>
                        <th>Modelo</th>
                        <th>R² Score</th>
                        <th>MSE</th>
                        <th>Rendimiento</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    Object.keys(metrics).forEach(modelName => {
        const metric = metrics[modelName];
        const performance = getPerformanceLevel(metric.r2);
        const performanceClass = getPerformanceClass(metric.r2);
        
        html += `
            <tr>
                <td><strong>${modelName}</strong></td>
                <td>${metric.r2}</td>
                <td>${metric.mse}</td>
                <td><span class="badge ${performanceClass}">${performance}</span></td>
            </tr>
        `;
    });
    
    html += `
                </tbody>
            </table>
        </div>
        <div class="mt-3">
            <small class="text-muted">
                <strong>R² Score:</strong> Coeficiente de determinación (más cercano a 1 es mejor)<br>
                <strong>MSE:</strong> Error cuadrático medio (menor es mejor)
            </small>
        </div>
    `;
    
    metricsContent.innerHTML = html;
    metricsCard.style.display = 'block';
    metricsCard.classList.add('fade-in-up');
    
    // Scroll a métricas
    metricsCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Mostrar gráficos
function displayPlots(plotBase64) {
    const plotsCard = document.getElementById('plotsResult');
    const plotsContent = document.getElementById('plotsContent');
    
    plotsContent.innerHTML = `
        <div class="plot-container">
            <h6 class="mb-3">Comparación de Predicciones vs Valores Reales</h6>
            <img src="data:image/png;base64,${plotBase64}" alt="Gráficos de Comparación" class="img-fluid">
            <div class="mt-3">
                <small class="text-muted">
                    Los puntos más cercanos a la línea diagonal indican mejores predicciones.
                </small>
            </div>
        </div>
    `;
    
    plotsCard.style.display = 'block';
    plotsCard.classList.add('fade-in-up');
    
    // Scroll a gráficos
    plotsCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Obtener nivel de rendimiento
function getPerformanceLevel(r2) {
    if (r2 >= 0.9) return 'Excelente';
    if (r2 >= 0.8) return 'Muy Bueno';
    if (r2 >= 0.7) return 'Bueno';
    if (r2 >= 0.6) return 'Regular';
    return 'Bajo';
}

// Obtener clase CSS para rendimiento
function getPerformanceClass(r2) {
    if (r2 >= 0.9) return 'bg-success';
    if (r2 >= 0.8) return 'bg-info';
    if (r2 >= 0.7) return 'bg-primary';
    if (r2 >= 0.6) return 'bg-warning';
    return 'bg-danger';
}

// Ocultar todos los resultados
function hideAllResults() {
    document.getElementById('predictionResult').style.display = 'none';
    document.getElementById('metricsResult').style.display = 'none';
    document.getElementById('plotsResult').style.display = 'none';
    document.getElementById('welcomeCard').style.display = 'none';
}

// Mostrar/ocultar loading
function showLoading(show) {
    const spinner = document.getElementById('loadingSpinner');
    spinner.style.display = show ? 'block' : 'none';
}

// Mostrar alerta
function showAlert(message, type = 'info') {
    // Crear elemento de alerta
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Añadir al DOM
    document.body.appendChild(alertDiv);
    
    // Auto-remover después de 5 segundos
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Función para limpiar formulario
function clearForm() {
    const form = document.getElementById('predictionForm');
    form.reset();
    fillDefaultValues();
}

// Función para exportar métricas (opcional)
function exportMetrics() {
    if (!currentModelMetrics) {
        showAlert('No hay métricas para exportar', 'warning');
        return;
    }
    
    const csvContent = "data:text/csv;charset=utf-8," 
        + "Modelo,R2_Score,MSE\n"
        + Object.keys(currentModelMetrics).map(model => 
            `${model},${currentModelMetrics[model].r2},${currentModelMetrics[model].mse}`
        ).join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "model_metrics.csv");
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Funciones de utilidad adicionales
function formatNumber(num, decimals = 2) {
    return Number(num).toFixed(decimals);
}

function validatePositiveNumber(value) {
    return !isNaN(value) && value >= 0;
}

// Event listeners adicionales
document.addEventListener('keydown', function(e) {
    // Esc para cerrar resultados
    if (e.key === 'Escape') {
        hideAllResults();
        document.getElementById('welcomeCard').style.display = 'block';
    }
    
    // Enter para predecir
    if (e.key === 'Enter' && e.target.type === 'number') {
        e.preventDefault();
        makePrediction();
    }
});

// Prevenir envío del formulario
document.getElementById('predictionForm').addEventListener('submit', function(e) {
    e.preventDefault();
    makePrediction();
});