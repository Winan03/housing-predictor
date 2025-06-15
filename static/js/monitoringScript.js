// =============================================================================
// MONITORING DASHBOARD SCRIPT - FIXED VERSION
// =============================================================================

class MonitoringDashboard {
    constructor() {
        this.autoRefreshEnabled = true;
        this.refreshInterval = null;
        this.refreshIntervalTime = 60000; // 1 minuto
        this.isLoading = false;
        this.charts = {};
        
        this.init();
    }
    
    init() {
        console.log('🚀 Inicializando Dashboard de Monitoreo...');
        
        // Configurar event listeners
        this.setupEventListeners();
        
        // Cargar datos iniciales
        this.loadAllData();
        
        // Configurar auto-refresh
        this.setupAutoRefresh();
        
        // Actualizar timestamp
        this.updateLastRefresh();
        
        console.log('✅ Dashboard de Monitoreo inicializado');
    }
    
    setupEventListeners() {
        // Botón de refresh manual
        const refreshBtn = document.getElementById('refreshData');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.handleManualRefresh());
        }
        
        // Toggle auto-refresh
        const autoRefreshBtn = document.getElementById('toggleAutoRefresh');
        if (autoRefreshBtn) {
            autoRefreshBtn.addEventListener('click', () => this.toggleAutoRefresh());
        }
        
        // Verificación manual de drift de rendimiento
        const checkPerformanceBtn = document.getElementById('checkPerformanceDrift');
        if (checkPerformanceBtn) {
            checkPerformanceBtn.addEventListener('click', () => this.checkPerformanceDrift());
        }
        
        // Verificación manual de drift de datos
        const checkDataBtn = document.getElementById('checkDataDrift');
        if (checkDataBtn) {
            checkDataBtn.addEventListener('click', () => this.checkDataDrift());
        }
    }
    
    setupAutoRefresh() {
        if (this.autoRefreshEnabled) {
            this.refreshInterval = setInterval(() => {
                if (!this.isLoading) {
                    this.loadAllData();
                }
            }, this.refreshIntervalTime);
        }
    }
    
    toggleAutoRefresh() {
        this.autoRefreshEnabled = !this.autoRefreshEnabled;
        const statusSpan = document.getElementById('auto-refresh-status');
        
        if (this.autoRefreshEnabled) {
            this.setupAutoRefresh();
            if (statusSpan) statusSpan.textContent = 'ON';
            this.showNotification('Auto-refresh activado', 'success');
        } else {
            if (this.refreshInterval) {
                clearInterval(this.refreshInterval);
                this.refreshInterval = null;
            }
            if (statusSpan) statusSpan.textContent = 'OFF';
            this.showNotification('Auto-refresh desactivado', 'info');
        }
    }
    
    async handleManualRefresh() {
        const refreshBtn = document.getElementById('refreshData');
        if (refreshBtn) {
            refreshBtn.disabled = true;
            refreshBtn.innerHTML = '🔄 Actualizando...';
        }
        
        try {
            await this.loadAllData();
            this.showNotification('Datos actualizados correctamente', 'success');
        } catch (error) {
            this.showNotification('Error al actualizar datos', 'error');
        } finally {
            if (refreshBtn) {
                refreshBtn.disabled = false;
                refreshBtn.innerHTML = '🔄 Actualizar Datos';
            }
        }
    }
    
    async loadAllData() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        
        try {
            // Cargar estadísticas básicas
            await this.loadSystemStats();
            
            // Cargar gráficos
            await this.loadPerformancePlot();
            await this.loadFeedbackPlot();
            
            // Cargar alertas de drift
            await this.loadDriftAlerts();
            
            this.updateLastRefresh();
            
        } catch (error) {
            console.error('Error cargando datos:', error);
            this.showNotification('Error al cargar algunos datos', 'warning');
        } finally {
            this.isLoading = false;
        }
    }
    
    // FIXED: Added JSON sanitization to handle invalid values
    sanitizeJSON(obj) {
        const sanitize = (value) => {
            if (typeof value === 'number') {
                if (!isFinite(value)) {
                    return 0; // Replace Infinity/NaN with 0
                }
                return value;
            }
            if (typeof value === 'object' && value !== null) {
                if (Array.isArray(value)) {
                    return value.map(sanitize);
                }
                const sanitized = {};
                for (const [key, val] of Object.entries(value)) {
                    sanitized[key] = sanitize(val);
                }
                return sanitized;
            }
            return value;
        };
        return sanitize(obj);
    }
    
    async loadSystemStats() {
        try {
            const response = await fetch('/api/monitor/stats');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const rawText = await response.text();
            let data;
            
            try {
                // Parse JSON with error handling
                data = JSON.parse(rawText);
                // Sanitize the data to handle invalid numeric values
                data = this.sanitizeJSON(data);
            } catch (parseError) {
                console.error('Error parsing JSON:', parseError);
                console.error('Raw response:', rawText);
                throw new Error('Invalid JSON response from server');
            }
            
            if (data.success) {
                this.updateSystemStats(data.stats);
            } else {
                throw new Error(data.error || 'Error desconocido');
            }
        } catch (error) {
            console.error('Error cargando estadísticas:', error);
            this.setLoadingError('system-health', 'Error');
            this.setLoadingError('recent-predictions', 'Error');
            this.setLoadingError('drift-alerts-count', 'Error');
            this.showNotification(`Error cargando estadísticas: ${error.message}`, 'error');
        }
    }
    
    updateSystemStats(stats) {
        try {
            // Estado del sistema - with safe access
            const systemHealth = document.getElementById('system-health');
            if (systemHealth) {
                const driftStatus = stats.drift_status || {};
                const alertsCount = driftStatus.critical_alerts || 0;
                const totalAlerts = driftStatus.total_alerts || 0;
                
                if (alertsCount > 0) {
                    systemHealth.innerHTML = '<span class="text-red-600">⚠️ Alertas Críticas</span>';
                } else if (totalAlerts > 0) {
                    systemHealth.innerHTML = '<span class="text-yellow-600">⚠️ Alertas Menores</span>';
                } else {
                    systemHealth.innerHTML = '<span class="text-green-600">✅ Saludable</span>';
                }
            }
            
            // Predicciones recientes - with safe access
            const recentPreds = stats.recent_predictions || {};
            this.updateElement('recent-predictions', recentPreds.count || 0);
            
            // Alertas de drift - with safe access
            const driftStatus = stats.drift_status || {};
            this.updateElement('drift-alerts-count', driftStatus.total_alerts || 0);
            
            // Feedback de usuarios - with safe access and numeric validation
            const feedback = stats.user_feedback || {};
            this.updateElement('feedback-total', feedback.total_feedback || 0);
            
            const avgRating = feedback.avg_rating;
            this.updateElement('feedback-avg-rating', 
                (typeof avgRating === 'number' && isFinite(avgRating)) ? avgRating.toFixed(1) : '0.0');
            
            this.updateElement('feedback-positive', feedback.positive_feedback || 0);
            
            const satisfactionRate = feedback.satisfaction_rate;
            this.updateElement('feedback-satisfaction', 
                (typeof satisfactionRate === 'number' && isFinite(satisfactionRate)) 
                    ? `${(satisfactionRate * 100).toFixed(1)}%` 
                    : '0.0%');
                    
        } catch (error) {
            console.error('Error actualizando estadísticas:', error);
            this.showNotification('Error procesando estadísticas', 'warning');
        }
    }
    
    async loadPerformancePlot() {
        const container = document.getElementById('performance-plot');
        if (!container) return;
    
        this.showLoading(container, 'Cargando gráfico de rendimiento...');
    
        try {
            const response = await fetch('/api/monitor/plots');
    
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
    
            const data = await response.json();
    
            if (data.success && data.plots && data.plots.performance_history) {
                container.innerHTML = `<img src="data:image/png;base64,${data.plots.performance_history}" 
                                     alt="Gráfico de Rendimiento" class="w-full h-full object-contain">`;
            } else {
                this.showPlaceholder(container, '📈', 'Gráfico de rendimiento no disponible');
            }
        } catch (error) {
            console.error('Error cargando gráfico de rendimiento:', error);
            this.showPlaceholder(container, '❌', 'Error al cargar gráfico de rendimiento');
        }
    }
    
    async loadFeedbackPlot() {
        const container = document.getElementById('feedback-plot');
        if (!container) return;
    
        this.showLoading(container, 'Cargando gráfico de feedback...');
    
        try {
            const response = await fetch('/api/monitor/plots');
    
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
    
            const data = await response.json();
    
            if (data.success && data.plots && data.plots.user_feedback) {
                container.innerHTML = `<img src="data:image/png;base64,${data.plots.user_feedback}" 
                                     alt="Gráfico de Feedback" class="w-full h-full object-contain">`;
            } else {
                this.showPlaceholder(container, '💬', 'Gráfico de feedback no disponible');
            }
        } catch (error) {
            console.error('Error cargando gráfico de feedback:', error);
            this.showPlaceholder(container, '❌', 'Error al cargar gráfico de feedback');
        }
    }    
    
    async loadDriftAlerts() {
        const container = document.getElementById('drift-alerts-list');
        if (!container) return;
        
        this.showLoading(container, 'Cargando alertas...');
        
        try {
            const response = await fetch('/api/monitor/stats');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const rawText = await response.text();
            let data;
            
            try {
                data = JSON.parse(rawText);
                data = this.sanitizeJSON(data); // Sanitize data
            } catch (parseError) {
                console.error('Error parsing JSON in loadDriftAlerts:', parseError);
                throw new Error('Invalid JSON response');
            }
            
            if (data.success) {
                const driftStatus = data.stats.drift_status || {};
                const alerts = driftStatus.recent_alerts || [];
                this.displayDriftAlerts(container, alerts);
            } else {
                throw new Error(data.error || 'Error desconocido');
            }
        } catch (error) {
            console.error('Error cargando alertas:', error);
            container.innerHTML = `
                <div class="text-center text-red-600 p-4">
                    ❌ Error al cargar alertas: ${error.message}
                </div>
            `;
        }
    }
    
    displayDriftAlerts(container, alerts) {
        if (!Array.isArray(alerts) || alerts.length === 0) {
            container.innerHTML = `
                <div class="text-center text-green-600 p-8">
                    ✅ No hay alertas de drift recientes
                    <p class="text-sm text-gray-500 mt-2">El modelo está funcionando correctamente</p>
                </div>
            `;
            return;
        }
        
        const tableHTML = `
            <table class="drift-alerts-table">
                <thead>
                    <tr>
                        <th>Fecha</th>
                        <th>Tipo</th>
                        <th>Métrica</th>
                        <th>Severidad</th>
                        <th>Descripción</th>
                    </tr>
                </thead>
                <tbody>
                    ${alerts.map(alert => {
                        // Sanitize alert data
                        const sanitizedAlert = this.sanitizeJSON(alert);
                        return `
                            <tr>
                                <td>${this.formatDate(sanitizedAlert.timestamp)}</td>
                                <td>${sanitizedAlert.drift_type || 'N/A'}</td>
                                <td>${sanitizedAlert.metric || 'N/A'}</td>
                                <td>
                                    <span class="severity-badge ${sanitizedAlert.severity || 'low'}">
                                        ${this.getSeverityIcon(sanitizedAlert.severity)} ${sanitizedAlert.severity || 'low'}
                                    </span>
                                </td>
                                <td class="max-w-xs truncate" title="${sanitizedAlert.description || ''}">
                                    ${sanitizedAlert.description || 'Sin descripción'}
                                </td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        `;
        
        container.innerHTML = tableHTML;
    }
    
    async checkPerformanceDrift() {
        const button = document.getElementById('checkPerformanceDrift');
        const statusDiv = document.getElementById('drift-status');
        
        if (button) {
            button.disabled = true;
            button.innerHTML = '🔍 Verificando...';
        }
        
        try {
            const response = await fetch('/api/monitor/drift-check', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    data: this.generateSampleData(),
                    check_type: 'performance'
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            const sanitizedResult = this.sanitizeJSON(result);
            
            if (sanitizedResult.success) {
                this.showDriftStatus(statusDiv, sanitizedResult, 'rendimiento');
            } else {
                throw new Error(sanitizedResult.error || 'Error en verificación');
            }
            
        } catch (error) {
            console.error('Error verificando drift de rendimiento:', error);
            this.showDriftStatus(statusDiv, {
                success: false,
                error: error.message
            }, 'rendimiento');
        } finally {
            if (button) {
                button.disabled = false;
                button.innerHTML = '🔍 Verificar Drift de Rendimiento';
            }
        }
    }
    
    async checkDataDrift() {
        const button = document.getElementById('checkDataDrift');
        const statusDiv = document.getElementById('drift-status');
        
        if (button) {
            button.disabled = true;
            button.innerHTML = '📊 Verificando...';
        }
        
        try {
            const response = await fetch('/api/monitor/drift-check', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    data: this.generateSampleData(),
                    check_type: 'data'
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const result = await response.json();
            const sanitizedResult = this.sanitizeJSON(result);
            
            if (sanitizedResult.success) {
                this.showDriftStatus(statusDiv, sanitizedResult, 'datos');
            } else {
                throw new Error(sanitizedResult.error || 'Error en verificación');
            }
            
        } catch (error) {
            console.error('Error verificando drift de datos:', error);
            this.showDriftStatus(statusDiv, {
                success: false,
                error: error.message
            }, 'datos');
        } finally {
            if (button) {
                button.disabled = false;
                button.innerHTML = '📊 Verificar Drift de Datos';
            }
        }
    }
    
    showDriftStatus(container, result, type) {
        if (!container) return;
        
        container.style.display = 'block';
        
        if (!result.success) {
            container.className = 'status-message error';
            container.innerHTML = `❌ Error verificando drift de ${type}: ${result.error}`;
            return;
        }
        
        const driftDetected = result.drift_detected || false;
        const alertsCount = result.alerts_count || 0;
        const severityBreakdown = result.severity_breakdown || { high: 0, medium: 0, low: 0 };
        
        if (driftDetected) {
            const severity = severityBreakdown.high > 0 ? 'error' : 
                           severityBreakdown.medium > 0 ? 'warning' : 'info';
            
            container.className = `status-message ${severity}`;
            container.innerHTML = `
                ⚠️ <strong>Drift detectado en ${type}!</strong><br>
                <small>
                    Total de alertas: ${alertsCount} 
                    (Alto: ${severityBreakdown.high}, 
                     Medio: ${severityBreakdown.medium}, 
                     Bajo: ${severityBreakdown.low})
                </small>
            `;
        } else {
            container.className = 'status-message success';
            container.innerHTML = `✅ No se detectó drift en ${type}. El modelo está funcionando correctamente.`;
        }
        
        // Auto-ocultar después de 10 segundos
        setTimeout(() => {
            container.style.display = 'none';
        }, 10000);
    }
    
    generateSampleData() {
        // Generar datos de muestra válidos (sin NaN o Infinity)
        const sampleData = [];
        for (let i = 0; i < 100; i++) {
            const area = Math.random() * 200 + 50;
            const bedrooms = Math.floor(Math.random() * 5) + 1;
            const bathrooms = Math.floor(Math.random() * 3) + 1;
            const locationScore = Math.random() * 10;
            const age = Math.floor(Math.random() * 50);
            const condition = Math.floor(Math.random() * 5) + 1;
            
            // Ensure all values are finite
            sampleData.push({
                area: isFinite(area) ? area : 100,
                bedrooms: isFinite(bedrooms) ? bedrooms : 2,
                bathrooms: isFinite(bathrooms) ? bathrooms : 1,
                location_score: isFinite(locationScore) ? locationScore : 5,
                age: isFinite(age) ? age : 10,
                condition: isFinite(condition) ? condition : 3
            });
        }
        return sampleData;
    }
    
    // Funciones de utilidad
    updateElement(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            // Ensure value is safe to display
            const safeValue = (typeof value === 'number' && !isFinite(value)) ? 0 : value;
            element.textContent = safeValue;
        }
    }
    
    setLoadingError(elementId, errorText) {
        this.updateElement(elementId, errorText);
    }
    
    showLoading(container, message = 'Cargando...') {
        container.innerHTML = `
            <div class="loading-spinner">
                <div class="spinner"></div>
                <p class="text-gray-600">${message}</p>
            </div>
        `;
    }
    
    showPlaceholder(container, icon, message) {
        container.innerHTML = `
            <div class="flex flex-col items-center justify-center text-gray-500">
                <div class="text-4xl mb-2">${icon}</div>
                <p class="text-center">${message}</p>
            </div>
        `;
    }
    
    formatDate(dateString) {
        if (!dateString) return 'N/A';
        
        try {
            const date = new Date(dateString);
            if (isNaN(date.getTime())) {
                return 'Fecha inválida';
            }
            return date.toLocaleString('es-ES', {
                day: '2-digit',
                month: '2-digit',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch (error) {
            return 'Fecha inválida';
        }
    }
    
    getSeverityIcon(severity) {
        switch (severity) {
            case 'high': return '🔴';
            case 'medium': return '🟡';
            case 'low': return '🟢';
            default: return '⚪';
        }
    }
    
    updateLastRefresh() {
        const lastRefreshElement = document.getElementById('last-refresh');
        if (lastRefreshElement) {
            const now = new Date();
            lastRefreshElement.textContent = now.toLocaleString('es-ES', {
                day: '2-digit',
                month: '2-digit',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            });
        }
    }
    
    showNotification(message, type = 'info') {
        // Crear elemento de notificación
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 p-4 rounded-md shadow-lg z-50 fade-in`;
        
        switch (type) {
            case 'success':
                notification.className += ' bg-green-100 border border-green-400 text-green-800';
                break;
            case 'error':
                notification.className += ' bg-red-100 border border-red-400 text-red-800';
                break;  
            case 'warning':
                notification.className += ' bg-yellow-100 border border-yellow-400 text-yellow-800';
                break;
            default:
                notification.className += ' bg-blue-100 border border-blue-400 text-blue-800';
        }
        
        notification.innerHTML = `
            <div class="flex items-center">
                <span class="mr-2">${this.getNotificationIcon(type)}</span>
                <span>${message}</span>
                <button class="ml-4 text-lg font-bold hover:opacity-75" onclick="this.parentElement.parentElement.remove()">
                    ×
                </button>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remover después de 5 segundos
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }
    
    getNotificationIcon(type) {
        switch (type) {
            case 'success': return '✅';
            case 'error': return '❌';
            case 'warning': return '⚠️';
            default: return 'ℹ️';
        }
    }
}

// =============================================================================
// INICIALIZACIÓN
// =============================================================================

// Inicializar dashboard cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎯 DOM cargado, inicializando Dashboard de Monitoreo...');
    
    // Verificar que los elementos necesarios existan
    const requiredElements = [
        'system-health',
        'recent-predictions',
        'drift-alerts-count',
        'performance-plot',
        'feedback-plot',
        'drift-alerts-list'
    ];
    
    const missingElements = requiredElements.filter(id => !document.getElementById(id));
    
    if (missingElements.length > 0) {
        console.warn('⚠️ Elementos faltantes en el DOM:', missingElements);
    }
    
    // Inicializar dashboard
    try {
        window.monitoringDashboard = new MonitoringDashboard();
    } catch (error) {
        console.error('❌ Error inicializando Dashboard de Monitoreo:', error);
        
        // Mostrar error en la página
        const errorDiv = document.createElement('div');
        errorDiv.className = 'fixed top-4 left-4 right-4 bg-red-100 border border-red-400 text-red-800 p-4 rounded-md z-50';
        errorDiv.innerHTML = `
            <div class="flex items-center">
                <span class="mr-2">❌</span>
                <span>Error inicializando el dashboard: ${error.message}</span>
            </div>
        `;
        document.body.appendChild(errorDiv);
    }
});

// Manejar errores globales
window.addEventListener('error', function(event) {
    console.error('Error global capturado:', event.error);
});

// Manejar promesas rechazadas
window.addEventListener('unhandledrejection', function(event) {
    console.error('Promesa rechazada no manejada:', event.reason);
});

console.log('📜 Script de monitoreo cargado correctamente');