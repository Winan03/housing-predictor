// indexScript.js - Sistema de Predicción de Precios de Vivienda

document.addEventListener('DOMContentLoaded', function() {
    
    // ========================================
    // INICIALIZACIÓN Y CONFIGURACIÓN
    // ========================================
    
    // Elementos del DOM
    const elements = {
        navbar: document.getElementById('mainNavbar'),
        heroSection: document.getElementById('heroSection'),
        accuracyBar: document.getElementById('accuracyBar'),
        accuracyText: document.getElementById('accuracyText'),
        counters: document.querySelectorAll('.counter'),
        startPredictionBtn: document.getElementById('startPredictionBtn'),
        demoBtn: document.getElementById('demoBtn'),
        mainCtaBtn: document.getElementById('mainCtaBtn'),
        learnMoreBtn: document.getElementById('learnMoreBtn'),
        modalStartBtn: document.getElementById('modalStartBtn'),
        featureCards: document.querySelectorAll('.feature-card'),
        stepItems: document.querySelectorAll('.step-item'),
        infoModal: new bootstrap.Modal(document.getElementById('infoModal'))
    };

    // ========================================
    // EFECTOS DE SCROLL Y NAVBAR
    // ========================================
    
    function handleScroll() {
        const scrolled = window.scrollY;
        
        // Efecto navbar transparente/sólido
        if (scrolled > 100) {
            elements.navbar.classList.add('navbar-scrolled');
            elements.navbar.style.background = 'rgba(13, 110, 253, 0.95)';
            elements.navbar.style.backdropFilter = 'blur(10px)';
        } else {
            elements.navbar.classList.remove('navbar-scrolled');
            elements.navbar.style.background = '';
            elements.navbar.style.backdropFilter = '';
        }
        
        // Parallax en hero section
        if (elements.heroSection && scrolled < window.innerHeight) {
            elements.heroSection.style.transform = `translateY(${scrolled * 0.5}px)`;
        }
    }

    // ========================================
    // ANIMACIONES DE CONTADORES
    // ========================================
    
    function animateCounter(element, target, duration = 2000) {
        const startValue = 0;
        const increment = target / (duration / 16);
        let currentValue = startValue;
        
        const updateCounter = () => {
            currentValue += increment;
            if (currentValue >= target) {
                element.textContent = target;
                return;
            }
            element.textContent = Math.floor(currentValue);
            requestAnimationFrame(updateCounter);
        };
        
        updateCounter();
    }

    function initCounterAnimations() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && !entry.target.classList.contains('animated')) {
                    const target = parseInt(entry.target.dataset.target);
                    animateCounter(entry.target, target);
                    entry.target.classList.add('animated');
                }
            });
        }, { threshold: 0.5 });

        elements.counters.forEach(counter => {
            observer.observe(counter);
        });
    }

    // ========================================
    // ANIMACIÓN DE BARRA DE PRECISIÓN
    // ========================================
    
    function animateAccuracyBar() {
        setTimeout(() => {
            if (elements.accuracyBar) {
                elements.accuracyBar.style.transition = 'width 2s ease-in-out';
                elements.accuracyBar.style.width = '94%';
            }
        }, 1000);
    }

    // ========================================
    // EFECTOS HOVER EN TARJETAS
    // ========================================
    
    function initCardHoverEffects() {
        elements.featureCards.forEach(card => {
            card.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-10px) scale(1.02)';
                this.style.transition = 'all 0.3s ease';
                this.style.boxShadow = '0 20px 40px rgba(0,0,0,0.1)';
            });
            
            card.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0) scale(1)';
                this.style.boxShadow = '';
            });
        });
    }

    // ========================================
    // ANIMACIONES DE PASOS
    // ========================================
    
    function initStepAnimations() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const stepNumber = entry.target.dataset.step;
                    entry.target.style.opacity = '0';
                    entry.target.style.transform = 'translateX(-30px)';
                    
                    setTimeout(() => {
                        entry.target.style.transition = 'all 0.6s ease';
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateX(0)';
                    }, stepNumber * 200);
                }
            });
        }, { threshold: 0.3 });

        elements.stepItems.forEach(item => {
            observer.observe(item);
        });
    }

    // ========================================
    // SMOOTH SCROLLING PARA NAVEGACIÓN
    // ========================================
    
    function initSmoothScrolling() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                e.preventDefault();
                const targetId = this.getAttribute('href');
                const targetElement = document.querySelector(targetId);
                
                if (targetElement) {
                    const offsetTop = targetElement.offsetTop - 80;
                    window.scrollTo({
                        top: offsetTop,
                        behavior: 'smooth'
                    });
                }
            });
        });
    }

    // ========================================
    // EFECTOS DE TYPING ANIMATION
    // ========================================
    
    function typeWriter(element, text, speed = 100) {
        element.innerHTML = '';
        let i = 0;
        
        function type() {
            if (i < text.length) {
                element.innerHTML += text.charAt(i);
                i++;
                setTimeout(type, speed);
            }
        }
        
        type();
    }

    // ========================================
    // PARTÍCULAS FLOTANTES
    // ========================================
    
    function createFloatingParticles() {
        const particlesContainer = document.createElement('div');
        particlesContainer.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        `;
        
        document.body.appendChild(particlesContainer);
        
        for (let i = 0; i < 20; i++) {
            const particle = document.createElement('div');
            particle.style.cssText = `
                position: absolute;
                width: ${Math.random() * 10 + 5}px;
                height: ${Math.random() * 10 + 5}px;
                background: rgba(102, 126, 234, 0.1);
                border-radius: 50%;
                animation: float ${Math.random() * 10 + 10}s infinite linear;
                left: ${Math.random() * 100}%;
                top: ${Math.random() * 100}%;
            `;
            particlesContainer.appendChild(particle);
        }
        
        // CSS para animación de partículas
        const style = document.createElement('style');
        style.textContent = `
            @keyframes float {
                0% { transform: translateY(0px) rotate(0deg); opacity: 0; }
                10% { opacity: 1; }
                90% { opacity: 1; }
                100% { transform: translateY(-100vh) rotate(360deg); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }

    // ========================================
    // MANEJADORES DE EVENTOS DE BOTONES
    // ========================================
    
    function initButtonHandlers() {
        // Botones para comenzar predicción
        const predictionButtons = [
            elements.startPredictionBtn,
            elements.mainCtaBtn,
            elements.modalStartBtn
        ];
        
        predictionButtons.forEach(btn => {
            if (btn) {
                btn.addEventListener('click', function() {
                    // Efecto de ripple
                    createRippleEffect(this);
                    
                    // Animación de loading
                    const originalHTML = this.innerHTML;
                    this.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Cargando...';
                    this.disabled = true;
                    
                    setTimeout(() => {
                        // Aquí iría la redirección a la página de predicción
                        window.location.href = '/predict'; // Ajusta la URL según tu estructura
                    }, 1500);
                });
            }
        });
        
        // Botón de demo
        if (elements.demoBtn) {
            elements.demoBtn.addEventListener('click', function() {
                createRippleEffect(this);
                showDemoAnimation();
            });
        }
        
        // Botón de más información
        if (elements.learnMoreBtn) {
            elements.learnMoreBtn.addEventListener('click', function() {
                createRippleEffect(this);
                elements.infoModal.show();
            });
        }
    }

    // ========================================
    // EFECTO RIPPLE PARA BOTONES
    // ========================================
    
    function createRippleEffect(button) {
        const ripple = document.createElement('span');
        const rect = button.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = event.clientX - rect.left - size / 2;
        const y = event.clientY - rect.top - size / 2;
        
        ripple.style.cssText = `
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.6);
            transform: scale(0);
            animation: ripple 0.6s linear;
            width: ${size}px;
            height: ${size}px;
            left: ${x}px;
            top: ${y}px;
        `;
        
        button.style.position = 'relative';
        button.style.overflow = 'hidden';
        button.appendChild(ripple);
        
        setTimeout(() => {
            ripple.remove();
        }, 600);
        
        // CSS para animación ripple
        if (!document.querySelector('#ripple-style')) {
            const style = document.createElement('style');
            style.id = 'ripple-style';
            style.textContent = `
                @keyframes ripple {
                    to {
                        transform: scale(4);
                        opacity: 0;
                    }
                }
            `;
            document.head.appendChild(style);
        }
    }

    // ========================================
    // ANIMACIÓN DE DEMO
    // ========================================
    
    function showDemoAnimation() {
        const demoOverlay = document.createElement('div');
        demoOverlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 1.5rem;
            text-align: center;
        `;
        
        demoOverlay.innerHTML = `
            <div>
                <i class="fas fa-play-circle fa-5x mb-4 text-primary"></i>
                <h3>Demo Interactivo</h3>
                <p class="lead">Próximamente disponible</p>
                <button class="btn btn-outline-light mt-3" onclick="this.parentElement.parentElement.remove()">
                    Cerrar
                </button>
            </div>
        `;
        
        document.body.appendChild(demoOverlay);
        
        setTimeout(() => {
            demoOverlay.remove();
        }, 3000);
    }

    // ========================================
    // ACTUALIZACIÓN DINÁMICA DE ESTADÍSTICAS
    // ========================================
    
    function updateStatistics() {
        const stats = {
            predictions: Math.floor(Math.random() * 50) + 1500,
            accuracy: Math.floor(Math.random() * 3) + 93,
            responseTime: Math.floor(Math.random() * 2) + 1
        };
        
        // Actualizar cada 30 segundos
        setInterval(() => {
            stats.predictions += Math.floor(Math.random() * 5) + 1;
            const predictionCounter = document.querySelector('[data-target="1500"]');
            if (predictionCounter && predictionCounter.classList.contains('animated')) {
                predictionCounter.textContent = stats.predictions;
            }
        }, 30000);
    }

    // ========================================
    // TOOLTIP DINÁMICOS
    // ========================================
    
    function initTooltips() {
        // Inicializar tooltips de Bootstrap si están presentes
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // ========================================
    // DETECCIÓN DE DISPOSITIVO MÓVIL
    // ========================================
    
    function isMobile() {
        return window.innerWidth <= 768;
    }

    function optimizeForMobile() {
        if (isMobile()) {
            // Reducir efectos para mejor rendimiento en móviles
            document.querySelectorAll('.feature-card').forEach(card => {
                card.style.transition = 'none';
            });
        }
    }

    // ========================================
    // INICIALIZACIÓN PRINCIPAL
    // ========================================
    
    function init() {
        console.log('🚀 Inicializando Sistema de Predicción de Viviendas IA...');
        
        // Inicializar todas las funcionalidades
        initCounterAnimations();
        initCardHoverEffects();
        initStepAnimations();
        initSmoothScrolling();
        initButtonHandlers();
        initTooltips();
        
        // Animaciones iniciales
        animateAccuracyBar();
        
        // Efectos visuales
        if (!isMobile()) {
            createFloatingParticles();
        }
        
        // Estadísticas dinámicas
        updateStatistics();
        
        // Optimizaciones
        optimizeForMobile();
        
        // Event listeners
        window.addEventListener('scroll', handleScroll);
        window.addEventListener('resize', optimizeForMobile);
        
        console.log('✅ Sistema inicializado correctamente');
    }

    // ========================================
    // FUNCIONES DE UTILIDAD
    // ========================================
    
    // Función para mostrar notificaciones
    window.showNotification = function(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} position-fixed`;
        notification.style.cssText = `
            top: 20px;
            right: 20px;
            z-index: 9999;
            min-width: 300px;
            animation: slideIn 0.3s ease;
        `;
        notification.innerHTML = `
            <i class="fas fa-info-circle me-2"></i>
            ${message}
            <button type="button" class="btn-close float-end" onclick="this.parentElement.remove()"></button>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 5000);
    };

    // CSS adicional para animaciones
    const additionalStyles = document.createElement('style');
    additionalStyles.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        .navbar-scrolled {
            box-shadow: 0 2px 20px rgba(0,0,0,0.1) !important;
        }
        
        .feature-card:hover {
            transform: translateY(-10px) scale(1.02) !important;
            transition: all 0.3s ease !important;
        }
        
        @media (prefers-reduced-motion: reduce) {
            * {
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
    `;
    document.head.appendChild(additionalStyles);

    // Inicializar cuando el DOM esté listo
    init();
    
    // Mensaje de bienvenida para desarrolladores
    console.log(`
    🏠 Sistema de Predicción de Viviendas IA
    ────────────────────────────────────────
    ✨ Desarrollado con tecnología avanzada
    🤖 Powered by Machine Learning
    📊 94% de precisión en predicciones
    ────────────────────────────────────────
    `);
});