#!/usr/bin/env python3
"""
Sistema de Verificación de Despliegue
=====================================

Este script verifica que todos los componentes necesarios para el despliegue
estén correctamente configurados y funcionando.

Autor: Sistema de ML
Fecha: 2025
"""

import os
import sys
import json
import warnings
import traceback
from pathlib import Path
from datetime import datetime
import importlib.util

# Suprimir warnings durante verificación
warnings.filterwarnings('ignore')

class DeploymentVerifier:
    """
    Clase principal para verificar el estado del despliegue
    """
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'UNKNOWN',
            'checks': {},
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        self.required_files = [
            'app.py',
            'model.py',
            'requirements.txt',
            'templates/index.html',
            'templates/prediccion.html',
            'static/css/style.css',
            'static/js/script.js'
        ]
        self.required_model_files = [
            'models/metadata.pkl',
            'models/scaler.pkl',
            'models/feature_selector.pkl'
        ]
    
    def print_header(self):
        """Imprime el header del verificador"""
        print("🔍 VERIFICADOR DE DESPLIEGUE")
        print("=" * 60)
        print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def check_python_version(self):
        """Verifica la versión de Python"""
        print("🐍 Verificando versión de Python...")
        
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major >= 3 and version.minor >= 7:
            self.results['checks']['python_version'] = {
                'status': 'PASS',
                'version': version_str,
                'message': f'Python {version_str} ✅'
            }
            print(f"   ✅ Python {version_str} - Compatible")
        else:
            self.results['checks']['python_version'] = {
                'status': 'FAIL',
                'version': version_str,
                'message': f'Python {version_str} es muy antiguo'
            }
            print(f"   ❌ Python {version_str} - Requiere Python 3.7+")
            self.results['errors'].append(f"Python {version_str} no es compatible")
    
    def check_required_packages(self):
        """Verifica que las librerías requeridas estén instaladas"""
        print("\n📦 Verificando librerías requeridas...")
        
        required_packages = {
            'flask': 'Flask',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'sklearn': 'scikit-learn',
            'joblib': 'joblib',
            'xgboost': 'xgboost',
            'lightgbm': 'lightgbm'
        }
        
        package_status = {}
        failed_packages = []
        
        for package, display_name in required_packages.items():
            try:
                if package == 'sklearn':
                    import sklearn
                    version = sklearn.__version__
                elif package == 'flask':
                    import flask
                    version = flask.__version__
                elif package == 'pandas':
                    import pandas
                    version = pandas.__version__
                elif package == 'numpy':
                    import numpy
                    version = numpy.__version__
                elif package == 'joblib':
                    import joblib
                    version = joblib.__version__
                elif package == 'xgboost':
                    import xgboost
                    version = xgboost.__version__
                elif package == 'lightgbm':
                    import lightgbm
                    version = lightgbm.__version__
                
                package_status[package] = {
                    'status': 'INSTALLED',
                    'version': version
                }
                print(f"   ✅ {display_name}: {version}")
                
            except ImportError:
                package_status[package] = {
                    'status': 'MISSING',
                    'version': None
                }
                failed_packages.append(display_name)
                print(f"   ❌ {display_name}: No instalado")
        
        self.results['checks']['packages'] = package_status
        
        if failed_packages:
            error_msg = f"Librerías faltantes: {', '.join(failed_packages)}"
            self.results['errors'].append(error_msg)
            self.results['recommendations'].append(
                f"Instalar librerías: pip install {' '.join(failed_packages)}"
            )
    
    def check_file_structure(self):
        """Verifica la estructura de archivos"""
        print("\n📁 Verificando estructura de archivos...")
        
        missing_files = []
        existing_files = []
        
        for file_path in self.required_files:
            if os.path.exists(file_path):
                existing_files.append(file_path)
                print(f"   ✅ {file_path}")
            else:
                missing_files.append(file_path)
                print(f"   ❌ {file_path} - No encontrado")
        
        self.results['checks']['file_structure'] = {
            'existing_files': existing_files,
            'missing_files': missing_files,
            'status': 'PASS' if not missing_files else 'FAIL'
        }
        
        if missing_files:
            self.results['errors'].append(f"Archivos faltantes: {', '.join(missing_files)}")
    
    def check_model_files(self):
        """Verifica que los archivos del modelo existan"""
        print("\n🤖 Verificando archivos del modelo...")
        
        missing_models = []
        existing_models = []
        model_info = {}
        
        for model_file in self.required_model_files:
            if os.path.exists(model_file):
                existing_models.append(model_file)
                print(f"   ✅ {model_file}")
                
                # Intentar cargar metadata para obtener información
                if model_file.endswith('metadata.pkl'):
                    try:
                        import joblib
                        metadata = joblib.load(model_file)
                        model_info = {
                            'features_count': len(metadata.get('feature_names', [])),
                            'models_trained': len(metadata.get('metrics', {})),
                            'best_model': self.find_best_model(metadata.get('metrics', {}))
                        }
                        print(f"      📊 Características: {model_info['features_count']}")
                        print(f"      🤖 Modelos entrenados: {model_info['models_trained']}")
                        if model_info['best_model']:
                            print(f"      🏆 Mejor modelo: {model_info['best_model']}")
                    except Exception as e:
                        print(f"      ⚠️  Error al leer metadata: {str(e)}")
            else:
                missing_models.append(model_file)
                print(f"   ❌ {model_file} - No encontrado")
        
        # Buscar archivos de modelos entrenados
        models_dir = Path('models')
        if models_dir.exists():
            model_files = list(models_dir.glob('*_model.pkl'))
            if model_files:
                print(f"   📈 Modelos encontrados: {len(model_files)}")
                for model_file in model_files:
                    print(f"      • {model_file.name}")
            else:
                print("   ⚠️  No se encontraron modelos entrenados")
                self.results['warnings'].append("No hay modelos entrenados disponibles")
        
        self.results['checks']['model_files'] = {
            'existing_models': existing_models,
            'missing_models': missing_models,
            'model_info': model_info,
            'status': 'PASS' if not missing_models else 'FAIL'
        }
        
        if missing_models:
            self.results['errors'].append(f"Archivos de modelo faltantes: {', '.join(missing_models)}")
            self.results['recommendations'].append("Ejecutar train_models.py para generar los modelos")
    
    def find_best_model(self, metrics):
        """Encuentra el mejor modelo basado en R²"""
        if not metrics:
            return None
        
        best_model = None
        best_r2 = -1
        
        for model_name, model_metrics in metrics.items():
            r2 = model_metrics.get('r2_test', 0)
            if r2 > best_r2:
                best_r2 = r2
                best_model = f"{model_name} (R²: {r2:.4f})"
        
        return best_model
    
    def check_flask_app(self):
        """Verifica que la aplicación Flask esté correctamente configurada"""
        print("\n🌐 Verificando aplicación Flask...")
        
        if not os.path.exists('app.py'):
            self.results['checks']['flask_app'] = {
                'status': 'FAIL',
                'message': 'app.py no encontrado'
            }
            print("   ❌ app.py no encontrado")
            return
        
        try:
            # Intentar importar la app
            spec = importlib.util.spec_from_file_location("app", "app.py")
            app_module = importlib.util.module_from_spec(spec)
            
            # Verificar contenido básico del archivo
            with open('app.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            flask_checks = {
                'has_flask_import': 'from flask import' in content or 'import flask' in content,
                'has_app_creation': 'Flask(__name__)' in content,
                'has_routes': '@app.route' in content,
                'has_main_block': "if __name__ == '__main__'" in content
            }
            
            all_passed = all(flask_checks.values())
            
            self.results['checks']['flask_app'] = {
                'status': 'PASS' if all_passed else 'PARTIAL',
                'checks': flask_checks,
                'message': 'Flask app configurada correctamente' if all_passed else 'Flask app parcialmente configurada'
            }
            
            for check, passed in flask_checks.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {check.replace('_', ' ').title()}")
            
            if not all_passed:
                self.results['warnings'].append("La aplicación Flask puede no estar completamente configurada")
        
        except Exception as e:
            self.results['checks']['flask_app'] = {
                'status': 'FAIL',
                'message': f'Error al verificar Flask: {str(e)}'
            }
            print(f"   ❌ Error al verificar Flask: {str(e)}")
    
    def check_templates(self):
        """Verifica los templates HTML"""
        print("\n📄 Verificando templates HTML...")
        
        template_files = ['templates/index.html', 'templates/prediccion.html']
        template_status = {}
        
        for template in template_files:
            if os.path.exists(template):
                try:
                    with open(template, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Verificaciones básicas de HTML
                    has_html_structure = all([
                        '<html' in content.lower(),
                        '<head>' in content.lower(),
                        '<body>' in content.lower()
                    ])
                    
                    template_status[template] = {
                        'exists': True,
                        'valid_structure': has_html_structure,
                        'size': len(content)
                    }
                    
                    status = "✅" if has_html_structure else "⚠️"
                    print(f"   {status} {template} ({len(content)} chars)")
                
                except Exception as e:
                    template_status[template] = {
                        'exists': True,
                        'valid_structure': False,
                        'error': str(e)
                    }
                    print(f"   ❌ {template} - Error al leer: {str(e)}")
            else:
                template_status[template] = {'exists': False}
                print(f"   ❌ {template} - No encontrado")
        
        self.results['checks']['templates'] = template_status
    
    def check_static_files(self):
        """Verifica archivos estáticos (CSS, JS)"""
        print("\n🎨 Verificando archivos estáticos...")
        
        static_files = {
            'static/css/style.css': 'CSS',
            'static/js/script.js': 'JavaScript'
        }
        
        static_status = {}
        
        for file_path, file_type in static_files.items():
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    static_status[file_path] = {
                        'exists': True,
                        'size': len(content),
                        'type': file_type
                    }
                    
                    print(f"   ✅ {file_path} ({len(content)} chars)")
                
                except Exception as e:
                    static_status[file_path] = {
                        'exists': True,
                        'error': str(e)
                    }
                    print(f"   ❌ {file_path} - Error: {str(e)}")
            else:
                static_status[file_path] = {'exists': False}
                print(f"   ❌ {file_path} - No encontrado")
        
        self.results['checks']['static_files'] = static_status
    
    def test_model_loading(self):
        """Prueba cargar los modelos principales"""
        print("\n🧪 Probando carga de modelos...")
        
        try:
            import joblib
            
            # Probar cargar archivos críticos
            critical_files = {
                'metadata': 'models/metadata.pkl',
                'scaler': 'models/scaler.pkl',
                'feature_selector': 'models/feature_selector.pkl'
            }
            
            loaded_components = {}
            
            for component, file_path in critical_files.items():
                if os.path.exists(file_path):
                    try:
                        loaded = joblib.load(file_path)
                        loaded_components[component] = {
                            'status': 'SUCCESS',
                            'type': type(loaded).__name__
                        }
                        print(f"   ✅ {component}: {type(loaded).__name__}")
                    except Exception as e:
                        loaded_components[component] = {
                            'status': 'ERROR',
                            'error': str(e)
                        }
                        print(f"   ❌ {component}: Error - {str(e)}")
                else:
                    loaded_components[component] = {'status': 'NOT_FOUND'}
                    print(f"   ❌ {component}: Archivo no encontrado")
            
            # Intentar cargar un modelo entrenado
            models_dir = Path('models')
            if models_dir.exists():
                model_files = list(models_dir.glob('*_model.pkl'))
                if model_files:
                    try:
                        test_model = joblib.load(model_files[0])
                        loaded_components['sample_model'] = {
                            'status': 'SUCCESS',
                            'type': type(test_model).__name__,
                            'file': str(model_files[0])
                        }
                        print(f"   ✅ Modelo de prueba: {type(test_model).__name__}")
                    except Exception as e:
                        loaded_components['sample_model'] = {
                            'status': 'ERROR',
                            'error': str(e)
                        }
                        print(f"   ❌ Error al cargar modelo: {str(e)}")
            
            self.results['checks']['model_loading'] = loaded_components
        
        except ImportError:
            print("   ❌ joblib no está disponible")
            self.results['checks']['model_loading'] = {'status': 'JOBLIB_MISSING'}
        except Exception as e:
            print(f"   ❌ Error general: {str(e)}")
            self.results['checks']['model_loading'] = {'status': 'ERROR', 'error': str(e)}
    
    def check_requirements_file(self):
        """Verifica el archivo requirements.txt"""
        print("\n📋 Verificando requirements.txt...")
        
        if not os.path.exists('requirements.txt'):
            print("   ❌ requirements.txt no encontrado")
            self.results['checks']['requirements'] = {'status': 'MISSING'}
            self.results['warnings'].append("requirements.txt faltante")
            return
        
        try:
            with open('requirements.txt', 'r', encoding='utf-8') as f:
                requirements = f.read().strip().split('\n')
            
            requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
            
            essential_packages = ['flask', 'pandas', 'numpy', 'scikit-learn', 'joblib']
            found_packages = []
            missing_packages = []
            
            for package in essential_packages:
                found = any(package.lower() in req.lower() for req in requirements)
                if found:
                    found_packages.append(package)
                else:
                    missing_packages.append(package)
            
            self.results['checks']['requirements'] = {
                'status': 'PASS' if not missing_packages else 'PARTIAL',
                'total_requirements': len(requirements),
                'found_packages': found_packages,
                'missing_packages': missing_packages
            }
            
            print(f"   📦 Total de requirements: {len(requirements)}")
            print(f"   ✅ Paquetes esenciales encontrados: {len(found_packages)}")
            
            if missing_packages:
                print(f"   ⚠️  Paquetes esenciales faltantes: {', '.join(missing_packages)}")
                self.results['warnings'].append(f"Requirements incompletos: {', '.join(missing_packages)}")
        
        except Exception as e:
            print(f"   ❌ Error al leer requirements.txt: {str(e)}")
            self.results['checks']['requirements'] = {'status': 'ERROR', 'error': str(e)}
    
    def perform_integration_test(self):
        """Realiza una prueba de integración básica"""
        print("\n🔬 Realizando prueba de integración...")
        
        try:
            # Verificar que podemos importar y usar los componentes básicos
            import pandas as pd
            import numpy as np
            
            # Crear datos de prueba
            test_data = {
                'crim': [0.1],
                'zn': [18.0],
                'indus': [2.31],
                'chas': [0],
                'nox': [0.538],
                'rm': [6.575],
                'age': [65.2],
                'dis': [4.09],
                'rad': [1],
                'tax': [296],
                'ptratio': [15.3],
                'b': [396.9],
                'lstat': [4.98]
            }
            
            df_test = pd.DataFrame(test_data)
            print("   ✅ Creación de DataFrame de prueba")
            
            # Verificar operaciones básicas
            df_test['test_feature'] = df_test['rm'] * df_test['lstat']
            print("   ✅ Operaciones con DataFrame")
            
            # Si existe metadata, verificar compatibilidad
            if os.path.exists('models/metadata.pkl'):
                import joblib
                metadata = joblib.load('models/metadata.pkl')
                
                if 'feature_names' in metadata:
                    print(f"   📊 Características esperadas: {len(metadata['feature_names'])}")
                
                print("   ✅ Carga de metadata")
            
            self.results['checks']['integration_test'] = {
                'status': 'PASS',
                'message': 'Prueba de integración exitosa'
            }
            print("   🎉 Prueba de integración EXITOSA")
        
        except Exception as e:
            self.results['checks']['integration_test'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"   ❌ Prueba de integración FALLÓ: {str(e)}")
            self.results['errors'].append(f"Prueba de integración falló: {str(e)}")
    
    def generate_summary(self):
        """Genera resumen final de la verificación"""
        print("\n" + "="*60)
        print("📊 RESUMEN DE VERIFICACIÓN")
        print("="*60)
        
        # Contar estados
        total_checks = len(self.results['checks'])
        passed_checks = sum(1 for check in self.results['checks'].values() 
                           if isinstance(check, dict) and check.get('status') == 'PASS')
        failed_checks = sum(1 for check in self.results['checks'].values() 
                           if isinstance(check, dict) and check.get('status') == 'FAIL')
        
        # Determinar estado general
        if len(self.results['errors']) == 0:
            if len(self.results['warnings']) == 0:
                self.results['overall_status'] = 'READY'
                status_emoji = "🟢"
                status_msg = "LISTO PARA DESPLIEGUE"
            else:
                self.results['overall_status'] = 'WARNING'
                status_emoji = "🟡"
                status_msg = "LISTO CON ADVERTENCIAS"
        else:
            self.results['overall_status'] = 'NOT_READY'
            status_emoji = "🔴"
            status_msg = "NO LISTO PARA DESPLIEGUE"
        
        print(f"\n{status_emoji} ESTADO GENERAL: {status_msg}")
        print(f"📈 Verificaciones exitosas: {passed_checks}/{total_checks}")
        print(f"❌ Errores críticos: {len(self.results['errors'])}")
        print(f"⚠️  Advertencias: {len(self.results['warnings'])}")
        
        # Mostrar errores críticos
        if self.results['errors']:
            print(f"\n🚨 ERRORES CRÍTICOS:")
            for i, error in enumerate(self.results['errors'], 1):
                print(f"   {i}. {error}")
        
        # Mostrar advertencias
        if self.results['warnings']:
            print(f"\n⚠️  ADVERTENCIAS:")
            for i, warning in enumerate(self.results['warnings'], 1):
                print(f"   {i}. {warning}")
        
        # Mostrar recomendaciones
        if self.results['recommendations']:
            print(f"\n💡 RECOMENDACIONES:")
            for i, rec in enumerate(self.results['recommendations'], 1):
                print(f"   {i}. {rec}")
        
        return self.results['overall_status'] == 'READY'
    
    def save_report(self):
        """Guarda el reporte en un archivo JSON"""
        try:
            report_file = f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Reporte guardado en: {report_file}")
            return report_file
        except Exception as e:
            print(f"\n❌ Error al guardar reporte: {str(e)}")
            return None
    
    def run_all_checks(self):
        """Ejecuta todas las verificaciones"""
        self.print_header()
        
        try:
            self.check_python_version()
            self.check_required_packages()
            self.check_file_structure()
            self.check_model_files()
            self.check_flask_app()
            self.check_templates()
            self.check_static_files()
            self.test_model_loading()
            self.check_requirements_file()
            self.perform_integration_test()
            
            is_ready = self.generate_summary()
            self.save_report()
            
            print(f"\n⏰ Verificación completada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*60)
            
            return is_ready
            
        except Exception as e:
            print(f"\n💥 ERROR CRÍTICO EN VERIFICACIÓN: {str(e)}")
            print(traceback.format_exc())
            return False

def main():
    """Función principal"""
    try:
        verifier = DeploymentVerifier()
        is_ready = verifier.run_all_checks()
        
        # Código de salida para CI/CD
        sys.exit(0 if is_ready else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Verificación interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 ERROR FATAL: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()