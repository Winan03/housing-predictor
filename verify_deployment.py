#!/usr/bin/env python3
"""
Script para verificar que todos los componentes estén listos antes del despliegue
"""
import os
import sys
import importlib.util

def check_file_exists(filepath, description):
    """Verificar que un archivo existe"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} NO ENCONTRADO: {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """Verificar que un directorio existe"""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ {description} NO ENCONTRADO: {dirpath}")
        return False

def check_models_directory():
    """Verificar que existan todos los archivos de modelos"""
    models_dir = 'models'
    if not check_directory_exists(models_dir, "Directorio de modelos"):
        return False
    
    required_files = [
        'random_forest_model.pkl',
        'regresion_lineal_model.pkl', 
        'xgboost_model.pkl',
        'scaler.pkl',
        'power_transformer.pkl',
        'metadata.pkl'
    ]
    
    all_exist = True
    for file in required_files:
        filepath = os.path.join(models_dir, file)
        if not check_file_exists(filepath, f"Modelo {file}"):
            all_exist = False
    
    return all_exist

def check_templates():
    """Verificar que existan los templates necesarios"""
    templates_dir = 'templates'
    if not check_directory_exists(templates_dir, "Directorio de templates"):
        return False
    
    required_templates = ['index.html', 'error.html']
    all_exist = True
    
    for template in required_templates:
        filepath = os.path.join(templates_dir, template)
        if not check_file_exists(filepath, f"Template {template}"):
            all_exist = False
    
    return all_exist

def check_static_files():
    """Verificar archivos estáticos"""
    static_dir = 'static'
    if not check_directory_exists(static_dir, "Directorio static"):
        return False
    
    css_dir = os.path.join(static_dir, 'css')
    js_dir = os.path.join(static_dir, 'js')
    
    all_exist = True
    if not check_directory_exists(css_dir, "Directorio CSS"):
        all_exist = False
    if not check_directory_exists(js_dir, "Directorio JS"):
        all_exist = False
    
    if not check_file_exists(os.path.join(css_dir, 'style.css'), "Archivo CSS"):
        all_exist = False
    if not check_file_exists(os.path.join(js_dir, 'script.js'), "Archivo JS"):
        all_exist = False
    
    return all_exist

def check_python_files():
    """Verificar archivos Python principales"""
    required_files = ['app.py', 'model.py', 'requirements.txt', 'Procfile']
    all_exist = True
    
    for file in required_files:
        if not check_file_exists(file, f"Archivo {file}"):
            all_exist = False
    
    return all_exist

def check_imports():
    """Verificar que se puedan importar los módulos principales"""
    try:
        print("\n🔍 Verificando imports...")
        
        # Verificar Flask
        import flask
        print("✅ Flask importado correctamente")
        
        # Verificar pandas
        import pandas
        print("✅ Pandas importado correctamente")
        
        # Verificar numpy
        import numpy
        print("✅ Numpy importado correctamente")
        
        # Verificar scikit-learn
        import sklearn
        print("✅ Scikit-learn importado correctamente")
        
        # Verificar matplotlib
        import matplotlib
        print("✅ Matplotlib importado correctamente")
        
        # Verificar xgboost
        import xgboost
        print("✅ XGBoost importado correctamente")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        return False

def main():
    """Función principal de verificación"""
    print("🚀 Verificando configuración para despliegue...\n")
    
    checks = [
        ("Archivos Python principales", check_python_files),
        ("Directorio de modelos", check_models_directory),
        ("Templates", check_templates),
        ("Archivos estáticos", check_static_files),
        ("Imports de Python", check_imports)
    ]
    
    all_passed = True
    results = []
    
    for check_name, check_func in checks:
        print(f"\n📋 Verificando: {check_name}")
        print("-" * 50)
        
        result = check_func()
        results.append((check_name, result))
        
        if not result:
            all_passed = False
    
    # Resumen final
    print("\n" + "="*60)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("="*60)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ¡TODAS LAS VERIFICACIONES PASARON!")
        print("✅ La aplicación está lista para el despliegue")
        sys.exit(0)
    else:
        print("❌ ALGUNAS VERIFICACIONES FALLARON")
        print("🔧 Por favor, corrija los errores antes del despliegue")
        sys.exit(1)

if __name__ == "__main__":
    main()