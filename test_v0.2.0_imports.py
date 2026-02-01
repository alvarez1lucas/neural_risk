#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_v0.2.0_imports.py
Validación rápida de que todos los nuevos módulos importan correctamente
"""

import sys
import traceback

def test_imports():
    """Test all new imports from v0.2.0"""
    
    tests = [
        # Core
        ("neural_risk", "Core package"),
        ("neural_risk.engine", "AutomatedRiskEngine"),
        
        # Clásicos
        ("neural_risk.models.risk_model", "NeuralRiskModel"),
        ("neural_risk.models.ensemble_trainer", "EnsembleTrainer"),
        
        # Nuevos 5 expertos
        ("neural_risk.models.garch_volatility", "GARCH/EGARCH Expert"),
        ("neural_risk.models.lstm_transformer", "LSTM/Transformer Expert"),
        ("neural_risk.models.reinforcement_learning", "RL Expert"),
        ("neural_risk.models.copula_expert", "Copula Expert"),
        ("neural_risk.models.anomaly_detection", "Anomaly Detection Expert"),
    ]
    
    print("\n" + "="*70)
    print("🧪 VALIDACIÓN DE IMPORTS v0.2.0")
    print("="*70 + "\n")
    
    passed = 0
    failed = 0
    
    for module_name, description in tests:
        try:
            __import__(module_name)
            print(f"✅ {description:40s} [{module_name}]")
            passed += 1
        except Exception as e:
            print(f"❌ {description:40s} [{module_name}]")
            print(f"   Error: {str(e)[:80]}")
            failed += 1
    
    print("\n" + "-"*70)
    print(f"Resultados: {passed} ✅ | {failed} ❌")
    print("-"*70 + "\n")
    
    if failed == 0:
        print("🚀 TODOS LOS IMPORTS OK - v0.2.0 LISTO\n")
        return True
    else:
        print(f"⚠️  {failed} módulos fallaron - revisar instalación\n")
        return False


def test_basic_instantiation():
    """Test that we can instantiate basic objects"""
    
    print("="*70)
    print("🔧 TEST DE INSTANCIACIÓN BÁSICA")
    print("="*70 + "\n")
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        print("Testing GARCH Expert...")
        from neural_risk.models.garch_volatility import GARCHVolatilityExpert
        garch = GARCHVolatilityExpert('egarch')
        print("   ✅ GARCHVolatilityExpert OK\n")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        tests_failed += 1
    
    try:
        print("Testing LSTM/Transformer Expert...")
        from neural_risk.models.lstm_transformer import SequentialForecastingEnsemble
        lstm = SequentialForecastingEnsemble(input_size=10)
        print("   ✅ SequentialForecastingEnsemble OK\n")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        tests_failed += 1
    
    try:
        print("Testing Anomaly Detector...")
        from neural_risk.models.anomaly_detection import AnomalyDetector
        ad = AnomalyDetector()
        print("   ✅ AnomalyDetector OK\n")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        tests_failed += 1
    
    try:
        print("Testing RL Expert...")
        from neural_risk.models.reinforcement_learning import RLAllocationExpert
        rl = RLAllocationExpert(observation_space_size=20)
        print("   ✅ RLAllocationExpert OK\n")
        tests_passed += 1
    except ImportError:
        print("   ⚠️  stable_baselines3 not installed (optional)\n")
        tests_passed += 1  # No contar como fallo
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        tests_failed += 1
    
    try:
        print("Testing Copula Expert...")
        from neural_risk.models.copula_expert import CopulaExpert
        cop = CopulaExpert()
        print("   ✅ CopulaExpert OK\n")
        tests_passed += 1
    except ImportError:
        print("   ⚠️  copulae not installed (optional)\n")
        tests_passed += 1  # No contar como fallo
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
        tests_failed += 1
    
    print("-"*70)
    print(f"Instanciación: {tests_passed} ✅ | {tests_failed} ❌\n")
    
    return tests_failed == 0


def main():
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + " NEURAL RISK ENGINE v0.2.0 - TEST SUITE ".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    imports_ok = test_imports()
    
    instantiation_ok = test_basic_instantiation()
    
    print("="*70)
    print("📊 RESUMEN FINAL")
    print("="*70)
    
    if imports_ok and instantiation_ok:
        print("\n✅ TODOS LOS TESTS PASARON - v0.2.0 ESTÁ LISTO\n")
        print("Próximos pasos:")
        print("1. pip install arch xgboost torch scipy scikit-learn statsmodels")
        print("2. python run_engine.py  (o tu script de backtesting)")
        print()
        return 0
    else:
        print("\n⚠️  ALGUNOS TESTS FALLARON - REVISAR ARRIBA\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
