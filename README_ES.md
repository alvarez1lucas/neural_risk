# 🤖 Neural Risk Engine v0.2.0

**Sistema automatizado de gestión de riesgo y optimización de portafolio para criptomonedas**

Arquitectura lista para producción con servicios independientes para procesamiento de datos, generación de señales, ejecución de órdenes y monitorización.

---

## 🎯 ¿Qué hace este sistema?

- Entrada: datos de mercado en tiempo real (Binance)
- Proceso: 9 expertos en conjunto + optimización de portafolio
- Salida: señales de trading con niveles de entrada/salida cada 5 minutos
- Despliegue: 4 servicios independientes funcionando 24/7

---

## 🏗️ Resumen de la arquitectura

1. Data Fetcher: obtiene y guarda velas cada 5 minutos en SQLite.
2. Entrenamiento: trabajo offline diario (00:00 UTC) que guarda modelos en cache.
3. Engine: carga modelos y ejecuta 9 expertos en paralelo, combina señales con PortfolioAgent (Criterio de Kelly).
4. Executor: valida tamaños de posición, límites de pérdida y envía órdenes a Binance (o modo MOCK).
5. Dashboard: monitor opcional con métricas en tiempo real.

---

## ⚡ Inicio rápido

1. Instalar dependencias:
```bash
pip install -r requirements.txt
```

2. Validar sistema:
```bash
python scripts/pre_deploy_check.py
# Salida esperada: [OK] PRE-DEPLOY CHECK PASSED - READY TO DEPLOY!
```

3. Ejecutar servicios (abrir 3-4 terminales):
```bash
python scripts/run_data_fetcher.py
python scripts/run_engine.py
python scripts/run_executor.py
jupyter notebook notebooks/live_dashboard.ipynb  # opcional
```

---

## ⚙️ Configuración principal

Editar `config/config.yaml` según necesidades (modo, activos, tiempos de ciclo, límites de riesgo).

Parámetros clave:
- `cycle_timing.data_fetcher`: 300 (5 min)
- `cycle_timing.engine`: 300 (5 min)
- `risk.max_position_size`: 0.10 (10%)
- `risk.daily_loss_limit`: -0.05 (-5%)

---

## 🛠️ Qué monitorear

- `logs/neural_risk.log` — actividad y errores
- `data/neural_risk.db` — historial y señales
- `data/trained_models.pkl` — cache de modelos
- Dashboard — PnL y métricas de rendimiento

---

## 📌 Notas importantes

- El sistema arranca en modo MOCK si no configuras las claves de Binance.
- Empieza con 2 activos (BTC, ETH) y escala gradualmente.
- Las decisiones se generan cada 5 minutos; el entrenamiento es diario por defecto.

---

## 📚 Documentación adicional

Revisa los archivos en la raíz para guías detalladas: `DEPLOY_FINAL_GUIDE.md`, `PRE_DEPLOY_CHECKLIST.md`, `VALIDACIÓN_TIMING_E_INTERVALOS.md`.

---

**Estado:** ✅ Producción lista | **Versión:** 0.2.0 | **Fecha:** Feb 1, 2026
