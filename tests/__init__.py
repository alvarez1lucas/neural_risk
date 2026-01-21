import pandas as pd
import numpy as np
from neural_risk.data.data_processor import DataProcessor


# 1. Crear datos sintéticos "sucios" (Precio con tendencia + Texto + Nulos)
fechas = pd.date_range(start='2023-01-01', periods=500)
precios = np.cumsum(np.random.randn(500)) + 100  # Random walk (No estacionario)
df = pd.DataFrame({'Close': precios}, index=fechas)

# Ensuciar datos para probar el cleaner
df.iloc[10:15, 0] = np.nan       # Hueco de precios
df['Volume'] = np.nan            # Columna vacía
df['Bad_Data'] = "1,200.50"      # String con coma

# 2. Instanciar Cleaner
cleaner = DataProcessor()

# 3. Probar Auto-Clean
print("--- Limpiando Datos ---")
df_clean = cleaner.auto_clean(df)
print(df_clean.head())
print(f"Nulos restantes: {df_clean.isnull().sum().sum()}")

# 4. Probar Stationarity (Debería fallar en precios puros)
print("\n--- Test Stationarity (Precio Original) ---")
stat_report = cleaner.check_stationarity(df_clean['Close'])
print(f"Es estacionario? {stat_report['is_stationary']} (p-value: {stat_report['p_value']})")
print(f"Recomendación: {stat_report['recommendation']}")

# 5. Aplicar FracDiff
print("\n--- Aplicando Fractional Differentiation (López de Prado) ---")
# Bajamos un poco el thres para que no consuma tantos datos de la serie corta
df_frac = cleaner.fractional_diff(df_clean['Close'], d=0.4, thres=1e-4)

# 6. Probar Stationarity con Debugging
stat_report_frac = cleaner.check_stationarity(df_frac)

if 'error' in stat_report_frac:
    print(f"Error en el test: {stat_report_frac['error']}")
    print(f"Longitud de la serie tras FracDiff: {len(df_frac)}")
else:
    print(f"Es estacionario (FracDiff)? {stat_report_frac['is_stationary']} (p-value: {stat_report_frac['p_value']})")