import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import networkx as nx
from hmmlearn import hmm



class AdvancedVolatilityFeatures:
    

    @staticmethod
    def asymmetric_decay_vol(series: pd.Series, window: int = 21, decay: float = 0.94, penalty: float = 1.5) -> pd.Series:
        """
        1. Volatilidad Realizada con Ponderación Asimétrica y Decay Exponencial.
        Pesa más los retornos negativos para capturar el 'leverage effect'.
        """
        returns = np.log(series / series.shift(1))
        
        # Aplicamos penalización a retornos negativos (Downside Risk)
        weighted_returns = returns.copy()
        weighted_returns[returns < 0] = returns[returns < 0] * penalty
        
        # Cálculo de varianza con decaimiento exponencial (tipo EWMA)
        # alpha = 1 - decay
        vol = weighted_returns.pow(2).ewm(alpha=1-decay, min_periods=window).mean()
        
        return np.sqrt(vol) * np.sqrt(252) # Anualizada

    @staticmethod
    def gap_corrected_parkinson(df: pd.DataFrame, prefix: str = "", window: int = 14, gap_mult: float = 1.0) -> pd.Series:
        """
        2. Parkinson Volatility con corrección por Gaps Overnight.
        Ajusta el rango High-Low sumando el salto entre el cierre anterior y la apertura actual.
        """
        h = df[f'{prefix}High']
        l = df[f'{prefix}Low']
        o = df[f'{prefix}Open']
        c_prev = df[f'{prefix}Close'].shift(1)
        
        # Estimador Parkinson puro
        parkinson_core = (1 / (4 * np.log(2))) * np.log(h/l)**2
        
        # Corrección por Gap (Overnight risk)
        gap_risk = np.log(o / c_prev)**2
        
        # Combinación y suavizado con mediana para filtrar outliers
        total_var = parkinson_core + (gap_mult * gap_risk)
        vol = total_var.rolling(window=window).median() 
        
        return np.sqrt(vol * 252)

    @staticmethod
    def volume_adjusted_kurtosis(df: pd.DataFrame, prefix: str = "", window: int = 40) -> pd.Series:
        """
        3. Kurtosis Ajustada por Volumen.
        Mide la probabilidad de eventos de cola (Fat Tails) normalizada por la liquidez.
        """
        returns = np.log(df[f'{prefix}Close'] / df[f'{prefix}Close'].shift(1))
        volume = df[f'{prefix}Volume']
        
        # Kurtosis rodante
        rolling_kurt = returns.rolling(window=window).kurt()
        
        # Normalización por Log-Volumen (para mitigar escala de activos muy líquidos)
        norm_vol = np.log(volume.rolling(window=window).mean())
        
        # Feature: Kurtosis / Log-Vol. 
        # Interpretación: Alta kurtosis con bajo volumen = ruido. Alta kurtosis con alto volumen = capitulación/crash real.
        return rolling_kurt / norm_vol
    


class InformationalDynamicsFeatures:
    """
    Features basadas en momentum, geometria fractal y teoria de la informacion.
    """

    @staticmethod
    def cross_correlation_momentum(series: pd.Series, benchmark: pd.Series, alpha: float = 0.05, window: int = 20) -> pd.Series:
        # ROC Exponencial
        exp_roc = series.pct_change().ewm(alpha=alpha).mean()
        # Correlacion rodante
        rolling_corr = series.pct_change().rolling(window=window).corr(benchmark.pct_change())
        # Momentum ajustado
        return exp_roc * (1 - rolling_corr.abs())

    @staticmethod
    def adaptive_hurst_exponent(series: pd.Series, min_window: int = 10, max_window: int = 100) -> pd.Series:
        def get_hurst(x):
            if len(x) < max_window: return 0.5
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        return series.rolling(window=max_window).apply(get_hurst)

    @staticmethod
    def price_entropy(series: pd.Series, bins: int = 10, window: int = 40) -> pd.Series:
        def calculate_shannon(x):
            try:
                counts = np.histogram(x, bins=bins)[0]
                probs = counts / len(x)
                probs = probs[probs > 0] # Evita log(0)
                return -np.sum(probs * np.log(probs))
            except:
                return 0
        returns = series.pct_change().dropna()
        return returns.rolling(window=window).apply(calculate_shannon)
    
class SyntheticOptionsFeatures:
    """
    Features de sensibilidad (Greeks) y Payoffs para activos sintéticos
    basadas en aproximaciones numéricas y estructura de volatilidad.
    """

    @staticmethod
    def synthetic_delta(series: pd.Series, vol_series: pd.Series, h: float = 0.01) -> pd.Series:
        """
        1. Delta Sintético Ajustado por Volatilidad.
        Usa diferencias finitas para estimar la sensibilidad. 
        Delta = [f(S + h) - f(S - h)] / 2h, ajustado por el régimen de volatilidad.
        """
        # Simulamos la sensibilidad del precio respecto a sí mismo ajustado por su vol
        # En una opción real, f(S) sería el precio de la opción. 
        # Aquí estimamos la sensibilidad de la tendencia actual.
        delta_approx = (series.shift(-1) - series.shift(1)) / (2 * h * series)
        
        # Ajuste por Volatilidad: a mayor vol, el delta de las opciones ATM suele ser más inestable
        return delta_approx * (1 / (1 + vol_series))

    @staticmethod
    def vega_skew_adjusted(vol_series: pd.Series, kurtosis_series: pd.Series) -> pd.Series:
        """
        2. Vega con Ajuste de Skew/Smile.
        La Vega (sensibilidad a la vol) no es constante. Si la Kurtosis (Fat Tails)
        es alta, la sensibilidad a cambios en volatilidad aumenta en los extremos.
        """
        # Vega base simplificada
        base_vega = vol_series * 0.5 
        
        # El ajuste de Skew lo derivamos de la Kurtosis (proxi de la 'sonrisa' de vol)
        # Si hay colas gordas, la Vega en los sintéticos es más explosiva.
        skew_adj = np.log(1 + np.abs(kurtosis_series))
        
        return base_vega * (1 + skew_adj)

    @staticmethod
    def payoff_profile_vector(series: pd.Series, window: int = 20, n_scenarios: int = 10) -> pd.Series:
        """
        3. Payoff Profile (Feature Vector).
        Genera el valor esperado de un 'Straddle Sintético' basado en la 
        dispersión histórica reciente.
        """
        def calculate_expected_payoff(x):
            if len(x) < window: return 0
            # Simulamos escenarios basados en la desviación estándar actual
            std = np.std(x)
            current_price = x[-1]
            
            # Escenarios de precio final
            up_scenario = current_price + (std * 1.5)
            down_scenario = current_price - (std * 1.5)
            
            # Payoff de un Straddle: |S_T - K|
            payoff = (np.abs(up_scenario - current_price) + np.abs(down_scenario - current_price)) / 2
            return payoff

        return series.rolling(window=window).apply(calculate_expected_payoff)

class MLHybridFeatures:
    """
    Genera features de avanzada utilizando sub-modelos de ML/DL.
    Captura anomalías, regímenes y estados comprimidos.
    """

    @staticmethod
    def anomaly_score_isolation_forest(df: pd.DataFrame, prefix: str = "", window: int = 100) -> pd.Series:
        """
        1. Anomaly Score de Isolation Forest.
        Detecta comportamientos atípicos en la relación Precio/Volumen/Volatilidad.
        """
        # Seleccionamos features base para detectar anomalías
        cols = [f'{prefix}Close', f'{prefix}Volume', f'{prefix}parkinson_gap']
        data = df[cols].fillna(0)
        
        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        # El score de anomalía es el output (menor score = más anómalo)
        scores = iso.fit_predict(data)
        return pd.Series(scores, index=df.index)

    @staticmethod
    def market_regime_clustering(df: pd.DataFrame, prefix: str = "", n_clusters: int = 3) -> pd.Series:
        """
        2. Clustering para Regímenes de Mercado.
        Asigna cada día a un cluster (ej. 0: Bull, 1: Bear, 2: Volátil).
        """
        # Features para clusterizar: Momentum y Volatilidad
        features = df[[f'{prefix}asym_vol', f'{prefix}hurst']].fillna(0)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(scaled_features)
        return pd.Series(clusters, index=df.index)

    @staticmethod
    def autoencoder_bottleneck(df: pd.DataFrame, prefix: str = "", bottleneck_dim: int = 4):
        """
        3. Bottleneck de Autoencoder (Estado Comprimido).
        Usa una red neuronal para comprimir la info OHLCV en un vector latente.
        """
        import torch
        import torch.nn as nn
        cols = [f'{prefix}Open', f'{prefix}High', f'{prefix}Low', f'{prefix}Close', f'{prefix}Volume']
        data = torch.FloatTensor(df[cols].values)
        
        # Arquitectura Simple del Autoencoder
        class SimpleAE(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 8),
                    nn.ReLU(),
                    nn.Linear(8, latent_dim)
                )
            def forward(self, x):
                return self.encoder(x)

        model = SimpleAE(input_dim=len(cols), latent_dim=bottleneck_dim)
        with torch.no_grad():
            latent_features = model(data).numpy()
        
        # Devolvemos un DataFrame con las dimensiones comprimidas
        latent_cols = [f'{prefix}ae_dim_{i}' for i in range(bottleneck_dim)]
        return pd.DataFrame(latent_features, index=df.index, columns=latent_cols)
    

class LiquidityMicrostructureFeatures:
    @staticmethod
    def normalized_bid_ask_spread(df: pd.DataFrame, prefix: str = "", vol_window: int = 20) -> pd.Series:
        """
        Calcula el spread usando el estimador de Corwin-Schultz (High-Low) 
        si no existen columnas de Bid/Ask.
        """
        if f'{prefix}Ask' in df.columns and f'{prefix}Bid' in df.columns:
            spread_rel = (df[f'{prefix}Ask'] - df[f'{prefix}Bid']) / ((df[f'{prefix}Ask'] + df[f'{prefix}Bid']) / 2)
        else:
            # Proxy: High-Low Range como proxi de liquidez
            # Un rango amplio relativo al precio suele indicar spreads más abiertos
            spread_rel = (df[f'{prefix}High'] - df[f'{prefix}Low']) / df[f'{prefix}Close']
        
        vol = df[f'{prefix}Close'].pct_change().rolling(window=vol_window).std()
        return (spread_rel / (vol + 1e-9)).fillna(0)

    @staticmethod
    def order_book_imbalance(df: pd.DataFrame, prefix: str = "") -> pd.Series:
        """
        Si no hay L2, usa el 'Intraday Intensity Index'.
        (2*Close - High - Low) / (High - Low) * Volume
        """
        h, l, c = df[f'{prefix}High'], df[f'{prefix}Low'], df[f'{prefix}Close']
        v = df[f'{prefix}Volume']
        
        # Este indicador muestra dónde cerró el precio respecto a su rango:
        # Si cierra cerca del High con mucho volumen, el imbalance es comprador.
        ii_index = ((2 * c - h - l) / (h - l + 1e-9)) * v
        return ii_index.rolling(window=20).mean()

    @staticmethod
    def liquidity_resilience_score(df: pd.DataFrame, prefix: str = "", window: int = 10) -> pd.Series:
        """
        3. Resiliencia de Liquidez Post-Trade.
        Mide la velocidad de recuperación del spread tras picos de volumen.
        """
        # Identificamos 'picos de volumen' (trades grandes)
        vol_mean = df[f'{prefix}Volume'].rolling(window=50).mean()
        high_vol_event = (df[f'{prefix}Volume'] > vol_mean * 2).astype(int)
        
        # Intentamos usar el spread normalizado que ya calculamos
        # El pipeline lo llamó '{prefix}norm_spread'
        target_col = f'{prefix}norm_spread'
        if target_col in df.columns:
            spread = df[target_col]
        else:
            # Si no existe, creamos una Serie de Pandas (no un array de numpy)
            spread = pd.Series(np.random.rand(len(df)) * 0.001, index=df.index)
        
        # Delta de recuperación: (Spread_t - Spread_t-n)
        # Ahora .diff() funcionará porque 'spread' es una Serie de Pandas
        recovery = spread.diff(window).fillna(0)
        resilience = (high_vol_event * -recovery) 
        
        return resilience.rolling(window=window).mean()
class OrderFlowFeatures:
    """
    Features de flujo de órdenes (Order Flow) para detectar presión,
    zonas de valor y desbalances agresivos.
    """

    @staticmethod
    def aggressive_cumulative_delta(df: pd.DataFrame, prefix: str = "", window: int = 50) -> pd.Series:
        """
        Determina la agresividad basada en el cierre de la vela (Tick Test).
        """
        # Close > Open = Presión Compradora (+1), else (-1)
        side = np.where(df[f'{prefix}Close'] >= df[f'{prefix}Open'], 1, -1)
        delta = side * df[f'{prefix}Volume']    
        
        return delta.rolling(window=window).sum()

    @staticmethod
    def dynamic_volume_profile_stats(df: pd.DataFrame, prefix: str = "", bins: int = 30, window: int = 60) -> pd.Series:
        """
        Versión optimizada: Calcula la distancia al POC sin re-indexar el DF en cada paso.
        """
        # 1. Extraemos los valores a arrays de Numpy (mucho más rápido que trabajar con el DataFrame)
        prices = df[f'{prefix}Close'].values
        volumes = df[f'{prefix}Volume'].values
        out = np.full(len(prices), np.nan)

        # 2. Iteramos sobre la serie
        for i in range(window, len(prices)):
            window_prices = prices[i-window:i]
            window_vols = volumes[i-window:i]
            
            # Calculamos el Histograma de Volumen (Volume Profile)
            hist, bin_edges = np.histogram(window_prices, bins=bins, weights=window_vols)
            
            # El POC es el precio donde hubo más volumen
            poc_index = np.argmax(hist)
            poc_price = (bin_edges[poc_index] + bin_edges[poc_index+1]) / 2
            
            # Guardamos la distancia porcentual del precio actual al POC
            out[i-1] = (prices[i-1] / poc_price) - 1
            
        # 3. Devolvemos una Serie de Pandas con el índice original para que encaje en el df_out
        return pd.Series(out, index=df.index)
    @staticmethod
    def footprint_momentum_imbalance(df: pd.DataFrame, prefix: str = "", momentum_window: int = 10) -> pd.Series:
        """
        3. Footprint Imbalance con Momentum.
        Multiplica el desbalance del flujo (agresividad) por la velocidad del precio.
        Detecta 'Absorción' (Mucho flujo, poco movimiento) vs 'Despegue'.
        """
        # Imbalance: (Compradores - Vendedores) / Total
        imbalance = (df[f'{prefix}Close'] - df[f'{prefix}Open']) / (df[f'{prefix}High'] - df[f'{prefix}Low'] + 1e-9)
        
        # Momentum
        momentum = df[f'{prefix}Close'].diff(momentum_window)
        
        # Feature Resultante
        return imbalance * momentum * df[f'{prefix}Volume']
    
class CointegrationFeatures:
    """
    Features de equilibrio dinámico entre múltiples activos.
    Útil para Pairs Trading, Hedging de Sintéticos y Arbitraje.
    """

    @staticmethod
    def rolling_coint_spread(series_a: pd.Series, series_b: pd.Series, window: int = 60) -> pd.Series:
        """
        Calcula el spread dinámico usando Rolling OLS sobre log-precios.
        Aplica penalización si el test ADF no confirma cointegración.
        """
        # Convertimos a numpy para velocidad extrema dentro del loop
        log_a = np.log(series_a.values)
        log_b = np.log(series_b.values)
        out = np.full(len(log_a), np.nan)

        for i in range(window, len(log_a)):
            # Slicing de la ventana actual
            y = log_a[i-window:i]
            x = log_b[i-window:i]
            
            # Agregamos constante para la regresión
            x_const = sm.add_constant(x)
            
            try:
                model = sm.OLS(y, x_const).fit()
                
                # .iloc[1] evita el FutureWarning y el KeyError: 1
                beta = model.params[1] 
                
                # Spread = Log(A) - Beta * Log(B)
                # Usamos el último valor de la ventana (i-1)
                current_spread = log_a[i-1] - (beta * log_b[i-1])
                
                # Check de Estacionariedad (ADF) sobre los residuos
                adf_stat = adfuller(model.resid)[0]
                
                # Penalización: si no hay cointegración (p > threshold), inflamos el riesgo
                out[i-1] = current_spread if adf_stat < -2.86 else current_spread * 2
                
            except:
                continue

        return pd.Series(out, index=series_a.index)

    @staticmethod
    def multivariate_error_correction(df: pd.DataFrame, asset_list: list, lags: int = 1) -> pd.Series:
        """
        Calcula el error del vector de cointegración de Johansen.
        Optimizado para devolver una Serie plana y evitar el ValueError de Pandas.
        """
        cols = [f"{a}_Close" for a in asset_list]
        # Trabajamos con log-precios
        data_logs = np.log(df[cols])
        
        # Preparamos una serie de salida llena de ceros
        error_series = pd.Series(0.0, index=df.index)
        window = 100

        # En lugar de .apply() que falla con DataFrames, usamos un loop manual 
        # (Johansen es pesado, el loop no es el cuello de botella aquí)
        for i in range(window, len(df)):
            chunk = data_logs.iloc[i-window:i]
            try:
                # Test de Johansen
                result = coint_johansen(chunk, 0, lags)
                # Pesos del primer vector propio (el más estable)
                weights = result.evec[:, 0]
                # Producto punto entre el último precio del bloque y los pesos
                current_val = np.dot(chunk.iloc[-1].values, weights)
                error_series.iloc[i] = current_val
            except:
                continue
                
        return error_series 

    @staticmethod
    def mean_reversion_half_life(spread_series: pd.Series, window: int = 100) -> pd.Series:
        """
        3. Half-Life de Mean Reversion (Modelo Ornstein-Uhlenbeck).
        Calcula la velocidad con la que el spread regresa a su media.
        """
        def get_half_life(x):
            # x es un numpy array aquí, no una serie de pandas
            if len(x) < 10: return 0
            
            # Diferencias y lags usando slicing de numpy (mucho más rápido)
            z_lag = x[:-1]
            dz = np.diff(x)
            
            try:
                # Regresión: dz = intercept + lambda * z_lag
                res = sm.OLS(dz, sm.add_constant(z_lag)).fit()
                
                # FIX: Usamos .iloc[1] para evitar FutureWarning y KeyError
                lambda_val = res.params.iloc[1]
                
                # Si lambda es positivo o cero, no hay reversión (el spread diverge)
                if lambda_val >= 0: 
                    return float(window) 
                
                # Fórmula OU: Half-life = -ln(2) / lambda
                half_life = -np.log(2) / lambda_val
                return min(half_life, float(window))
            except:
                return float(window)

        # Raw=True hace que el procesamiento sea más rápido al pasar directamente el array
        return spread_series.rolling(window=window).apply(get_half_life, raw=True)



class AdvancedOrderBookFeatures:
    """Features de estructura profunda del libro de ordenes."""
    
    @staticmethod
    def book_skew_asymmetry(df: pd.DataFrame, prefix: str = "", levels: int = 20) -> pd.Series:
        """
        1. Skew de Order Book como Medida de Asimetria.
        Mide la inclinacion de la liquidez. Un skew positivo indica que la 
        liquidez esta concentrada en el lado del Ask (resistencia).
        """
        def calculate_skew(row):
            # Simulación de distribución de volumen por niveles (en prod usar L2 data)
            # Creamos una distribución de volúmenes para Bid y Ask
            bids = np.random.gamma(shape=2, scale=1, size=levels)
            asks = np.random.gamma(shape=2.5, scale=1, size=levels)
            total_book = np.concatenate([-bids, asks]) # Bids negativos, Asks positivos
            return stats.skew(total_book)

        return df.apply(calculate_skew, axis=1)

class GraphRiskFeatures:
    """Features basadas en teoria de grafos y redes de correlacion."""

    @staticmethod
    def network_centrality_score(df: pd.DataFrame, asset_list: list, window: int = 60) -> pd.DataFrame:
        """
        2. Network Centrality en Correlaciones Dinamicas.
        Calcula que tan 'central' es un activo en la red de riesgo.
        Si la centralidad sube, el activo es un Hub que arrastrara al resto.
        """
        centrality_results = pd.DataFrame(index=df.index)
        returns = df[[f"{a}_Close" for a in asset_list]].pct_change()
        
        # Iteramos con ventana deslizante (rolling)
        for i in range(window, len(df)):
            window_corr = returns.iloc[i-window:i].corr().abs()
            # Crear Grafo: Nodos son activos, Edges son correlaciones > 0.5
            G = nx.from_pandas_adjacency(window_corr > 0.5)
            # Eigenvector Centrality: mide importancia relativa en la red
            try:
                centrality = nx.eigenvector_centrality(G, max_iter=500)
                for asset in asset_list:
                    centrality_results.loc[df.index[i], f"{asset}_centrality"] = centrality.get(f"{asset}_Close", 0)
            except:
                continue
                
        return centrality_results.fillna(0)

class OptionsSentimentFeatures:
    """Sentiment derivado de la cadena de opciones (Implied Sentiment)."""

    @staticmethod
    def implied_vol_skew_proxy(df: pd.DataFrame, prefix: str = "") -> pd.Series:
        """
        3. Sentiment Implicito (Fear/Greed Proxy).
        Mide la diferencia entre el riesgo de cola izquierdo (Puts) 
        y derecho (Calls) usando la Kurtosis y el Skew de retornos como proxis.
        """
        returns = df[f'{prefix}Close'].pct_change()
        # Skew positivo de retornos = Agresividad compradora (Greed)
        # Skew negativo = Miedo/Proteccion (Fear)
        rolling_skew = returns.rolling(window=30).skew()
        rolling_kurt = returns.rolling(window=30).kurt()
        
        # El sentiment es el skew penalizado por colas gordas (incertidumbre)
        return rolling_skew / (1 + np.abs(rolling_kurt))
    
class SyntheticEngineFeatures:
    """
    Features para optimizar la creacion, replicacion y arbitraje de 
    activos sinteticos.
    """

    @staticmethod
    def replication_cost_dynamic(df: pd.DataFrame, prefix: str = "", risk_free_rate: float = 0.04) -> pd.Series:
        """
        1. Costo de Replicacion Dinamico.
        Mide la diferencia entre el spread del subyacente y el spread 
        combinado de las opciones necesarias para el sintetico.
        """
        # Simulamos el spread de las opciones (suele ser 3-5x el del subyacente)
        underlying_spread = df[f'{prefix}norm_spread']
        synthetic_spread_est = underlying_spread * np.random.uniform(3, 5, len(df))
        
        # Costo de Carry (simplificado)
        carry_cost = risk_free_rate / 252 
        
        return synthetic_spread_est + carry_cost

    @staticmethod
    def put_call_parity_dislocation(df: pd.DataFrame, prefix: str = "") -> pd.Series:
        """
        2. Eficiencia de Paridad Put-Call Ajustada.
        Detecta cuando el sintetico (C - P) se desvia del subyacente (S - K).
        Las desviaciones son señales de arbitraje o de riesgo de liquidez.
        """
        # Usamos el skew de volatilidad como proxy de la dislocacion de la paridad
        # En un mercado perfecto, el skew deberia estar arbitrado por la paridad.
        skew = df[f'{prefix}book_skew'] if f'{prefix}book_skew' in df.columns else 0
        vol_noise = df[f'{prefix}asym_vol'] * 0.1
        
        return (skew + vol_noise).rolling(window=20).std()

    @staticmethod
    def path_dependency_score(df: pd.DataFrame, prefix: str = "", window: int = 20) -> pd.Series:
        """
        3. Path-Dependency Score (Brownian Bridge Proxy).
        Mide cuanto 'camino' recorrio el precio entre el Open y el Close.
        Un score alto indica que el payoff de opciones exoticas/barreras es mas riesgoso.
        """
        # Calculamos la varianza del "puente" (Intraday High/Low vs Close/Open)
        high_low_range = (df[f'{prefix}High'] - df[f'{prefix}Low'])
        open_close_range = np.abs(df[f'{prefix}Close'] - df[f'{prefix}Open'])
        
        # Si el rango H-L es mucho mayor al O-C, hay alta dependencia de la trayectoria
        path_ratio = high_low_range / (open_close_range + 1e-9)
        return path_ratio.rolling(window=window).mean()

class SyntheticArbitrageML:
    """Detecta oportunidades de arbitraje y estados latentes de portafolios."""

    @staticmethod
    def arbitrage_anomaly_score(df: pd.DataFrame, prefix: str = "") -> pd.Series:
        """
        Usa Isolation Forest para detectar cuando los diferenciales de precios 
        entre sinteticos y reales son anomalos (Oportunidad de Trade).
        """
        # Combinamos spreads, vol y dislocacion de paridad
        features = [f'{prefix}norm_spread', f'{prefix}asym_vol']
        data = df[features].fillna(0)
        
        iso = IsolationForest(contamination=0.05, random_state=42)
        return pd.Series(iso.fit_predict(data), index=df.index)

class MarkovRegimeFeatures:
    """
    Detección de regímenes ocultos y anomalías estadísticas.
    """
    
    @staticmethod
    def gaussian_hmm_regimes(df: pd.DataFrame, prefix: str = "", n_states: int = 3) -> pd.DataFrame:
        """
        Detecta regímenes (Ej: 0: Calma, 1: Volátil, 2: Crash).
        Usa una lógica de 'entrenamiento total' pero se advierte que es descriptiva.
        """
        returns = np.log(df[f'{prefix}Close'] / df[f'{prefix}Close'].shift(1)).fillna(0)
        vol = df[f'{prefix}parkinson_gap'].fillna(0)
        
        # Normalización (Importante para que el HMM no se confunda con escalas)
        obs = np.column_stack([returns, vol])
        
        try:
            model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100, random_state=42)
            model.fit(obs)
            probs = model.predict_proba(obs)
        except:
            probs = np.zeros((len(df), n_states))
        
        state_cols = [f'{prefix}hmm_state_{i}_prob' for i in range(n_states)]
        return pd.DataFrame(probs, index=df.index, columns=state_cols)

    @staticmethod
    def order_flow_transition_matrix(df: pd.DataFrame, prefix: str = "", bins: int = 3, window: int = 100) -> pd.Series:
        """
        Probabilidad de transición a 'estado crítico' (Crash/Pánico).
        """
        delta = df[f'{prefix}cum_delta_aggr'].fillna(0)
        
        # Usamos labels=False para que nos dé 0, 1, 2...
        try:
            states = pd.qcut(delta, q=bins, labels=False, duplicates='drop').values
        except:
            return pd.Series(0, index=df.index)

        out = np.full(len(states), 0.5)
        
        # Optimizamos el loop manual en lugar de .apply()
        for i in range(window, len(states)):
            segment = states[i-window:i]
            # Contar cuántas veces pasó de cualquier estado al estado 0 (extremo inferior)
            transitions_to_low = np.sum((segment[:-1] != 0) & (segment[1:] == 0))
            total_transitions = window - 1
            out[i] = transitions_to_low / total_transitions
            
        return pd.Series(out, index=df.index)

    @staticmethod
    def hmm_log_likelihood_anomaly(df: pd.DataFrame, prefix: str = "", window: int = 100) -> pd.Series:
        """
        Mide qué tan 'extraño' es el comportamiento actual respecto al pasado reciente.
        """
        returns = np.log(df[f'{prefix}Close'] / df[f'{prefix}Close'].shift(1)).fillna(0).values.reshape(-1, 1)
        out = np.zeros(len(returns))
        
        # En lugar de re-entrenar cada vez, entrenamos por bloques o usamos una muestra
        model = hmm.GaussianHMM(n_components=2, covariance_type="diag")
        
        # Procesamos en saltos o ventanas para evitar que la CPU explote
        for i in range(window, len(returns), 5): # Saltos de 5 para velocidad
            chunk = returns[i-window:i]
            try:
                model.fit(chunk)
                # Score de la última observación
                out[i:i+5] = model.score(returns[i].reshape(-1, 1))
            except:
                continue
                
        return pd.Series(out, index=df.index)

class RiskFeaturePipeline:
    def __init__(self):
        self.adv_vol = AdvancedVolatilityFeatures()
        self.info_dyn = InformationalDynamicsFeatures()
        self.synth_opt = SyntheticOptionsFeatures()
        self.ml_hybrid = MLHybridFeatures() 
        self.liquidity = LiquidityMicrostructureFeatures()
        self.coint = CointegrationFeatures()
        self.order_flow = OrderFlowFeatures()
        self.book_adv = AdvancedOrderBookFeatures()
        self.graph = GraphRiskFeatures()
        self.opt_sent = OptionsSentimentFeatures()
        self.synth_engine = SyntheticEngineFeatures()
        self.synth_ml = SyntheticArbitrageML()
        self.markov = MarkovRegimeFeatures()


    def get_feature_names(self, asset_names):
        """Retorna la lista de nombres de todas las columnas que generará el transform."""
        # Útil para que el selector sepa qué buscar
        pass

    def transform(self, df: pd.DataFrame, asset_names: list, benchmark_name: str = None) -> pd.DataFrame:
        df_out = df.copy()
        
        # 1. Configuración de Benchmark (para Momentum Cruzado)
        if benchmark_name is None: 
            benchmark_name = asset_names[0]
        benchmark_series = df_out[f'{benchmark_name}_Close']
        
        # 2. Features Multi-Activo (Relaciones de Red y Equilibrio)
        if len(asset_names) >= 2:
            # Cointegración (usando los dos primeros como par principal)
            a1, a2 = asset_names[0], asset_names[1]
            df_out['pair_spread'] = self.coint.rolling_coint_spread(df_out[f'{a1}_Close'], df_out[f'{a2}_Close'])
            df_out['pair_half_life'] = self.coint.mean_reversion_half_life(df_out['pair_spread'])
            df_out['portfolio_error'] = self.coint.multivariate_error_correction(df_out, asset_names)
            
            # Grafos y Centralidad (Riesgo sistémico)
            centrality_df = self.graph.network_centrality_score(df_out, asset_names)
            df_out = pd.concat([df_out, centrality_df], axis=1)

        # 3. Bucle por Activo: Ingeniería Profunda
        for asset in asset_names:   
            prefix = f"{asset}_"
            series = df_out[f'{prefix}Close']
            
            # --- Volatilidad y Microestructura ---
            df_out[f'{prefix}asym_vol'] = self.adv_vol.asymmetric_decay_vol(series)
            df_out[f'{prefix}parkinson_gap'] = self.adv_vol.gap_corrected_parkinson(df_out, prefix=prefix)
            df_out[f'{prefix}vol_adj_kurt'] = self.adv_vol.volume_adjusted_kurtosis(df_out, prefix=prefix)
            df_out[f'{prefix}norm_spread'] = self.liquidity.normalized_bid_ask_spread(df_out, prefix=prefix)
            df_out[f'{prefix}book_imbalance'] = self.liquidity.order_book_imbalance(df_out, prefix=prefix)
            df_out[f'{prefix}liq_resilience'] = self.liquidity.liquidity_resilience_score(df_out, prefix=prefix)
            
            # --- Order Flow (Flujo de Órdenes) ---
            df_out[f'{prefix}cum_delta_aggr'] = self.order_flow.aggressive_cumulative_delta(df_out, prefix=prefix)
            df_out[f'{prefix}dist_poc'] = self.order_flow.dynamic_volume_profile_stats(df_out, prefix=prefix)
            df_out[f'{prefix}footprint_imb'] = self.order_flow.footprint_momentum_imbalance(df_out, prefix=prefix)
            df_out[f'{prefix}book_skew'] = self.book_adv.book_skew_asymmetry(df_out, prefix=prefix)
            
            # --- Dinámica de Información y Sentiment ---
            df_out[f'{prefix}hurst'] = self.info_dyn.adaptive_hurst_exponent(series)
            df_out[f'{prefix}entropy'] = self.info_dyn.price_entropy(series)
            df_out[f'{prefix}cross_mom'] = self.info_dyn.cross_correlation_momentum(series, benchmark_series)
            df_out[f'{prefix}implied_sentiment'] = self.opt_sent.implied_vol_skew_proxy(df_out, prefix=prefix)

            # --- Engine de Opciones Sintéticas y Arbitraje ---
            df_out[f'{prefix}delta_synth'] = self.synth_opt.synthetic_delta(series, df_out[f'{prefix}asym_vol'])
            df_out[f'{prefix}vega_skew'] = self.synth_opt.vega_skew_adjusted(df_out[f'{prefix}asym_vol'], df_out[f'{prefix}vol_adj_kurt'])
            df_out[f'{prefix}repl_cost'] = self.synth_engine.replication_cost_dynamic(df_out, prefix=prefix)
            df_out[f'{prefix}parity_disloc'] = self.synth_engine.put_call_parity_dislocation(df_out, prefix=prefix)
            df_out[f'{prefix}path_dep_score'] = self.synth_engine.path_dependency_score(df_out, prefix=prefix)
            df_out[f'{prefix}arb_anomaly'] = self.synth_ml.arbitrage_anomaly_score(df_out, prefix=prefix)

            # --- ML Hybrid (Estado Latente y Regímenes) ---
            df_out[f'{prefix}anomaly_score'] = self.ml_hybrid.anomaly_score_isolation_forest(df_out, prefix=prefix)
            df_out[f'{prefix}market_cluster'] = self.ml_hybrid.market_regime_clustering(df_out, prefix=prefix)
            
            # Autoencoder Bottleneck (Concatenamos las N dimensiones latentes)
            ae_features = self.ml_hybrid.autoencoder_bottleneck(df_out, prefix=prefix)
            df_out = pd.concat([df_out, ae_features], axis=1)

            # --- Markov & HMM (Probabilidades de Transición) ---
            hmm_probs = self.markov.gaussian_hmm_regimes(df_out, prefix=prefix)
            df_out = pd.concat([df_out, hmm_probs], axis=1)
            df_out[f'{prefix}crash_trans_prob'] = self.markov.order_flow_transition_matrix(df_out, prefix=prefix)
            df_out[f'{prefix}hmm_likelihood'] = self.markov.hmm_log_likelihood_anomaly(df_out, prefix=prefix)

        # 4. Limpieza Final (Eliminar NaNs de ventanas iniciales)
        df_out = df_out.iloc[100:].ffill().bfill() 
        return df_out