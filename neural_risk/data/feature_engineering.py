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
        weighted_returns = returns.copy()
        weighted_returns[returns < 0] = returns[returns < 0] * penalty
        vol = weighted_returns.pow(2).ewm(alpha=1-decay, min_periods=window).mean()
        return np.sqrt(vol) * np.sqrt(252)

    @staticmethod
    def gap_corrected_parkinson(df: pd.DataFrame, prefix: str = "", window: int = 14, gap_mult: float = 1.0) -> pd.Series:
        """
        2. Parkinson Volatility con corrección por Gaps Overnight.
        """
        h = df[f'{prefix}High']
        l = df[f'{prefix}Low']
        o = df[f'{prefix}Open']
        c_prev = df[f'{prefix}Close'].shift(1)
        parkinson_core = (1 / (4 * np.log(2))) * np.log(h/l)**2
        gap_risk = np.log(o / c_prev)**2
        total_var = parkinson_core + (gap_mult * gap_risk)
        vol = total_var.rolling(window=window).median()
        return np.sqrt(vol * 252)

    @staticmethod
    def volume_adjusted_kurtosis(df: pd.DataFrame, prefix: str = "", window: int = 40) -> pd.Series:
        """
        3. Kurtosis Ajustada por Volumen.
        """
        returns = np.log(df[f'{prefix}Close'] / df[f'{prefix}Close'].shift(1))
        volume = df[f'{prefix}Volume']
        rolling_kurt = returns.rolling(window=window).kurt()
        norm_vol = np.log(volume.rolling(window=window).mean())
        return rolling_kurt / norm_vol


class InformationalDynamicsFeatures:
    """Features basadas en momentum, geometria fractal y teoria de la informacion."""

    @staticmethod
    def cross_correlation_momentum(series: pd.Series, benchmark: pd.Series, alpha: float = 0.05, window: int = 20) -> pd.Series:
        exp_roc = series.pct_change().ewm(alpha=alpha).mean()
        rolling_corr = series.pct_change().rolling(window=window).corr(benchmark.pct_change())
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
                probs = probs[probs > 0]
                return -np.sum(probs * np.log(probs))
            except:
                return 0
        returns = series.pct_change().dropna()
        return returns.rolling(window=window).apply(calculate_shannon)


class SyntheticOptionsFeatures:
    """Features de sensibilidad (Greeks) y Payoffs para activos sintéticos."""

    @staticmethod
    def synthetic_delta(series: pd.Series, vol_series: pd.Series, h: float = 0.01) -> pd.Series:
        """
        1. Delta Sintético Ajustado por Volatilidad.
        Usa diferencias finitas para estimar la sensibilidad.
        Delta = [f(S + h) - f(S - h)] / 2h, ajustado por el régimen de volatilidad.

        FIX (look-ahead bias CRÍTICO): la versión original usaba
        series.shift(-1), es decir, el precio del PERÍODO SIGUIENTE
        (futuro) para calcular el valor de HOY. En backtest esto genera
        resultados artificialmente buenos (la feature "sabe" el futuro
        inmediato); en vivo esa columna directamente no puede existir.
        Es data leakage clásico -- si el jurado llegaba a aprobar esta
        feature (muy probable, dada su correlación artificial con el
        target), cualquier modelo que la usara sería inválido para
        producción aunque luciera excelente en backtest.

        Ahora se usa una diferencia hacia atrás (solo pasado/presente):
        aproxima la derivada usando 'series' (dato actual, disponible)
        y 'series.shift(2)' (dato de hace 2 períodos, también
        disponible) -- ningún término requiere datos futuros.
        """
        delta_approx = (series - series.shift(2)) / (2 * h * series)
        return delta_approx * (1 / (1 + vol_series))

    @staticmethod
    def vega_skew_adjusted(vol_series: pd.Series, kurtosis_series: pd.Series) -> pd.Series:
        """2. Vega con Ajuste de Skew/Smile."""
        base_vega = vol_series * 0.5
        skew_adj = np.log(1 + np.abs(kurtosis_series))
        return base_vega * (1 + skew_adj)

    @staticmethod
    def payoff_profile_vector(series: pd.Series, window: int = 20, n_scenarios: int = 10) -> pd.Series:
        """3. Payoff Profile (Feature Vector). NOTA: nunca se llama desde transform() -- huérfano."""
        def calculate_expected_payoff(x):
            if len(x) < window: return 0
            std = np.std(x)
            current_price = x[-1]
            up_scenario = current_price + (std * 1.5)
            down_scenario = current_price - (std * 1.5)
            payoff = (np.abs(up_scenario - current_price) + np.abs(down_scenario - current_price)) / 2
            return payoff
        return series.rolling(window=window).apply(calculate_expected_payoff)


class MLHybridFeatures:
    """Genera features de avanzada utilizando sub-modelos de ML/DL."""

    @staticmethod
    def anomaly_score_isolation_forest(df: pd.DataFrame, prefix: str = "", window: int = 100) -> pd.Series:
        cols = [f'{prefix}Close', f'{prefix}Volume', f'{prefix}parkinson_gap']
        data = df[cols].fillna(0)
        iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        scores = iso.fit_predict(data)
        return pd.Series(scores, index=df.index)

    @staticmethod
    def market_regime_clustering(df: pd.DataFrame, prefix: str = "", n_clusters: int = 3) -> pd.Series:
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

        AVISO (no arreglado, requiere decision de diseño): se instancia
        una red NUEVA Y SIN ENTRENAR en cada llamada -- no hay training
        loop, solo un forward pass con pesos random. Las columnas
        'ae_dim_*' resultantes son una proyección aleatoria de OHLCV,
        no una compresión aprendida. No se arregla acá porque entrenar
        de verdad requiere decidir dónde vive ese entrenamiento (¿un
        tier más en train_models.py, con qué función de pérdida --
        reconstrucción?) -- decisión de diseño pendiente, no un bug de
        una línea.
        """
        import torch
        import torch.nn as nn
        cols = [f'{prefix}Open', f'{prefix}High', f'{prefix}Low', f'{prefix}Close', f'{prefix}Volume']
        data = torch.FloatTensor(df[cols].values)

        class SimpleAE(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 8), nn.ReLU(), nn.Linear(8, latent_dim)
                )
            def forward(self, x):
                return self.encoder(x)

        model = SimpleAE(input_dim=len(cols), latent_dim=bottleneck_dim)
        with torch.no_grad():
            latent_features = model(data).numpy()

        latent_cols = [f'{prefix}ae_dim_{i}' for i in range(bottleneck_dim)]
        return pd.DataFrame(latent_features, index=df.index, columns=latent_cols)


class LiquidityMicrostructureFeatures:
    @staticmethod
    def normalized_bid_ask_spread(df: pd.DataFrame, prefix: str = "", vol_window: int = 20) -> pd.Series:
        if f'{prefix}Ask' in df.columns and f'{prefix}Bid' in df.columns:
            spread_rel = (df[f'{prefix}Ask'] - df[f'{prefix}Bid']) / ((df[f'{prefix}Ask'] + df[f'{prefix}Bid']) / 2)
        else:
            spread_rel = (df[f'{prefix}High'] - df[f'{prefix}Low']) / df[f'{prefix}Close']
        vol = df[f'{prefix}Close'].pct_change().rolling(window=vol_window).std()
        return (spread_rel / (vol + 1e-9)).fillna(0)

    @staticmethod
    def order_book_imbalance(df: pd.DataFrame, prefix: str = "") -> pd.Series:
        h, l, c = df[f'{prefix}High'], df[f'{prefix}Low'], df[f'{prefix}Close']
        v = df[f'{prefix}Volume']
        ii_index = ((2 * c - h - l) / (h - l + 1e-9)) * v
        return ii_index.rolling(window=20).mean()

    @staticmethod
    def liquidity_resilience_score(df: pd.DataFrame, prefix: str = "", window: int = 10) -> pd.Series:
        vol_mean = df[f'{prefix}Volume'].rolling(window=50).mean()
        high_vol_event = (df[f'{prefix}Volume'] > vol_mean * 2).astype(int)
        target_col = f'{prefix}norm_spread'
        if target_col in df.columns:
            spread = df[target_col]
        else:
            spread = pd.Series(np.random.rand(len(df)) * 0.001, index=df.index)
        recovery = spread.diff(window).fillna(0)
        resilience = (high_vol_event * -recovery)
        return resilience.rolling(window=window).mean()


class OrderFlowFeatures:
    """Features de flujo de órdenes."""

    @staticmethod
    def aggressive_cumulative_delta(df: pd.DataFrame, prefix: str = "", window: int = 50) -> pd.Series:
        side = np.where(df[f'{prefix}Close'] >= df[f'{prefix}Open'], 1, -1)
        delta = side * df[f'{prefix}Volume']
        return delta.rolling(window=window).sum()

    @staticmethod
    def dynamic_volume_profile_stats(df: pd.DataFrame, prefix: str = "", bins: int = 30, window: int = 60) -> pd.Series:
        prices = df[f'{prefix}Close'].values
        volumes = df[f'{prefix}Volume'].values
        out = np.full(len(prices), np.nan)

        # FIX (off-by-one): antes el rango terminaba en len(prices) y se
        # escribía out[i-1] -- la ÚLTIMA fila (la más reciente, la que
        # importa para una decisión en vivo) nunca se llenaba y quedaba
        # NaN hasta el ffill() final de transform(), usando entonces el
        # valor de la vela ANTERIOR en vez del actual. Ahora el rango
        # llega hasta len(prices)+1 para que la fila más reciente
        # también se calcule.
        for i in range(window, len(prices) + 1):
            window_prices = prices[i-window:i]
            window_vols = volumes[i-window:i]
            hist, bin_edges = np.histogram(window_prices, bins=bins, weights=window_vols)
            poc_index = np.argmax(hist)
            poc_price = (bin_edges[poc_index] + bin_edges[poc_index+1]) / 2
            out[i-1] = (prices[i-1] / poc_price) - 1

        return pd.Series(out, index=df.index)

    @staticmethod
    def footprint_momentum_imbalance(df: pd.DataFrame, prefix: str = "", momentum_window: int = 10) -> pd.Series:
        imbalance = (df[f'{prefix}Close'] - df[f'{prefix}Open']) / (df[f'{prefix}High'] - df[f'{prefix}Low'] + 1e-9)
        momentum = df[f'{prefix}Close'].diff(momentum_window)
        return imbalance * momentum * df[f'{prefix}Volume']


class CointegrationFeatures:
    """Features de equilibrio dinámico entre múltiples activos."""

    @staticmethod
    def rolling_coint_spread(series_a: pd.Series, series_b: pd.Series, window: int = 60) -> pd.Series:
        log_a = np.log(series_a.values)
        log_b = np.log(series_b.values)
        out = np.full(len(log_a), np.nan)

        # FIX (mismo off-by-one que en dynamic_volume_profile_stats):
        # rango extendido a len(log_a)+1. NOTA: esta función solo se
        # llama con >=2 activos, y hoy engine.py siempre procesa 1
        # activo a la vez -- código actualmente inalcanzable, pero
        # corregido para cuando se active multi-asset.
        for i in range(window, len(log_a) + 1):
            y = log_a[i-window:i]
            x = log_b[i-window:i]
            x_const = sm.add_constant(x)
            try:
                model = sm.OLS(y, x_const).fit()
                beta = model.params[1]
                current_spread = log_a[i-1] - (beta * log_b[i-1])
                adf_stat = adfuller(model.resid)[0]
                out[i-1] = current_spread if adf_stat < -2.86 else current_spread * 2
            except:
                continue

        return pd.Series(out, index=series_a.index)

    @staticmethod
    def multivariate_error_correction(df: pd.DataFrame, asset_list: list, lags: int = 1) -> pd.Series:
        cols = [f"{a}_Close" for a in asset_list]
        data_logs = np.log(df[cols])
        error_series = pd.Series(0.0, index=df.index)
        window = 100

        for i in range(window, len(df)):
            chunk = data_logs.iloc[i-window:i]
            try:
                result = coint_johansen(chunk, 0, lags)
                weights = result.evec[:, 0]
                current_val = np.dot(chunk.iloc[-1].values, weights)
                error_series.iloc[i] = current_val
            except:
                continue

        return error_series

    @staticmethod
    def mean_reversion_half_life(spread_series: pd.Series, window: int = 100) -> pd.Series:
        def get_half_life(x):
            if len(x) < 10: return 0
            z_lag = x[:-1]
            dz = np.diff(x)
            try:
                res = sm.OLS(dz, sm.add_constant(z_lag)).fit()
                lambda_val = res.params.iloc[1]
                if lambda_val >= 0:
                    return float(window)
                half_life = -np.log(2) / lambda_val
                return min(half_life, float(window))
            except:
                return float(window)
        return spread_series.rolling(window=window).apply(get_half_life, raw=True)


class AdvancedOrderBookFeatures:
    """Features de estructura profunda del libro de ordenes."""

    @staticmethod
    def book_skew_asymmetry(df: pd.DataFrame, prefix: str = "", levels: int = 20) -> pd.Series:
        """
        1. Skew de Order Book como Medida de Asimetria.

        AVISO (no arreglado, requiere decision de diseño): calculate_skew
        genera bids/asks con np.random.gamma() en CADA fila, ignorando
        por completo el contenido real de 'row' -- esta columna es ruido
        estadístico puro, no mide nada del mercado real. Se usa después
        en put_call_parity_dislocation(), contaminándola también. No se
        arregla acá porque la solución real requiere datos L2 reales del
        order book (que hoy no existen en el sistema) -- mientras tanto,
        esta feature debería tratarse como no confiable si aparece
        aprobada por el jurado.
        """
        def calculate_skew(row):
            bids = np.random.gamma(shape=2, scale=1, size=levels)
            asks = np.random.gamma(shape=2.5, scale=1, size=levels)
            total_book = np.concatenate([-bids, asks])
            return stats.skew(total_book)
        return df.apply(calculate_skew, axis=1)


class GraphRiskFeatures:
    """Features basadas en teoria de grafos y redes de correlacion."""

    @staticmethod
    def network_centrality_score(df: pd.DataFrame, asset_list: list, window: int = 60) -> pd.DataFrame:
        centrality_results = pd.DataFrame(index=df.index)
        returns = df[[f"{a}_Close" for a in asset_list]].pct_change()

        for i in range(window, len(df)):
            window_corr = returns.iloc[i-window:i].corr().abs()
            G = nx.from_pandas_adjacency(window_corr > 0.5)
            try:
                centrality = nx.eigenvector_centrality(G, max_iter=500)
                for asset in asset_list:
                    centrality_results.loc[df.index[i], f"{asset}_centrality"] = centrality.get(f"{asset}_Close", 0)
            except:
                continue

        return centrality_results.fillna(0)


class OptionsSentimentFeatures:
    """Sentiment derivado de la cadena de opciones."""

    @staticmethod
    def implied_vol_skew_proxy(df: pd.DataFrame, prefix: str = "") -> pd.Series:
        returns = df[f'{prefix}Close'].pct_change()
        rolling_skew = returns.rolling(window=30).skew()
        rolling_kurt = returns.rolling(window=30).kurt()
        return rolling_skew / (1 + np.abs(rolling_kurt))


class SyntheticEngineFeatures:
    """Features para optimizar la creacion, replicacion y arbitraje de activos sinteticos."""

    @staticmethod
    def replication_cost_dynamic(df: pd.DataFrame, prefix: str = "", risk_free_rate: float = 0.04) -> pd.Series:
        underlying_spread = df[f'{prefix}norm_spread']
        synthetic_spread_est = underlying_spread * np.random.uniform(3, 5, len(df))
        carry_cost = risk_free_rate / 252
        return synthetic_spread_est + carry_cost

    @staticmethod
    def put_call_parity_dislocation(df: pd.DataFrame, prefix: str = "") -> pd.Series:
        skew = df[f'{prefix}book_skew'] if f'{prefix}book_skew' in df.columns else 0
        vol_noise = df[f'{prefix}asym_vol'] * 0.1
        return (skew + vol_noise).rolling(window=20).std()

    @staticmethod
    def path_dependency_score(df: pd.DataFrame, prefix: str = "", window: int = 20) -> pd.Series:
        high_low_range = (df[f'{prefix}High'] - df[f'{prefix}Low'])
        open_close_range = np.abs(df[f'{prefix}Close'] - df[f'{prefix}Open'])
        path_ratio = high_low_range / (open_close_range + 1e-9)
        return path_ratio.rolling(window=window).mean()


class SyntheticArbitrageML:
    """Detecta oportunidades de arbitraje y estados latentes de portafolios."""

    @staticmethod
    def arbitrage_anomaly_score(df: pd.DataFrame, prefix: str = "") -> pd.Series:
        features = [f'{prefix}norm_spread', f'{prefix}asym_vol']
        data = df[features].fillna(0)
        iso = IsolationForest(contamination=0.05, random_state=42)
        return pd.Series(iso.fit_predict(data), index=df.index)


class MarkovRegimeFeatures:
    """Detección de regímenes ocultos y anomalías estadísticas."""

    @staticmethod
    def gaussian_hmm_regimes(df: pd.DataFrame, prefix: str = "", n_states: int = 3) -> pd.DataFrame:
        returns = np.log(df[f'{prefix}Close'] / df[f'{prefix}Close'].shift(1)).fillna(0)
        vol = df[f'{prefix}parkinson_gap'].fillna(0)
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
        delta = df[f'{prefix}cum_delta_aggr'].fillna(0)
        try:
            states = pd.qcut(delta, q=bins, labels=False, duplicates='drop').values
        except:
            return pd.Series(0, index=df.index)

        out = np.full(len(states), 0.5)
        for i in range(window, len(states)):
            segment = states[i-window:i]
            transitions_to_low = np.sum((segment[:-1] != 0) & (segment[1:] == 0))
            total_transitions = window - 1
            out[i] = transitions_to_low / total_transitions

        return pd.Series(out, index=df.index)

    @staticmethod
    def hmm_log_likelihood_anomaly(df: pd.DataFrame, prefix: str = "", window: int = 100) -> pd.Series:
        returns = np.log(df[f'{prefix}Close'] / df[f'{prefix}Close'].shift(1)).fillna(0).values.reshape(-1, 1)
        out = np.zeros(len(returns))
        model = hmm.GaussianHMM(n_components=2, covariance_type="diag")

        for i in range(window, len(returns), 5):
            chunk = returns[i-window:i]
            try:
                model.fit(chunk)
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
        """NOTA: stub, retorna None siempre. No lo usa nadie hoy (FeatureJury decide dinámicamente)."""
        pass

    def transform(self, df: pd.DataFrame, asset_names: list, benchmark_name: str = None) -> pd.DataFrame:
        df_out = df.copy()

        if benchmark_name is None:
            benchmark_name = asset_names[0]
        benchmark_series = df_out[f'{benchmark_name}_Close']

        if len(asset_names) >= 2:
            a1, a2 = asset_names[0], asset_names[1]
            df_out['pair_spread'] = self.coint.rolling_coint_spread(df_out[f'{a1}_Close'], df_out[f'{a2}_Close'])
            df_out['pair_half_life'] = self.coint.mean_reversion_half_life(df_out['pair_spread'])
            df_out['portfolio_error'] = self.coint.multivariate_error_correction(df_out, asset_names)

            centrality_df = self.graph.network_centrality_score(df_out, asset_names)
            df_out = pd.concat([df_out, centrality_df], axis=1)

        for asset in asset_names:
            prefix = f"{asset}_"
            series = df_out[f'{prefix}Close']

            df_out[f'{prefix}asym_vol'] = self.adv_vol.asymmetric_decay_vol(series)
            df_out[f'{prefix}parkinson_gap'] = self.adv_vol.gap_corrected_parkinson(df_out, prefix=prefix)
            df_out[f'{prefix}vol_adj_kurt'] = self.adv_vol.volume_adjusted_kurtosis(df_out, prefix=prefix)
            df_out[f'{prefix}norm_spread'] = self.liquidity.normalized_bid_ask_spread(df_out, prefix=prefix)
            df_out[f'{prefix}book_imbalance'] = self.liquidity.order_book_imbalance(df_out, prefix=prefix)
            df_out[f'{prefix}liq_resilience'] = self.liquidity.liquidity_resilience_score(df_out, prefix=prefix)

            df_out[f'{prefix}cum_delta_aggr'] = self.order_flow.aggressive_cumulative_delta(df_out, prefix=prefix)
            df_out[f'{prefix}dist_poc'] = self.order_flow.dynamic_volume_profile_stats(df_out, prefix=prefix)
            df_out[f'{prefix}footprint_imb'] = self.order_flow.footprint_momentum_imbalance(df_out, prefix=prefix)
            df_out[f'{prefix}book_skew'] = self.book_adv.book_skew_asymmetry(df_out, prefix=prefix)

            df_out[f'{prefix}hurst'] = self.info_dyn.adaptive_hurst_exponent(series)
            df_out[f'{prefix}entropy'] = self.info_dyn.price_entropy(series)
            df_out[f'{prefix}cross_mom'] = self.info_dyn.cross_correlation_momentum(series, benchmark_series)
            df_out[f'{prefix}implied_sentiment'] = self.opt_sent.implied_vol_skew_proxy(df_out, prefix=prefix)

            df_out[f'{prefix}delta_synth'] = self.synth_opt.synthetic_delta(series, df_out[f'{prefix}asym_vol'])
            df_out[f'{prefix}vega_skew'] = self.synth_opt.vega_skew_adjusted(df_out[f'{prefix}asym_vol'], df_out[f'{prefix}vol_adj_kurt'])
            df_out[f'{prefix}repl_cost'] = self.synth_engine.replication_cost_dynamic(df_out, prefix=prefix)
            df_out[f'{prefix}parity_disloc'] = self.synth_engine.put_call_parity_dislocation(df_out, prefix=prefix)
            df_out[f'{prefix}path_dep_score'] = self.synth_engine.path_dependency_score(df_out, prefix=prefix)
            df_out[f'{prefix}arb_anomaly'] = self.synth_ml.arbitrage_anomaly_score(df_out, prefix=prefix)

            df_out[f'{prefix}anomaly_score'] = self.ml_hybrid.anomaly_score_isolation_forest(df_out, prefix=prefix)
            df_out[f'{prefix}market_cluster'] = self.ml_hybrid.market_regime_clustering(df_out, prefix=prefix)

            ae_features = self.ml_hybrid.autoencoder_bottleneck(df_out, prefix=prefix)
            df_out = pd.concat([df_out, ae_features], axis=1)

            hmm_probs = self.markov.gaussian_hmm_regimes(df_out, prefix=prefix)
            df_out = pd.concat([df_out, hmm_probs], axis=1)
            df_out[f'{prefix}crash_trans_prob'] = self.markov.order_flow_transition_matrix(df_out, prefix=prefix)
            df_out[f'{prefix}hmm_likelihood'] = self.markov.hmm_log_likelihood_anomaly(df_out, prefix=prefix)

        df_out = df_out.iloc[100:].ffill().bfill()
        return df_out