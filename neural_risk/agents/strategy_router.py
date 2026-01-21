class StrategyRouter:
    """
    Orquestador de alto nivel. Toma decisiones basadas en 
    el contexto del mercado generado por el pipeline de features.
    """
    def __init__(self, risk_threshold=0.7):
        self.risk_threshold = risk_threshold

    def decide_strategy(self, market_state: dict):
        """
        market_state: Diccionario con outputs de features clave
        (HMM probs, Sentiment, Volatility Z-Score).
        """
        # 1. Prioridad: Proteccion de Capital (Crash Detection)
        if market_state.get('crash_prob', 0) > self.risk_threshold:
            return {
                "action": "HEAVY_HEDGE",
                "model": "Volatility_Tail_Risk_Model",
                "reason": "High probability of regime shift (HMM)"
            }

        # 2. Analisis de Regimen para Tendencia vs Reversion
        # Usamos el Hurst Exponent y la Entropia
        if market_state.get('hurst', 0.5) > 0.55 and market_state.get('entropy', 1.0) < 0.6:
            return {
                "action": "TREND_FOLLOWING",
                "model": "LSTM_Momentum_Model",
                "reason": "Market is persistent and ordered"
            }

        # 3. Analisis de Corto Plazo (Order Flow / Ineficiencias)
        if market_state.get('arb_opportunity', 0) > 0.8:
            return {
                "action": "SCALPING_ARBITRAGE",
                "model": "Microstructure_Model",
                "reason": "Order flow imbalance detected"
            }

        return {"action": "NEUTRAL", "model": "Base_Model", "reason": "Wait for signal"}