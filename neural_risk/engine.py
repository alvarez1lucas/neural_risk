# neural_risk/engine.py

class AutomatedRiskEngine:
    def __init__(self, processor, pipeline, selector, model, router):
        self.processor = processor
        self.pipeline = pipeline
        self.selector = selector
        self.model = model
        self.router = router

    def run_full_automation(self, raw_data, target_col):
        # 1. Limpieza
        df_clean = self.processor.auto_clean(raw_data)
        
        # 2. Generación de Features
        df_features = self.pipeline.transform(df_clean)
        
        # 3. Selección Automática (Tu pedido: "que elija las mejores")
        best_features = self.selector.filter_by_causality(df_features, target=target_col)
        
        # 4. Retornamos el estado para que el usuario decida si entrenar o inferir
        return df_features[best_features], best_features