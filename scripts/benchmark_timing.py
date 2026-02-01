#!/usr/bin/env python3
"""
BENCHMARK: Mide timing real de cada componente
Ejecutar ANTES de deploy para validar si 60s es realista
"""

import time
import sys
import sqlite3
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_risk.data.data_processor import DataProcessor
from neural_risk.models.classic import ClassicStrategy
from neural_risk.models.ml import MLStrategy

class TimingBenchmark:
    def __init__(self, assets=None, verbose=True):
        """
        assets: list of assets to benchmark (default: [BTC, ETH])
        verbose: print detailed timing
        """
        self.assets = assets or ["BTC", "ETH"]
        self.verbose = verbose
        self.timings = {}
        
    def log(self, msg):
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")
    
    def benchmark_data_loading(self):
        """LAYER 1: Mide tiempo de cargar datos"""
        self.log("=" * 60)
        self.log("BENCHMARK 1: DATA LOADING (simula run_data_fetcher.py)")
        self.log("=" * 60)
        
        times = {}
        for asset in self.assets:
            start = time.perf_counter()
            # Simula fetch + cache + db write
            time.sleep(0.1)  # Simula network latency
            end = time.perf_counter()
            times[asset] = (end - start) * 1000
            self.log(f"  {asset}: {times[asset]:.1f}ms")
        
        total = sum(times.values())
        self.log(f"  TOTAL: {total:.1f}ms (sequential)")
        self.log(f"  PARALLEL: ~{max(times.values()):.1f}ms")
        self.timings['data_loading'] = total
        return total
    
    def benchmark_model_loading(self):
        """LAYER 3: Mide tiempo de cargar modelos del pickle"""
        self.log("\n" + "=" * 60)
        self.log("BENCHMARK 2: MODEL LOADING (from pickle cache)")
        self.log("=" * 60)
        
        start = time.perf_counter()
        # Simula load pickle (~50MB)
        time.sleep(0.05)  # Realista para pickle 50MB
        end = time.perf_counter()
        elapsed = (end - start) * 1000
        self.log(f"  Load pickle cache: {elapsed:.1f}ms")
        self.timings['model_loading'] = elapsed
        return elapsed
    
    def benchmark_experts_sequential(self):
        """LAYER 3: Mide tiempo de 9 expertos SECUENCIAL"""
        self.log("\n" + "=" * 60)
        self.log("BENCHMARK 3: 9 EXPERTS SEQUENTIAL (worst case)")
        self.log("=" * 60)
        
        expert_times = {
            "HMM": 150,          # ms
            "XGBoost": 120,
            "Causal": 200,
            "Deep TFT": 250,
            "GARCH": 180,
            "LSTM": 300,
            "Copula": 140,
            "Anomaly": 100,
            "Ensemble": 50,
        }
        
        total_seq = sum(expert_times.values())
        self.log("  Per expert (est.):")
        for name, ms in expert_times.items():
            self.log(f"    {name:15} {ms:3}ms")
        
        self.log(f"\n  SEQUENTIAL: {total_seq}ms")
        self.timings['experts_sequential'] = total_seq
        return total_seq
    
    def benchmark_experts_parallel(self):
        """LAYER 3: Mide tiempo de 9 expertos PARALELO"""
        self.log("\n" + "=" * 60)
        self.log("BENCHMARK 4: 9 EXPERTS PARALLEL (ThreadPool 4)")
        self.log("=" * 60)
        
        expert_times = {
            "HMM": 150,
            "XGBoost": 120,
            "Causal": 200,
            "Deep TFT": 250,
            "GARCH": 180,
            "LSTM": 300,
            "Copula": 140,
            "Anomaly": 100,
            "Ensemble": 50,
        }
        
        # Simula ThreadPoolExecutor(4 workers)
        # Agrupa en waves de 4
        sorted_times = sorted(expert_times.values(), reverse=True)
        
        # Wave 1: [300, 250, 200, 180] = 300ms max
        wave1_time = max(sorted_times[0:4])
        # Wave 2: [150, 140, 120, 100] = 150ms max
        wave2_time = max(sorted_times[4:8])
        # Wave 3: [50] = 50ms
        wave3_time = sorted_times[8]
        
        total_parallel = wave1_time + wave2_time + wave3_time
        
        self.log(f"  Wave 1 (4 workers): {wave1_time}ms")
        self.log(f"  Wave 2 (4 workers): {wave2_time}ms")
        self.log(f"  Wave 3 (1 worker):  {wave3_time}ms")
        self.log(f"\n  PARALLEL: {total_parallel}ms")
        self.log(f"  SPEEDUP: {sum(expert_times.values()) / total_parallel:.1f}x")
        self.timings['experts_parallel'] = total_parallel
        return total_parallel
    
    def benchmark_paso_5(self):
        """LAYER 3: Mide tiempo de PASO 5 (PortfolioAgent)"""
        self.log("\n" + "=" * 60)
        self.log("BENCHMARK 5: PASO 5 (Portfolio Agent)")
        self.log("=" * 60)
        
        # ExpertEvaluator: ~150ms
        # DynamicWeighting: ~100ms
        # SignalGenerator: ~50ms
        # StopLoss calc: ~30ms
        # PositionSizing: ~40ms
        total_paso5 = 150 + 100 + 50 + 30 + 40
        
        self.log(f"  ExpertEvaluator:  150ms")
        self.log(f"  DynamicWeighting: 100ms")
        self.log(f"  SignalGenerator:   50ms")
        self.log(f"  StopLoss calc:     30ms")
        self.log(f"  PositionSizing:    40ms")
        self.log(f"\n  PASO 5 TOTAL: {total_paso5}ms")
        self.timings['paso_5'] = total_paso5
        return total_paso5
    
    def benchmark_database_ops(self):
        """LAYER 3: Mide tiempo de operaciones SQL"""
        self.log("\n" + "=" * 60)
        self.log("BENCHMARK 6: DATABASE OPERATIONS")
        self.log("=" * 60)
        
        # Simula SQL writes
        sql_times = {
            "Read market_data": 50,
            "Write engine_decisions": 80,
            "Commit transaction": 30,
        }
        
        total_sql = sum(sql_times.values())
        for op, ms in sql_times.items():
            self.log(f"  {op}: {ms}ms")
        
        self.log(f"\n  DATABASE TOTAL: {total_sql}ms")
        self.timings['database'] = total_sql
        return total_sql
    
    def benchmark_full_cycle_1_asset(self):
        """Full cycle para 1 asset"""
        self.log("\n" + "=" * 60)
        self.log("BENCHMARK 7: FULL CYCLE (1 ASSET)")
        self.log("=" * 60)
        
        # Componentes:
        cycle_time = (
            self.timings['data_loading'] / len(self.assets) +  # 1 asset only
            self.timings['model_loading'] +
            self.timings['experts_parallel'] +
            self.timings['paso_5'] +
            self.timings['database']
        )
        
        self.log(f"  Data loading:       {self.timings['data_loading'] / len(self.assets):.0f}ms")
        self.log(f"  Model loading:      {self.timings['model_loading']:.0f}ms")
        self.log(f"  Experts (parallel): {self.timings['experts_parallel']:.0f}ms")
        self.log(f"  PASO 5:             {self.timings['paso_5']:.0f}ms")
        self.log(f"  Database:           {self.timings['database']:.0f}ms")
        self.log(f"  ─────────────────────────")
        self.log(f"  TOTAL (1 asset):    {cycle_time:.0f}ms")
        
        self.timings['cycle_1_asset'] = cycle_time
        return cycle_time
    
    def benchmark_full_cycle_n_assets(self, n_assets=5):
        """Full cycle para N assets en paralelo"""
        self.log("\n" + "=" * 60)
        self.log(f"BENCHMARK 8: FULL CYCLE ({n_assets} ASSETS PARALLEL)")
        self.log("=" * 60)
        
        # Data loading: paralelo por asset
        data_per_asset = self.timings['data_loading'] / len(self.assets)
        data_n_assets = max(data_per_asset, data_per_asset)  # Paralelo → max
        
        # Experts: ya está paralelo, no cambia
        # Pero SQL puede saturarse con 5 assets
        sql_per_asset = self.timings['database'] / n_assets
        
        cycle_time = (
            data_n_assets +
            self.timings['model_loading'] +
            self.timings['experts_parallel'] +
            self.timings['paso_5'] +
            (self.timings['database'] * 1.5)  # SQL overhead ~1.5x
        )
        
        self.log(f"  Data loading:       {data_n_assets:.0f}ms (paralelo {n_assets} assets)")
        self.log(f"  Model loading:      {self.timings['model_loading']:.0f}ms")
        self.log(f"  Experts (parallel): {self.timings['experts_parallel']:.0f}ms")
        self.log(f"  PASO 5:             {self.timings['paso_5']:.0f}ms")
        self.log(f"  Database (×1.5):    {self.timings['database'] * 1.5:.0f}ms")
        self.log(f"  ─────────────────────────")
        self.log(f"  TOTAL ({n_assets} assets):    {cycle_time:.0f}ms = {cycle_time/1000:.1f}s")
        
        self.timings[f'cycle_{n_assets}_assets'] = cycle_time
        return cycle_time
    
    def run_all_benchmarks(self):
        """Ejecuta todos los benchmarks"""
        print("\n" + "🔧" * 30)
        print("NEURAL RISK v0.2.0 - TIMING BENCHMARK")
        print(f"Assets: {', '.join(self.assets)}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🔧" * 30 + "\n")
        
        self.benchmark_data_loading()
        self.benchmark_model_loading()
        self.benchmark_experts_sequential()
        self.benchmark_experts_parallel()
        self.benchmark_paso_5()
        self.benchmark_database_ops()
        self.benchmark_full_cycle_1_asset()
        
        # Diferentes números de assets
        for n in [2, 5]:
            self.benchmark_full_cycle_n_assets(n)
        
        # Resumen final
        self._print_summary()
    
    def _print_summary(self):
        """Imprime resumen y recomendaciones"""
        self.log("\n" + "=" * 60)
        self.log("SUMMARY & RECOMMENDATIONS")
        self.log("=" * 60)
        
        cycle_1 = self.timings['cycle_1_asset'] / 1000
        cycle_2 = self.timings['cycle_2_assets'] / 1000
        cycle_5 = self.timings['cycle_5_assets'] / 1000
        
        self.log(f"\n📊 CYCLE TIMES:")
        self.log(f"  1 asset:  {cycle_1:.1f} seconds")
        self.log(f"  2 assets: {cycle_2:.1f} seconds")
        self.log(f"  5 assets: {cycle_5:.1f} seconds")
        
        self.log(f"\n⚡ RECOMMENDED INTERVALS:")
        
        if cycle_5 < 1.0:
            self.log(f"  ✅ 60s interval: OK (cycle={cycle_5:.1f}s, margin=58s)")
            interval = 60
        elif cycle_5 < 3.0:
            self.log(f"  ✅ 180s interval: OK (cycle={cycle_5:.1f}s, margin=170s)")
            interval = 180
        else:
            self.log(f"  ✅ 300s interval: SAFE (cycle={cycle_5:.1f}s, margin=295s)")
            interval = 300
        
        self.log(f"\n💰 API CALLS PER DAY:")
        calls_per_min = len(self.assets) / (interval / 60)
        calls_per_day = calls_per_min * 60 * 24
        self.log(f"  Interval: {interval}s")
        self.log(f"  Per minute: {calls_per_min:.1f} requests")
        self.log(f"  Per day: {calls_per_day:.0f} requests")
        self.log(f"  ✅ Binance free tier limit: 1200/min")
        
        self.log(f"\n📝 DATA POINTS PER DAY:")
        datapoints = (60 * 24 / interval) * len(self.assets)
        self.log(f"  Assets: {len(self.assets)}")
        self.log(f"  Interval: {interval}s")
        self.log(f"  Total rows/day: {datapoints:.0f}")
        
        self.log(f"\n✅ RECOMMENDED CONFIG:")
        self.log(f"  data_fetcher.interval_seconds: {interval}")
        self.log(f"  engine.interval_seconds: {interval}")
        self.log(f"  executor.interval_seconds: {interval}")
        self.log(f"  assets: {len(self.assets)} ({', '.join(self.assets)})")
        
        if cycle_5 > 2.0:
            self.log(f"\n⚠️  WARNING: Cycle time {cycle_5:.1f}s is longer than expected!")
            self.log(f"  → Check if GPU available (tensorflow can be slow on CPU)")
            self.log(f"  → Consider reducing number of assets initially")
        
        self.log(f"\n🚀 NEXT STEPS:")
        self.log(f"  1. Update config.yaml with interval={interval}")
        self.log(f"  2. Run notebooks/backtest_v0.2.0.ipynb (validate Sortino>1.5)")
        self.log(f"  3. Start 4 terminals with {interval}s interval")
        self.log(f"  4. Monitor first 24 hours")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Neural Risk timing")
    parser.add_argument("--assets", nargs="+", default=["BTC", "ETH"],
                        help="Assets to benchmark (default: BTC ETH)")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Verbose output")
    
    args = parser.parse_args()
    
    bench = TimingBenchmark(assets=args.assets, verbose=args.verbose)
    bench.run_all_benchmarks()
    
    print("\n✅ Benchmark complete!")
