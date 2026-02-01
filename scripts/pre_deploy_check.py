#!/usr/bin/env python3
"""
PRE-DEPLOY VALIDATION SCRIPT
Valida automáticamente que todo está listo para deploy
Ejecuta: python scripts/pre_deploy_check.py
"""

import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime
from colorama import Fore, Style, init

# Init colorama for Windows console colors
init(autoreset=True, convert=True)  # convert=True para Windows

class PreDeployChecker:
    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        self.errors = []
        
    def print_header(self, title):
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}{title:^60}")
        print(f"{Fore.CYAN}{'='*60}")
    
    def ok(self, msg):
        self.checks_passed += 1
        print(f"{Fore.GREEN}[OK] {msg}")
    
    def warn(self, msg):
        print(f"{Fore.YELLOW}[WARN] {msg}")
        self.warnings.append(msg)
    
    def fail(self, msg):
        self.checks_failed += 1
        print(f"{Fore.RED}[FAIL] {msg}")
        self.errors.append(msg)
    
    def check_python_version(self):
        """Check Python 3.9+"""
        self.print_header("PHASE 0: PYTHON & DEPENDENCIES")
        
        version = sys.version_info
        if version.major >= 3 and version.minor >= 9:
            self.ok(f"Python {version.major}.{version.minor} (>= 3.9)")
        else:
            self.fail(f"Python {version.major}.{version.minor} (need >= 3.9)")
    
    def check_dependencies(self):
        """Check critical packages"""
        deps = {
            'numpy': 'Core array',
            'pandas': 'Data frames',
            'sklearn': 'ML basics',
            'xgboost': 'Gradient boosting',
            'hmmlearn': 'Hidden Markov Models',
            'arch': 'GARCH models',
        }
        
        for dep, desc in deps.items():
            try:
                __import__(dep)
                self.ok(f"{dep:15} - {desc}")
            except ImportError:
                self.fail(f"{dep:15} - {desc} (missing)")
    
    def check_neural_risk_imports(self):
        """Check neural_risk package imports"""
        self.print_header("PHASE 1: NEURAL_RISK IMPORTS")
        
        try:
            from neural_risk.agents.portfolio_agent import PortfolioAgent
            self.ok("PortfolioAgent imports")
        except Exception as e:
            self.fail(f"PortfolioAgent: {str(e)[:50]}")
        
        try:
            from neural_risk.models.ensemble_trainer import EnsembleTrainer
            self.ok("EnsembleTrainer imports")
        except Exception as e:
            self.fail(f"EnsembleTrainer: {str(e)[:50]}")
        
        try:
            from neural_risk.data.data_processor import DataProcessor
            self.ok("DataProcessor imports")
        except Exception as e:
            self.fail(f"DataProcessor: {str(e)[:50]}")
    
    def check_directory_structure(self):
        """Check required directories"""
        self.print_header("PHASE 2: DIRECTORY STRUCTURE")
        
        required = {
            'data': 'Data storage',
            'logs': 'Logging',
            'scripts': 'Service scripts',
            'config': 'Configuration',
            'notebooks': 'Jupyter notebooks',
        }
        
        for dir_name, desc in required.items():
            if Path(dir_name).exists():
                self.ok(f"{dir_name:15} - {desc}")
            else:
                self.fail(f"{dir_name:15} - {desc} (missing)")
    
    def check_required_files(self):
        """Check required files"""
        self.print_header("PHASE 3: REQUIRED FILES")
        
        files = {
            'config/config.yaml': 'Config',
            'scripts/run_data_fetcher.py': 'Data fetcher',
            'scripts/run_engine.py': 'Engine',
            'scripts/run_executor.py': 'Executor',
            'scripts/train_models.py': 'Training',
            'scripts/benchmark_timing.py': 'Benchmark',
            'notebooks/live_dashboard.ipynb': 'Dashboard',
            'notebooks/backtest_v0.2.0.ipynb': 'Backtest',
            'data/BTC_USD_data.csv': 'BTC historical',
            'data/ETH_USD_data.csv': 'ETH historical',
        }
        
        for file_path, desc in files.items():
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
                if file_size > 0:
                    self.ok(f"{file_path:30} ({file_size/1024:.0f}KB) - {desc}")
                else:
                    self.warn(f"{file_path:30} (empty!) - {desc}")
            else:
                self.fail(f"{file_path:30} (missing) - {desc}")
    
    def check_config(self):
        """Check config.yaml values"""
        self.print_header("PHASE 4: CONFIGURATION")
        
        try:
            import yaml
            with open('config/config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Check mode
            mode = config.get('mode', 'unknown')
            if mode == 'production':
                self.ok(f"Mode: {mode} ✅")
            elif mode == 'development':
                self.warn(f"Mode: {mode} (OK for testing)")
            else:
                self.warn(f"Mode: {mode} (unknown)")
            
            # Check assets
            assets = config.get('exchanges', {}).get('assets', [])
            self.ok(f"Assets configured: {assets}")
            
            # Check intervals
            timing = config.get('cycle_timing', {})
            fetcher_int = timing.get('data_fetcher_interval', 300)
            engine_int = timing.get('engine_interval', 300)
            executor_int = timing.get('executor_interval', 300)
            
            if fetcher_int >= 60 and engine_int >= 60 and executor_int >= 60:
                self.ok(f"Intervals: {fetcher_int}s (data), {engine_int}s (engine), {executor_int}s (executor)")
            else:
                self.warn(f"Intervals very tight: {fetcher_int}s, {engine_int}s, {executor_int}s")
            
            # Check thresholds
            signals = config.get('signals', {})
            long_th = signals.get('long_threshold', 0.60)
            short_th = signals.get('short_threshold', -0.60)
            self.ok(f"Signal thresholds: LONG>{long_th}, SHORT<{short_th}")
            
        except Exception as e:
            self.fail(f"Config error: {str(e)[:50]}")
    
    def check_permissions(self):
        """Check write permissions"""
        self.print_header("PHASE 5: PERMISSIONS")
        
        dirs_to_write = ['data', 'logs']
        for dir_name in dirs_to_write:
            path = Path(dir_name)
            if not path.exists():
                path.mkdir(parents=True)
                self.ok(f"Created {dir_name}/")
            
            try:
                test_file = path / '.write_test'
                test_file.write_text('test')
                test_file.unlink()
                self.ok(f"{dir_name}/ is writable")
            except Exception as e:
                self.fail(f"{dir_name}/ write error: {str(e)[:30]}")
    
    def check_database(self):
        """Check database"""
        self.print_header("PHASE 6: DATABASE")
        
        db_path = 'data/neural_risk.db'
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check tables exist or will be created
            tables = ['market_data', 'engine_decisions', 'orders', 'fills']
            for table in tables:
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                if cursor.fetchone():
                    self.ok(f"Table {table} exists")
                else:
                    self.warn(f"Table {table} will be created on first run")
            
            conn.close()
        except Exception as e:
            self.warn(f"Database check: {str(e)[:50]} (will be created on first run)")
    
    def check_models_cache(self):
        """Check trained models cache"""
        self.print_header("PHASE 7: TRAINED MODELS")
        
        model_file = Path('data/trained_models.pkl')
        if model_file.exists():
            size_mb = model_file.stat().st_size / (1024*1024)
            mtime = datetime.fromtimestamp(model_file.stat().st_mtime)
            age_hours = (datetime.now() - mtime).total_seconds() / 3600
            
            if age_hours < 24:
                self.ok(f"Models cache OK ({size_mb:.1f}MB, {age_hours:.1f}h old)")
            elif age_hours < 168:  # 7 days
                self.warn(f"Models cache old ({size_mb:.1f}MB, {age_hours:.1f}h old) - consider retraining")
            else:
                self.warn(f"Models cache very old ({size_mb:.1f}MB, {age_hours:.1f}h old) - RUN train_models.py")
        else:
            self.warn(f"No models cache found - RUN: python scripts/train_models.py")
    
    def check_env_vars(self):
        """Check environment variables"""
        self.print_header("PHASE 8: API KEYS")
        
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if api_key and len(api_key) > 10:
            self.ok(f"BINANCE_API_KEY configured ({len(api_key)} chars)")
        else:
            self.warn(f"BINANCE_API_KEY not set (MOCK mode will be used)")
        
        if api_secret and len(api_secret) > 10:
            self.ok(f"BINANCE_API_SECRET configured ({len(api_secret)} chars)")
        else:
            self.warn(f"BINANCE_API_SECRET not set (MOCK mode will be used)")
    
    def print_summary(self):
        """Print summary"""
        self.print_header("SUMMARY")
        
        total = self.checks_passed + self.checks_failed
        pct = (self.checks_passed / total * 100) if total > 0 else 0
        
        print(f"\n{Fore.CYAN}Checks passed: {self.checks_passed}/{total} ({pct:.0f}%)")
        
        if self.warnings:
            print(f"\n{Fore.YELLOW}Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"  [WARN] {w}")
        
        if self.errors:
            print(f"\n{Fore.RED}Errors ({len(self.errors)}):")
            for e in self.errors:
                print(f"  [FAIL] {e}")
        
        print(f"\n{Fore.CYAN}{'='*60}")
        
        if self.checks_failed == 0 and len(self.errors) == 0:
            print(f"{Fore.GREEN}[OK] PRE-DEPLOY CHECK PASSED - READY TO DEPLOY!")
            return True
        elif self.checks_failed <= 2:
            print(f"{Fore.YELLOW}[WARN] PRE-DEPLOY CHECK: FIX ERRORS ABOVE BEFORE DEPLOYING")
            return False
        else:
            print(f"{Fore.RED}[FAIL] PRE-DEPLOY CHECK FAILED - FIX ERRORS ABOVE")
            return False
    
    def run_all_checks(self):
        """Run all checks"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}{'NEURAL RISK PRE-DEPLOY VALIDATOR':^60}")
        print(f"{Fore.CYAN}Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{Fore.CYAN}{'='*60}")
        
        self.check_python_version()
        self.check_dependencies()
        self.check_neural_risk_imports()
        self.check_directory_structure()
        self.check_required_files()
        self.check_config()
        self.check_permissions()
        self.check_database()
        self.check_models_cache()
        self.check_env_vars()
        
        return self.print_summary()

if __name__ == "__main__":
    checker = PreDeployChecker()
    ready = checker.run_all_checks()
    sys.exit(0 if ready else 1)
