# scripts/plot_backtest.py
"""
PLOT BACKTEST: Genera gráfico institucional completo de performance y riesgo.
Uso:
    python scripts/plot_backtest.py --asset BTC --data data/BTC_USD_data.csv
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Genera gráfico del backtest de Neural Risk")
    parser.add_argument("--asset", required=True, help="Ticker, ej. BTC")
    parser.add_argument("--data", required=True, help="Ruta al CSV OHLCV original")
    parser.add_argument("--output", default="", help="Nombre de archivo PNG de salida (opcional)")
    args = parser.parse_args()

    equity_file = f"backtest_equity_{args.asset}.csv"
    trades_file = f"backtest_trades_{args.asset}.csv"
    metrics_file = f"backtest_metrics_{args.asset}.json"

    if not os.path.exists(equity_file):
        raise FileNotFoundError(f"No existe {equity_file}. Corré primero: python scripts/backtest.py --asset {args.asset} --data {args.data}")

    equity_df = pd.read_csv(equity_file)
    equity_df['date'] = pd.to_datetime(equity_df['date'])
    equity_df = equity_df.sort_values('date').set_index('date')

    # Data original para Buy & Hold benchmark
    from scripts.backtest import load_ohlcv_csv
    ohlcv_df = load_ohlcv_csv(args.data)
    ohlcv_df = ohlcv_df.reindex(equity_df.index).ffill()

    # Buy & Hold normalized
    initial_equity = equity_df['equity'].iloc[0]
    bh_equity = initial_equity * (ohlcv_df['Close'] / ohlcv_df['Close'].iloc[0])

    # Trades y métricas si existen
    trades_df = pd.DataFrame()
    if os.path.exists(trades_file) and os.path.getsize(trades_file) > 0:
        try:
            trades_df = pd.read_csv(trades_file)
        except Exception:
            trades_df = pd.DataFrame()

    metrics = {}
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)

    # Configuración de estilos Matplotlib
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12), dpi=150)
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 0.8], hspace=0.35, wspace=0.25)

    # -------------------------------------------------------------------------
    # Panel 1: Equity vs Buy & Hold
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(equity_df.index, equity_df['equity'], label='Neural Risk Engine', color='#00E676', linewidth=2.2)
    ax1.plot(equity_df.index, bh_equity, label=f'Buy & Hold ({args.asset})', color='#78909C', linestyle='--', linewidth=1.5, alpha=0.8)
    
    if not trades_df.empty and 'exit_time' in trades_df.columns:
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]
        
        # Superponer puntos de trade
        win_dates = wins['exit_time'].to_numpy()
        win_equity = [equity_df.loc[d, 'equity'] if d in equity_df.index else np.nan for d in win_dates]
        loss_dates = losses['exit_time'].to_numpy()
        loss_equity = [equity_df.loc[d, 'equity'] if d in equity_df.index else np.nan for d in loss_dates]
        
        ax1.scatter(win_dates, win_equity, color='#00E676', marker='^', s=60, label='Trade Ganador', zorder=5)
        ax1.scatter(loss_dates, loss_equity, color='#FF5252', marker='v', s=60, label='Trade Perdedor', zorder=5)

    ax1.set_title(f"Performance Neural Risk vs Buy & Hold — {args.asset}", fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel("Valor de Portafolio (USD)", fontsize=11)
    ax1.legend(loc='upper left', frameon=True, facecolor='#1C2331', edgecolor='none')
    ax1.grid(True, linestyle=':', alpha=0.3)
    ax1.yaxis.set_major_formatter('${x:,.0f}')

    # -------------------------------------------------------------------------
    # Panel 2: Drawdown %
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    peak = equity_df['equity'].cummax()
    drawdown = (equity_df['equity'] - peak) / peak * 100
    ax2.fill_between(equity_df.index, drawdown, 0, color='#FF5252', alpha=0.4, label='Drawdown %')
    ax2.plot(equity_df.index, drawdown, color='#FF5252', linewidth=1)
    ax2.set_title("Drawdown Porcentual", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Drawdown %", fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.yaxis.set_major_formatter('{x:.1f}%')

    # -------------------------------------------------------------------------
    # Panel 3: Distribución de PnL por trade
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    if not trades_df.empty and 'pnl' in trades_df.columns:
        pnl_vals = trades_df['pnl'].dropna()
        colors = ['#00E676' if x > 0 else '#FF5252' for x in pnl_vals]
        ax3.bar(range(len(pnl_vals)), pnl_vals, color=colors, alpha=0.85, width=0.8)
        ax3.axhline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
        ax3.set_title(f"PnL por Trade ({len(pnl_vals)} trades cerrados)", fontsize=11, fontweight='bold')
        ax3.set_xlabel("Número de Trade", fontsize=10)
        ax3.set_ylabel("PnL (USD)", fontsize=10)
        ax3.grid(True, linestyle=':', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Sin trades cerrados", ha='center', va='center', color='gray')

    # -------------------------------------------------------------------------
    # Panel 4: Resumen de Métricas
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    if metrics:
        metric_items = [
            f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}",
            f"Sortino Ratio: {metrics.get('sortino_ratio', 'N/A')}",
            f"Max Drawdown: {metrics.get('max_drawdown', 'N/A')}",
            f"Win Rate: {metrics.get('win_rate', 'N/A')}",
            f"Profit Factor: {metrics.get('profit_factor', 'N/A')}",
            f"Total Trades: {metrics.get('total_trades', 'N/A')}",
            f"VaR 95%: {metrics.get('var_95', 'N/A')}",
            f"CVaR 95%: {metrics.get('cvar_95', 'N/A')}"
        ]
        
        # Formatear como tabla en cuadro
        text_content = " | ".join(metric_items[:4]) + "\n" + " | ".join(metric_items[4:])
        ax4.text(0.5, 0.5, text_content, ha='center', va='center', fontsize=11,
                 fontfamily='monospace', bbox=dict(boxstyle='round,pad=1', facecolor='#1C2331', edgecolor='#00E676', alpha=0.9))

    out_file = args.output if args.output else f"backtest_chart_{args.asset}.png"
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    plt.close()
    print(f"Chart guardado exitosamente en: {out_file}")


if __name__ == "__main__":
    main()
