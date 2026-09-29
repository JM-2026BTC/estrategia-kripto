import json
import ccxt
import pandas as pd
import numpy as np

# Configuración
CRIPTOS = ['BTC/USDT', 'HYPE/USDT', 'DOGE/USDT', '1000PEPE/USDT']
TEMPORALIDADES = ['1h', '4h', '1d', '1w']
RSI_PERIODO = 14

def calcular_rsi(close, periodo=14):
    """Calcula el RSI manualmente (método de Wilder)."""
    delta = close.diff()
    ganancia = delta.where(delta > 0, 0)
    perdida = -delta.where(delta < 0, 0)
    
    # Media móvil exponencial de Wilder
    avg_ganancia = ganancia.ewm(alpha=1/periodo, min_periods=periodo).mean()
    avg_perdida = perdida.ewm(alpha=1/periodo, min_periods=periodo).mean()
    
    rs = avg_ganancia / avg_perdida
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calcular_rsi_par(exchange, symbol, timeframe):
    """Calcula el RSI para un par y temporalidad específicos."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        
        rsi = calcular_rsi(df['close'], RSI_PERIODO)
        ultimo_rsi = round(float(rsi.iloc[-1]), 2)
        return ultimo_rsi
    
    except Exception as e:
        print(f"Error calculando RSI para {symbol} en {timeframe}: {e}")
        return None

def main():
    exchange = ccxt.binanceusdm()
    
    resultado = {
        'last_update': pd.Timestamp.now(tz='UTC').isoformat(),
        'data': {}
    }
    
    for cripto in CRIPTOS:
        symbol = cripto.replace('/', '')
        resultado['data'][symbol] = {}
        
        for tf in TEMPORALIDADES:
            rsi = calcular_rsi_par(exchange, cripto, tf)
            resultado['data'][symbol][tf] = rsi
            print(f"{symbol} en {tf}: RSI = {rsi}")
    
    with open('rsi_data.json', 'w') as f:
        json.dump(resultado, f, indent=2)
    
    print("\n✅ Datos guardados en rsi_data.json")

if __name__ == '__main__':
    main()
