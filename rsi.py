import json
import ccxt
import pandas as pd
import pandas_ta as ta

# Configuración
CRIPTOS = ['BTC/USDT', 'HYPE/USDT', 'DOGE/USDT', '1000PEPE/USDT']
TEMPORALIDADES = ['1h', '4h', '1d', '1w']
RSI_PERIODO = 14

def calcular_rsi(exchange, symbol, timeframe):
    """Calcula el RSI para un par y temporalidad específicos."""
    try:
        # Bajar velas (necesitamos al menos 100 para que el RSI sea preciso)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        
        # Convertir a DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        
        # Calcular RSI
        rsi = ta.rsi(df['close'], length=RSI_PERIODO)
        
        # Obtener el último valor (el más reciente)
        ultimo_rsi = round(float(rsi.iloc[-1]), 2)
        return ultimo_rsi
    
    except Exception as e:
        print(f"Error calculando RSI para {symbol} en {timeframe}: {e}")
        return None

def main():
    # Usar Binance Futures (USD-M)
    exchange = ccxt.binanceusdm()
    
    # Estructura de datos
    resultado = {
        'last_update': pd.Timestamp.now(tz='UTC').isoformat(),
        'data': {}
    }
    
    # Calcular RSI para cada cripto y temporalidad
    for cripto in CRIPTOS:
        symbol = cripto.replace('/', '')  # BTC/USDT → BTCUSDT
        resultado['data'][symbol] = {}
        
        for tf in TEMPORALIDADES:
            rsi = calcular_rsi(exchange, cripto, tf)
            resultado['data'][symbol][tf] = rsi
            print(f"{symbol} en {tf}: RSI = {rsi}")
    
    # Guardar en JSON
    with open('rsi_data.json', 'w') as f:
        json.dump(resultado, f, indent=2)
    
    print("\n✅ Datos guardados en rsi_data.json")

if __name__ == '__main__':
    main()
