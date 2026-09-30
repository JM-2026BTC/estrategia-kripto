import json
import ccxt
import pandas as pd
import numpy as np

# Configuración
CRIPTOS = ['BTC/USDT', 'HYPE/USDT', 'DOGE/USDT', 'PEPE/USDT']
TEMPORALIDADES = ['1h', '4h', '1d', '1w']
RSI_PERIODO = 14
FVG_TEMPORALIDADES = ['1h', '4h', '1d']
FVG_MIN_TAMANO_PCT = 0.15  # Bajado de 0.2 a 0.15
FVG_MAX_DISTANCIA_PCT = 5.0


def redondear(valor, precio_actual):
    """Redondea según la magnitud del precio."""
    if precio_actual >= 1000:
        return round(valor, 2)      # BTC: 2 decimales
    elif precio_actual >= 1:
        return round(valor, 4)      # HYPE, DOGE: 4 decimales
    elif precio_actual >= 0.001:
        return round(valor, 6)      # PEPE: 6 decimales
    else:
        return round(valor, 8)      # PEPE muy chico: 8 decimales


def calcular_rsi(close, periodo=14):
    """Calcula el RSI manualmente (método de Wilder)."""
    delta = close.diff()
    ganancia = delta.where(delta > 0, 0)
    perdida = -delta.where(delta < 0, 0)

    avg_ganancia = ganancia.ewm(alpha=1/periodo, min_periods=periodo).mean()
    avg_perdida = perdida.ewm(alpha=1/periodo, min_periods=periodo).mean()

    rs = avg_ganancia / avg_perdida
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detectar_fvg(df):
    """Detecta FVG y marca si están mitigados o activos.
    
    Lógica: un FVG se considera mitigado solo si el precio
    CRUZA COMPLETAMENTE la zona.
    """
    fvgs = []
    precio_actual = float(df['close'].iloc[-1])

    for k in range(2, len(df)):
        high_2 = float(df['high'].iloc[k-2])
        low_2 = float(df['low'].iloc[k-2])
        high_0 = float(df['high'].iloc[k])
        low_0 = float(df['low'].iloc[k])

        # FVG Alcista
        if low_0 > high_2:
            desde = high_2
            hasta = low_0
            mitigado = False
            for j in range(k+1, len(df)):
                if float(df['low'].iloc[j]) <= desde:
                    mitigado = True
                    break
            fvgs.append({
                'tipo': 'alcista',
                'desde': redondear(desde, precio_actual),
                'hasta': redondear(hasta, precio_actual),
                'mitigado': mitigado,
                'timestamp': int(df.index[k]) if hasattr(df.index[k], '__int__') else str(df.index[k])
            })

        # FVG Bajista
        if high_0 < low_2:
            desde = high_0
            hasta = low_2
            mitigado = False
            for j in range(k+1, len(df)):
                if float(df['high'].iloc[j]) >= hasta:
                    mitigado = True
                    break
            fvgs.append({
                'tipo': 'bajista',
                'desde': redondear(desde, precio_actual),
                'hasta': redondear(hasta, precio_actual),
                'mitigado': mitigado,
                'timestamp': int(df.index[k]) if hasattr(df.index[k], '__int__') else str(df.index[k])
            })

    return fvgs


def filtrar_fvgs(fvgs, precio_actual):
    """Filtra solo los FVG activos (no mitigados) y relevantes."""
    filtrados = []
    for fvg in fvgs:
        if fvg.get('mitigado', False):
            continue

        desde = fvg['desde']
        hasta = fvg['hasta']
        tamano = abs(hasta - desde)
        tamano_pct = (tamano / precio_actual) * 100

        if tamano_pct < FVG_MIN_TAMANO_PCT:
            continue

        distancia = min(abs(precio_actual - desde), abs(precio_actual - hasta))
        distancia_pct = (distancia / precio_actual) * 100
        if distancia_pct > FVG_MAX_DISTANCIA_PCT:
            continue

        filtrados.append(fvg)

    return filtrados


def calcular_rsi_par(exchange, symbol, timeframe):
    """Calcula el RSI para un par y temporalidad específicos."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)

        rsi = calcular_rsi(df['close'], RSI_PERIODO)
        ultimo_rsi = round(float(rsi.iloc[-1]), 2)
        return ultimo_rsi

    except Exception as e:
        print(f"Error calculando RSI para {symbol} en {timeframe}: {e}")
        return None


def calcular_fvgs_par(exchange, symbol, timeframe):
    """Calcula los FVG activos para un par y temporalidad específicos."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)

        precio_actual = float(df['close'].iloc[-1])
        fvgs = detectar_fvg(df)
        fvgs_filtrados = filtrar_fvgs(fvgs, precio_actual)
        return fvgs_filtrados

    except Exception as e:
        print(f"Error calculando FVG para {symbol} en {timeframe}: {e}")
        return []


def main():
    exchange = ccxt.okx()

    rsi_resultado = {
        'last_update': pd.Timestamp.now(tz='UTC').isoformat(),
        'data': {}
    }

    fvg_resultado = {
        'last_update': pd.Timestamp.now(tz='UTC').isoformat(),
        'data': {}
    }

    for cripto in CRIPTOS:
        symbol = cripto.replace('/', '')
        rsi_resultado['data'][symbol] = {}
        fvg_resultado['data'][symbol] = {}

        for tf in TEMPORALIDADES:
            rsi = calcular_rsi_par(exchange, cripto, tf)
            rsi_resultado['data'][symbol][tf] = rsi
            print(f"{symbol} en {tf}: RSI = {rsi}")

        for tf in FVG_TEMPORALIDADES:
            fvgs = calcular_fvgs_par(exchange, cripto, tf)
            fvg_resultado['data'][symbol][tf] = fvgs
            print(f"{symbol} en {tf}: {len(fvgs)} FVG activos detectados")

    with open('rsi_data.json', 'w') as f:
        json.dump(rsi_resultado, f, indent=2)

    with open('fvg_data.json', 'w') as f:
        json.dump(fvg_resultado, f, indent=2)

    print("\n✅ Datos guardados en rsi_data.json y fvg_data.json")


if __name__ == '__main__':
    main()
