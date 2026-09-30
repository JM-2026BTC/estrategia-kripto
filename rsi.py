import json
import ccxt
import pandas as pd
import numpy as np

# Configuración
CRIPTOS = ['BTC/USDT', 'HYPE/USDT', 'DOGE/USDT', 'PEPE/USDT']
TEMPORALIDADES = ['1h', '4h', '1d', '1w']
RSI_PERIODO = 14
FVG_TEMPORALIDADES = ['1h', '4h', '1d']
SR_TEMPORALIDADES = ['1h', '4h', '1d', '1w']

# Configuración por temporalidad
VELAS_POR_TF = {
    '1h': 100,
    '4h': 100,
    '1d': 300,
    '1w': 200
}

FVG_MAX_DISTANCIA_PCT = {
    '1h': 5.0,
    '4h': 5.0,
    '1d': 15.0,
    '1w': 25.0
}

FVG_MIN_TAMANO_PCT = 0.15

# Configuración de S/R
SR_SWING_STRENGTH = 3
SR_CLUSTER_PCT = 0.5
SR_MIN_TOQUES = 2
SR_MAX_ZONAS = 4


def redondear(valor, precio_actual):
    """Redondea según la magnitud del precio."""
    if precio_actual >= 1000:
        return round(valor, 2)
    elif precio_actual >= 1:
        return round(valor, 4)
    elif precio_actual >= 0.001:
        return round(valor, 6)
    else:
        return round(valor, 8)


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
    """Detecta FVG."""
    fvgs = []
    precio_actual = float(df['close'].iloc[-1])
    for k in range(2, len(df)):
        high_2 = float(df['high'].iloc[k-2])
        low_2 = float(df['low'].iloc[k-2])
        high_0 = float(df['high'].iloc[k])
        low_0 = float(df['low'].iloc[k])

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


def filtrar_fvgs(fvgs, precio_actual, timeframe):
    filtrados = []
    distancia_max = FVG_MAX_DISTANCIA_PCT.get(timeframe, 5.0)
    for fvg in fvgs:
        if fvg.get('mitigado', False):
            continue
        desde = fvg['desde']
        hasta = fvg['hasta']
        tamano = abs(hasta - desde)
        tamano_pct = (tamano / precio_actual) * 100 if precio_actual > 0 else 0
        if tamano_pct < FVG_MIN_TAMANO_PCT:
            continue
        distancia = min(abs(precio_actual - desde), abs(precio_actual - hasta))
        distancia_pct = (distancia / precio_actual) * 100 if precio_actual > 0 else 0
        if distancia_pct > distancia_max:
            continue
        filtrados.append(fvg)
    return filtrados


def detectar_pivotes(df, strength=3):
    """Detecta swing highs y swing lows."""
    pivotes_altos = []
    pivotes_bajos = []
    highs = df['high'].values
    lows = df['low'].values

    for i in range(strength, len(df) - strength):
        # Swing high
        es_alto = True
        for j in range(1, strength + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                es_alto = False
                break
        if es_alto:
            pivotes_altos.append({'precio': float(highs[i]), 'indice': i})

        # Swing low
        es_bajo = True
        for j in range(1, strength + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                es_bajo = False
                break
        if es_bajo:
            pivotes_bajos.append({'precio': float(lows[i]), 'indice': i})

    return pivotes_altos, pivotes_bajos


def agrupar_pivotes(pivotes, precio_actual):
    """Agrupa pivotes cercanos en zonas."""
    if not pivotes:
        return []
    
    # Ordenar por precio
    pivotes_ord = sorted(pivotes, key=lambda x: x['precio'])
    
    zonas = []
    grupo_actual = [pivotes_ord[0]]
    
    for i in range(1, len(pivotes_ord)):
        precio_actual_pivote = pivotes_ord[i]['precio']
        precio_ultimo_grupo = grupo_actual[-1]['precio']
        
        distancia_pct = abs(precio_actual_pivote - precio_ultimo_grupo) / precio_ultimo_grupo * 100
        
        if distancia_pct <= SR_CLUSTER_PCT:
            grupo_actual.append(pivotes_ord[i])
        else:
            # Cerrar grupo anterior
            if len(grupo_actual) >= SR_MIN_TOQUES:
                precios = [p['precio'] for p in grupo_actual]
                zonas.append({
                    'precio': redondear(sum(precios) / len(precios), precio_actual),
                    'toques': len(grupo_actual)
                })
            grupo_actual = [pivotes_ord[i]]
    
    # Cerrar último grupo
    if len(grupo_actual) >= SR_MIN_TOQUES:
        precios = [p['precio'] for p in grupo_actual]
        zonas.append({
            'precio': redondear(sum(precios) / len(precios), precio_actual),
            'toques': len(grupo_actual)
        })
    
    return zonas


def calcular_sr_par(exchange, symbol, timeframe):
    """Calcula S/R para un par y temporalidad."""
    try:
        limit = VELAS_POR_TF.get(timeframe, 100)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)

        precio_actual = float(df['close'].iloc[-1])
        
        pivotes_altos, pivotes_bajos = detectar_pivotes(df, SR_SWING_STRENGTH)
        
        # Resistencias: pivotes altos por encima del precio actual
        altos_arriba = [p for p in pivotes_altos if p['precio'] > precio_actual]
        resistencias = agrupar_pivotes(altos_arriba, precio_actual)
        resistencias = sorted(resistencias, key=lambda x: x['precio'])[:SR_MAX_ZONAS]
        
        # Soportes: pivotes bajos por debajo del precio actual
        bajos_abajo = [p for p in pivotes_bajos if p['precio'] < precio_actual]
        soportes = agrupar_pivotes(bajos_abajo, precio_actual)
        soportes = sorted(soportes, key=lambda x: x['precio'], reverse=True)[:SR_MAX_ZONAS]
        
        return {
            'precio_actual': redondear(precio_actual, precio_actual),
            'resistencias': resistencias,
            'soportes': soportes
        }
    except Exception as e:
        print(f"Error calculando S/R para {symbol} en {timeframe}: {e}")
        return {'precio_actual': None, 'resistencias': [], 'soportes': []}


def calcular_rsi_par(exchange, symbol, timeframe):
    try:
        limit = VELAS_POR_TF.get(timeframe, 100)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        rsi = calcular_rsi(df['close'], RSI_PERIODO)
        return round(float(rsi.iloc[-1]), 2)
    except Exception as e:
        print(f"Error calculando RSI para {symbol} en {timeframe}: {e}")
        return None


def calcular_fvgs_par(exchange, symbol, timeframe):
    try:
        limit = VELAS_POR_TF.get(timeframe, 100)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        precio_actual = float(df['close'].iloc[-1])
        fvgs = detectar_fvg(df)
        return filtrar_fvgs(fvgs, precio_actual, timeframe)
    except Exception as e:
        print(f"Error calculando FVG para {symbol} en {timeframe}: {e}")
        return []


def main():
    exchange = ccxt.okx()

    rsi_resultado = {'last_update': pd.Timestamp.now(tz='UTC').isoformat(), 'data': {}}
    fvg_resultado = {'last_update': pd.Timestamp.now(tz='UTC').isoformat(), 'data': {}}
    sr_resultado = {'last_update': pd.Timestamp.now(tz='UTC').isoformat(), 'data': {}}

    for cripto in CRIPTOS:
        symbol = cripto.replace('/', '')
        rsi_resultado['data'][symbol] = {}
        fvg_resultado['data'][symbol] = {}
        sr_resultado['data'][symbol] = {}

        for tf in TEMPORALIDADES:
            rsi = calcular_rsi_par(exchange, cripto, tf)
            rsi_resultado['data'][symbol][tf] = rsi
            print(f"{symbol} en {tf}: RSI = {rsi}")

        for tf in FVG_TEMPORALIDADES:
            fvgs = calcular_fvgs_par(exchange, cripto, tf)
            fvg_resultado['data'][symbol][tf] = fvgs
            print(f"{symbol} en {tf}: {len(fvgs)} FVG activos")

        for tf in SR_TEMPORALIDADES:
            sr = calcular_sr_par(exchange, cripto, tf)
            sr_resultado['data'][symbol][tf] = sr
            print(f"{symbol} en {tf}: {len(sr['resistencias'])} R / {len(sr['soportes'])} S")

    with open('rsi_data.json', 'w') as f:
        json.dump(rsi_resultado, f, indent=2)
    with open('fvg_data.json', 'w') as f:
        json.dump(fvg_resultado, f, indent=2)
    with open('sr_data.json', 'w') as f:
        json.dump(sr_resultado, f, indent=2)

    print("\n✅ Datos guardados en rsi_data.json, fvg_data.json y sr_data.json")


if __name__ == '__main__':
    main()
