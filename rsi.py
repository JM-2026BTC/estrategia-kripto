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
GP_TEMPORALIDADES = ['1h', '4h', '1d', '1w']

# Configuración por temporalidad (AJUSTADA)
VELAS_POR_TF = {
    '1h': 300,   # ← Antes 100
    '4h': 300,   # ← Antes 200
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

# Configuración de S/R (AJUSTADA)
SWING_STRENGTH = {
    '1h': 2,   # ← Antes 3
    '4h': 3,   # ← Antes 3 (igual)
    '1d': 4,
    '1w': 5
}

SR_MIN_TOQUES = {
    'BTCUSDT': 3,
    'HYPEUSDT': 2,
    'DOGEUSDT': 2,
    'PEPEUSDT': 2
}

SR_MAX_ZONAS = 4


def redondear(valor, precio_actual):
    if precio_actual >= 1000:
        return round(valor, 2)
    elif precio_actual >= 1:
        return round(valor, 3)
    elif precio_actual >= 0.01:
        return round(valor, 5)
    elif precio_actual >= 0.0001:
        return round(valor, 8)
    else:
        return round(valor, 10)


def calcular_cluster_pct(df):
    try:
        rango = (df['high'] - df['low']).tail(14).mean()
        precio = df['close'].iloc[-1]
        atr_pct = (rango / precio) * 100
        return max(atr_pct * 0.4, 0.3)
    except:
        return 0.5


def calcular_rsi(close, periodo=14):
    delta = close.diff()
    ganancia = delta.where(delta > 0, 0)
    perdida = -delta.where(delta < 0, 0)
    avg_ganancia = ganancia.ewm(alpha=1/periodo, min_periods=periodo).mean()
    avg_perdida = perdida.ewm(alpha=1/periodo, min_periods=periodo).mean()
    rs = avg_ganancia / avg_perdida
    rsi = 100 - (100 / (1 + rs))
    return rsi


def detectar_fvg(df):
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


def detectar_pivotes(df, strength):
    pivotes_altos = []
    pivotes_bajos = []
    highs = df['high'].values
    lows = df['low'].values

    for i in range(strength, len(df) - strength):
        es_alto = True
        for j in range(1, strength + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                es_alto = False
                break
        if es_alto:
            pivotes_altos.append({'precio': float(highs[i]), 'indice': i})

        es_bajo = True
        for j in range(1, strength + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                es_bajo = False
                break
        if es_bajo:
            pivotes_bajos.append({'precio': float(lows[i]), 'indice': i})

    return pivotes_altos, pivotes_bajos


def agrupar_pivotes(pivotes, precio_actual, cluster_pct, min_toques):
    if not pivotes:
        return []
    pivotes_ord = sorted(pivotes, key=lambda x: x['precio'])
    zonas = []
    grupo_actual = [pivotes_ord[0]]
    for i in range(1, len(pivotes_ord)):
        precio_actual_pivote = pivotes_ord[i]['precio']
        precio_ultimo_grupo = grupo_actual[-1]['precio']
        distancia_pct = abs(precio_actual_pivote - precio_ultimo_grupo) / precio_ultimo_grupo * 100
        if distancia_pct <= cluster_pct:
            grupo_actual.append(pivotes_ord[i])
        else:
            if len(grupo_actual) >= min_toques:
                precios = [p['precio'] for p in grupo_actual]
                zonas.append({
                    'precio': redondear(sum(precios) / len(precios), precio_actual),
                    'toques': len(grupo_actual)
                })
            grupo_actual = [pivotes_ord[i]]
    if len(grupo_actual) >= min_toques:
        precios = [p['precio'] for p in grupo_actual]
        zonas.append({
            'precio': redondear(sum(precios) / len(precios), precio_actual),
            'toques': len(grupo_actual)
        })
    return zonas


def calcular_sr_par(exchange, symbol, timeframe):
    try:
        limit = VELAS_POR_TF.get(timeframe, 100)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)

        precio_actual = float(df['close'].iloc[-1])
        symbol_limpio = symbol.replace('/', '')
        strength = SWING_STRENGTH.get(timeframe, 3)
        cluster_pct = calcular_cluster_pct(df)
        min_toques = SR_MIN_TOQUES.get(symbol_limpio, 2)

        pivotes_altos, pivotes_bajos = detectar_pivotes(df, strength)

        altos_arriba = [p for p in pivotes_altos if p['precio'] > precio_actual]
        resistencias = agrupar_pivotes(altos_arriba, precio_actual, cluster_pct, min_toques)
        resistencias = sorted(resistencias, key=lambda x: x['precio'])[:SR_MAX_ZONAS]

        bajos_abajo = [p for p in pivotes_bajos if p['precio'] < precio_actual]
        soportes = agrupar_pivotes(bajos_abajo, precio_actual, cluster_pct, min_toques)
        soportes = sorted(soportes, key=lambda x: x['precio'], reverse=True)[:SR_MAX_ZONAS]

        return {
            'precio_actual': redondear(precio_actual, precio_actual),
            'cluster_usado': round(cluster_pct, 2),
            'strength_usado': strength,
            'min_toques_usado': min_toques,
            'resistencias': resistencias,
            'soportes': soportes
        }
    except Exception as e:
        print(f"Error calculando S/R para {symbol} en {timeframe}: {e}")
        return {'precio_actual': None, 'resistencias': [], 'soportes': []}


def calcular_golden_pocket_par(exchange, symbol, timeframe):
    try:
        limit = VELAS_POR_TF.get(timeframe, 100)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)

        precio_actual = float(df['close'].iloc[-1])
        strength = SWING_STRENGTH.get(timeframe, 3)

        pivotes_altos, pivotes_bajos = detectar_pivotes(df, strength)

        if not pivotes_altos or not pivotes_bajos:
            return None

        ultimo_alto = pivotes_altos[-1]
        ultimo_bajo = pivotes_bajos[-1]

        if ultimo_alto['indice'] > ultimo_bajo['indice']:
            tipo = 'alcista'
            swing_low = ultimo_bajo['precio']
            swing_high = ultimo_alto['precio']
            rango = swing_high - swing_low
            nivel_05 = swing_low + rango * 0.5
            nivel_0618 = swing_low + rango * 0.618
            zona_desde = min(nivel_05, nivel_0618)
            zona_hasta = max(nivel_05, nivel_0618)
            accion = 'SOPORTE'
        else:
            tipo = 'bajista'
            swing_high = ultimo_alto['precio']
            swing_low = ultimo_bajo['precio']
            rango = swing_high - swing_low
            nivel_05 = swing_high - rango * 0.5
            nivel_0618 = swing_high - rango * 0.618
            zona_desde = min(nivel_05, nivel_0618)
            zona_hasta = max(nivel_05, nivel_0618)
            accion = 'RESISTENCIA'

        precio_sugerido = (zona_desde + zona_hasta) / 2

        return {
            'tipo': tipo,
            'accion': accion,
            'swing_low': redondear(swing_low, precio_actual),
            'swing_high': redondear(swing_high, precio_actual),
            'nivel_05': redondear(nivel_05, precio_actual),
            'nivel_0618': redondear(nivel_0618, precio_actual),
            'zona_desde': redondear(zona_desde, precio_actual),
            'zona_hasta': redondear(zona_hasta, precio_actual),
            'precio_sugerido': redondear(precio_sugerido, precio_actual)
        }
    except Exception as e:
        print(f"Error calculando Golden Pocket para {symbol} en {timeframe}: {e}")
        return None


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
    gp_resultado = {'last_update': pd.Timestamp.now(tz='UTC').isoformat(), 'data': {}}

    for cripto in CRIPTOS:
        symbol = cripto.replace('/', '')
        rsi_resultado['data'][symbol] = {}
        fvg_resultado['data'][symbol] = {}
        sr_resultado['data'][symbol] = {}
        gp_resultado['data'][symbol] = {}

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

        for tf in GP_TEMPORALIDADES:
            gp = calcular_golden_pocket_par(exchange, cripto, tf)
            gp_resultado['data'][symbol][tf] = gp
            if gp:
                print(f"{symbol} en {tf}: GP {gp['tipo']} ({gp['accion']}) zona {gp['zona_desde']}-{gp['zona_hasta']}")
            else:
                print(f"{symbol} en {tf}: GP sin datos")

    with open('rsi_data.json', 'w') as f:
        json.dump(rsi_resultado, f, indent=2)
    with open('fvg_data.json', 'w') as f:
        json.dump(fvg_resultado, f, indent=2)
    with open('sr_data.json', 'w') as f:
        json.dump(sr_resultado, f, indent=2)
    with open('gp_data.json', 'w') as f:
        json.dump(gp_resultado, f, indent=2)

    print("\n✅ Datos guardados en rsi_data.json, fvg_data.json, sr_data.json y gp_data.json")


if __name__ == '__main__':
    main()
