import json
import ccxt
import pandas as pd
import numpy as np

# ===== CONFIGURACIÓN =====
CRIPTOS = ['BTC/USDT', 'HYPE/USDT', 'DOGE/USDT', 'PEPE/USDT']
TEMPORALIDADES = ['1h', '4h', '1d', '1w']
RSI_PERIODO = 14

# RSI
RSI_VELAS = {'1h': 100, '4h': 100, '1d': 300, '1w': 200}

# FVG
FVG_VELAS = {'1h': 100, '4h': 100, '1d': 300, '1w': 200}
FVG_TEMPORALIDADES = ['1h', '4h', '1d']
FVG_MAX_DISTANCIA_PCT = {'1h': 5.0, '4h': 5.0, '1d': 15.0, '1w': 25.0}
FVG_MIN_TAMANO_PCT = 0.15

# ===== S/R — LÓGICA JUNIORQTRADER =====
# Parámetros basados en Auto Support Resistance Channels
SR_VELAS = {'1h': 300, '4h': 300, '1d': 300, '1w': 300}
SR_TEMPORALIDADES = ['1h', '4h', '1d', '1w']

# Parámetros del indicador
SR_PIVOT_LENGTH = 10
SR_ATR_LEN = 14
SR_ATR_MULT = 0.5
SR_MIN_PIVOTS = 1
SR_TOP_ZONES = 2  # 2 niveles por TF

# Pesos (weights) del score
SR_PIVOT_WEIGHT = 5
SR_BREAK_WEIGHT = 3
SR_DWELL_WEIGHT = 2
SR_CLOSE_INSIDE_WEIGHT = 3
SR_REACTION_WEIGHT = 4
SR_REACTION_BARS = 12
SR_REACTION_ATR = 1.2
SR_REACTION_RETRACE_FRAC = 0.5

# Merge de zonas
SR_MERGE_ATR_FRAC = 0.8

# Umbrales de clasificación (texto)
# Score >= 15 → "Muy fuerte"
# Score 5-14 → "Media"
# Score < 5  → "Débil"
SR_SCORE_MUY_FUERTE = 15
SR_SCORE_MEDIA = 5

# Golden Pocket
GP_VELAS = {'1h': 300, '4h': 300, '1d': 300, '1w': 200}
GP_TEMPORALIDADES = ['1h', '4h', '1d', '1w']
GP_STRENGTH = {'1h': 5, '4h': 5, '1d': 5, '1w': 5}

# Liquidaciones
LIQ_VELAS = 300
LIQ_APALANCAMIENTOS = [10, 12, 15, 20, 30]
LIQ_PCT = {10: 9.5, 12: 7.9, 15: 6.25, 20: 4.5, 30: 2.8}
LIQ_LOOKBACK = 100
LIQ_SENSIBILIDAD = 2.0
LIQ_TOLERANCIA_PCT = 1.0


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


def calcular_atr(df, periodo=14):
    """ATR (Average True Range) — versión simple"""
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=periodo).mean()
    return atr


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


def detectar_pivotes_juniorq(df, pivot_length=10):
    """Detecta pivotes al estilo JuniorQTrader"""
    pivotes = []
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)

    for i in range(pivot_length, n - pivot_length):
        # Pivote alto
        es_alto = True
        for j in range(1, pivot_length + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                es_alto = False
                break
        if es_alto:
            pivotes.append({'precio': float(highs[i]), 'indice': i, 'tipo': +1})

        # Pivote bajo
        es_bajo = True
        for j in range(1, pivot_length + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                es_bajo = False
                break
        if es_bajo:
            pivotes.append({'precio': float(lows[i]), 'indice': i, 'tipo': -1})

    return pivotes


def calcular_reaction_score(df, pivot, atr_series, config):
    """Calcula el reaction score de un pivote"""
    idx = pivot['indice']
    tipo = pivot['tipo']
    precio = pivot['precio']
    n = len(df)

    bars_available = n - 1 - idx
    n_bars = min(SR_REACTION_BARS, bars_available)

    if n_bars < 2:
        return 0.0

    atr_at_pivot = atr_series.iloc[idx]
    if pd.isna(atr_at_pivot) or atr_at_pivot == 0:
        return 0.0

    if tipo == -1:
        # Pivote bajo → queremos que suba
        best_high = df['high'].iloc[idx+1:idx+1+n_bars].max()
        worst_low_after = df['low'].iloc[idx+1:idx+1+n_bars].min()
        excursion = best_high - precio
        bad_retrace = max(0.0, precio - worst_low_after)
    else:
        # Pivote alto → queremos que baje
        best_low = df['low'].iloc[idx+1:idx+1+n_bars].min()
        worst_high_after = df['high'].iloc[idx+1:idx+1+n_bars].max()
        excursion = precio - best_low
        bad_retrace = max(0.0, worst_high_after - precio)

    enough_impulse = excursion >= atr_at_pivot * SR_REACTION_ATR
    retrace_ok = bad_retrace <= excursion * SR_REACTION_RETRACE_FRAC

    if enough_impulse and retrace_ok:
        return (excursion / atr_at_pivot) * SR_REACTION_WEIGHT
    return 0.0


def calcular_penalizaciones(df, zona_desde, zona_hasta, idx_inicio):
    """Calcula breaks, dwell, close inside para una zona"""
    breaks = 0
    dwell = 0
    close_inside = 0

    en_zona = False
    for i in range(idx_inicio, len(df)):
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        close = df['close'].iloc[i]

        # Close inside
        if zona_desde <= close <= zona_hasta:
            close_inside += 1

        # Dwell (vela dentro de la zona)
        if zona_desde <= low and high <= zona_hasta:
            dwell += 1

        # Break (atraviesa la zona)
        if not en_zona:
            if high > zona_hasta and close > zona_hasta:
                breaks += 1
                en_zona = True
            elif low < zona_desde and close < zona_desde:
                breaks += 1
                en_zona = True
        else:
            if close < zona_desde or close > zona_hasta:
                en_zona = False

    return breaks, dwell, close_inside


def agrupar_pivotes_juniorq(pivotes, atr_promedio, df):
    """Agrupa pivotes en zonas por ATR × factor"""
    if not pivotes:
        return []

    zona_width = atr_promedio * SR_ATR_MULT
    merge_threshold = atr_promedio * SR_MERGE_ATR_FRAC

    # Ordenar por precio
    pivotes_ord = sorted(pivotes, key=lambda x: x['precio'])

    zonas = []
    grupo_actual = [pivotes_ord[0]]

    for i in range(1, len(pivotes_ord)):
        precio_actual_pivote = pivotes_ord[i]['precio']
        precio_prom_grupo = np.mean([p['precio'] for p in grupo_actual])

        if abs(precio_actual_pivote - precio_prom_grupo) <= merge_threshold:
            grupo_actual.append(pivotes_ord[i])
        else:
            if len(grupo_actual) >= SR_MIN_PIVOTS:
                zonas.append(crear_zona(grupo_actual, zona_width, df))
            grupo_actual = [pivotes_ord[i]]

    if len(grupo_actual) >= SR_MIN_PIVOTS:
        zonas.append(crear_zona(grupo_actual, zona_width, df))

    return zonas


def crear_zona(pivotes, zona_width, df):
    """Crea una zona con score calculado"""
    precios = [p['precio'] for p in pivotes]
    precio_prom = np.mean(precios)
    zona_desde = precio_prom - zona_width
    zona_hasta = precio_prom + zona_width

    # Índice del pivote más antiguo del grupo
    idx_min = min(p['indice'] for p in pivotes)

    # Calcular penalizaciones
    breaks, dwell, close_inside = calcular_penalizaciones(df, zona_desde, zona_hasta, idx_min)

    # Calcular score
    pivot_score = len(pivotes) * SR_PIVOT_WEIGHT
    reaction_score = sum(p.get('reaction', 0) for p in pivotes)
    penalty = (breaks * SR_BREAK_WEIGHT) + (dwell * SR_DWELL_WEIGHT) + (close_inside * SR_CLOSE_INSIDE_WEIGHT)

    score = pivot_score + reaction_score - penalty

    # Clasificar
    if score >= SR_SCORE_MUY_FUERTE:
        texto = 'Muy fuerte'
        emoji = '🔴'
    elif score >= SR_SCORE_MEDIA:
        texto = 'Media'
        emoji = '🟡'
    else:
        texto = 'Débil'
        emoji = '🟢'

    return {
        'precio': round(precio_prom, 2),
        'toques': len(pivotes),
        'score': round(score, 2),
        'texto': texto,
        'emoji': emoji,
        'breaks': breaks,
        'dwell': dwell,
        'close_inside': close_inside
    }


def calcular_sr_par(exchange, symbol, timeframe):
    """Nueva lógica S/R basada en JuniorQTrader"""
    try:
        limit = SR_VELAS.get(timeframe, 300)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)

        precio_actual = float(df['close'].iloc[-1])

        # ATR
        atr_series = calcular_atr(df, SR_ATR_LEN)
        atr_promedio = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else precio_actual * 0.01

        # Detectar pivotes
        pivotes = detectar_pivotes_juniorq(df, SR_PIVOT_LENGTH)

        # Calcular reaction score de cada pivote
        for piv in pivotes:
            piv['reaction'] = calcular_reaction_score(df, piv, atr_series, {})

        # Separar por tipo
        pivotes_altos = [p for p in pivotes if p['tipo'] == +1]
        pivotes_bajos = [p for p in pivotes if p['tipo'] == -1]

        # Agrupar en zonas
        zonas_altas = agrupar_pivotes_juniorq(pivotes_altos, atr_promedio, df)
        zonas_bajas = agrupar_pivotes_juniorq(pivotes_bajos, atr_promedio, df)

        # Filtrar arriba/abajo del precio
        resistencias = [z for z in zonas_altas if z['precio'] > precio_actual]
        soportes = [z for z in zonas_bajas if z['precio'] < precio_actual]

        # Ordenar por score (mayor primero) y tomar top N
        resistencias = sorted(resistencias, key=lambda x: -x['score'])[:SR_TOP_ZONES]
        soportes = sorted(soportes, key=lambda x: -x['score'])[:SR_TOP_ZONES]

        # Ordenar por cercanía al precio para mostrar
        resistencias = sorted(resistencias, key=lambda x: abs(x['precio'] - precio_actual))
        soportes = sorted(soportes, key=lambda x: abs(x['precio'] - precio_actual))

        # Redondear
        for z in resistencias:
            z['precio'] = redondear(z['precio'], precio_actual)
        for z in soportes:
            z['precio'] = redondear(z['precio'], precio_actual)

        return {
            'precio_actual': redondear(precio_actual, precio_actual),
            'atr': round(atr_promedio, 4),
            'resistencias': resistencias,
            'soportes': soportes
        }

    except Exception as e:
        print(f"Error S/R {symbol} {timeframe}: {e}")
        return {'precio_actual': None, 'resistencias': [], 'soportes': []}


def detectar_pivotes(df, strength):
    """Pivotes genéricos (para GP)"""
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


def encontrar_ultimo_impulso(pivotes_altos, pivotes_bajos):
    todos = []
    for p in pivotes_altos:
        todos.append({'tipo': 'alto', 'precio': p['precio'], 'indice': p['indice']})
    for p in pivotes_bajos:
        todos.append({'tipo': 'bajo', 'precio': p['precio'], 'indice': p['indice']})
    todos.sort(key=lambda x: x['indice'])
    for i in range(len(todos) - 1, 0, -1):
        actual = todos[i]
        anterior = todos[i - 1]
        if actual['tipo'] != anterior['tipo']:
            return anterior, actual
    return None, None


def calcular_golden_pocket_par(exchange, symbol, timeframe):
    try:
        limit = GP_VELAS.get(timeframe, 100)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)

        precio_actual = float(df['close'].iloc[-1])
        strength = GP_STRENGTH.get(timeframe, 5)

        pivotes_altos, pivotes_bajos = detectar_pivotes(df, strength)
        if not pivotes_altos or not pivotes_bajos:
            return None

        pivote_inicio, pivote_fin = encontrar_ultimo_impulso(pivotes_altos, pivotes_bajos)
        if not pivote_inicio or not pivote_fin:
            return None

        if pivote_fin['tipo'] == 'alto':
            tipo = 'alcista'
            swing_low = pivote_inicio['precio']
            swing_high = pivote_fin['precio']
            rango = swing_high - swing_low
            nivel_05 = swing_low + rango * 0.5
            nivel_0618 = swing_low + rango * 0.618
            zona_desde = min(nivel_05, nivel_0618)
            zona_hasta = max(nivel_05, nivel_0618)
            accion = 'SOPORTE'
        else:
            tipo = 'bajista'
            swing_high = pivote_inicio['precio']
            swing_low = pivote_fin['precio']
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
        print(f"Error GP {symbol} {timeframe}: {e}")
        return None


def calcular_liquidaciones_par(exchange, symbol):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=LIQ_VELAS)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['volume'] = df['volume'].astype(float)

        precio_actual = float(df['close'].iloc[-1])

        volumen = df['volume']
        media = volumen.rolling(window=LIQ_LOOKBACK).mean()
        std = volumen.rolling(window=LIQ_LOOKBACK).std()
        spikes = volumen > (media + LIQ_SENSIBILIDAD * std)

        zonas_long = {}
        zonas_short = {}

        for i in range(len(df)):
            if pd.isna(spikes.iloc[i]) or not spikes.iloc[i]:
                continue

            precio = float(df['close'].iloc[i])
            peso = float(df['volume'].iloc[i])

            for apal in LIQ_APALANCAMIENTOS:
                liq_long = precio * (1 - LIQ_PCT[apal] / 100)
                liq_short = precio * (1 + LIQ_PCT[apal] / 100)

                key_long = round(liq_long / (precio_actual * 0.001)) * (precio_actual * 0.001)
                key_short = round(liq_short / (precio_actual * 0.001)) * (precio_actual * 0.001)

                if key_long not in zonas_long:
                    zonas_long[key_long] = {'precio': liq_long, 'peso': 0, 'apalancamientos': set()}
                zonas_long[key_long]['peso'] += peso
                zonas_long[key_long]['apalancamientos'].add(apal)

                if key_short not in zonas_short:
                    zonas_short[key_short] = {'precio': liq_short, 'peso': 0, 'apalancamientos': set()}
                zonas_short[key_short]['peso'] += peso
                zonas_short[key_short]['apalancamientos'].add(apal)

        def agrupar_zonas(zonas_dict):
            lista = sorted(zonas_dict.values(), key=lambda x: x['precio'])
            if not lista:
                return []
            agrupadas = []
            grupo_actual = [lista[0]]
            for i in range(1, len(lista)):
                precio_actual_z = lista[i]['precio']
                precio_anterior = grupo_actual[-1]['precio']
                dist_pct = abs(precio_actual_z - precio_anterior) / precio_anterior * 100
                if dist_pct <= LIQ_TOLERANCIA_PCT:
                    grupo_actual.append(lista[i])
                else:
                    agrupadas.append(combinar_grupo(grupo_actual, precio_actual))
                    grupo_actual = [lista[i]]
            if grupo_actual:
                agrupadas.append(combinar_grupo(grupo_actual, precio_actual))
            return agrupadas

        def combinar_grupo(grupo, precio_actual):
            peso_total = sum(g['peso'] for g in grupo)
            if peso_total > 0:
                precio_pond = sum(g['precio'] * g['peso'] for g in grupo) / peso_total
            else:
                precio_pond = grupo[0]['precio']
            apals = set()
            for g in grupo:
                apals.update(g['apalancamientos'])
            return {
                'precio': redondear(precio_pond, precio_actual),
                'peso': redondear(peso_total, precio_actual),
                'apalancamientos': sorted(list(apals))
            }

        zonas_long_agr = agrupar_zonas(zonas_long)
        zonas_short_agr = agrupar_zonas(zonas_short)

        zonas_long_final = [z for z in zonas_long_agr if z['precio'] < precio_actual]
        zonas_short_final = [z for z in zonas_short_agr if z['precio'] > precio_actual]

        zonas_long_final.sort(key=lambda x: abs(x['precio'] - precio_actual))
        zonas_short_final.sort(key=lambda x: abs(x['precio'] - precio_actual))

        zonas_long_final = zonas_long_final[:5]
        zonas_short_final = zonas_short_final[:5]

        for z in zonas_long_final:
            z['distancia_pct'] = round((z['precio'] - precio_actual) / precio_actual * 100, 2)
        for z in zonas_short_final:
            z['distancia_pct'] = round((z['precio'] - precio_actual) / precio_actual * 100, 2)

        return {
            'precio_actual': redondear(precio_actual, precio_actual),
            'long_zones': zonas_long_final,
            'short_zones': zonas_short_final
        }
    except Exception as e:
        print(f"Error Liquidaciones {symbol}: {e}")
        return {'precio_actual': None, 'long_zones': [], 'short_zones': []}


def calcular_rsi_par(exchange, symbol, timeframe):
    try:
        limit = RSI_VELAS.get(timeframe, 100)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        rsi = calcular_rsi(df['close'], RSI_PERIODO)
        return round(float(rsi.iloc[-1]), 2)
    except Exception as e:
        print(f"Error RSI {symbol} {timeframe}: {e}")
        return None


def calcular_fvgs_par(exchange, symbol, timeframe):
    try:
        limit = FVG_VELAS.get(timeframe, 100)
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        precio_actual = float(df['close'].iloc[-1])
        fvgs = detectar_fvg(df)
        return filtrar_fvgs(fvgs, precio_actual, timeframe)
    except Exception as e:
        print(f"Error FVG {symbol} {timeframe}: {e}")
        return []


def main():
    exchange = ccxt.okx()

    rsi_resultado = {'last_update': pd.Timestamp.now(tz='UTC').isoformat(), 'data': {}}
    fvg_resultado = {'last_update': pd.Timestamp.now(tz='UTC').isoformat(), 'data': {}}
    sr_resultado = {'last_update': pd.Timestamp.now(tz='UTC').isoformat(), 'data': {}}
    gp_resultado = {'last_update': pd.Timestamp.now(tz='UTC').isoformat(), 'data': {}}
    liq_resultado = {'last_update': pd.Timestamp.now(tz='UTC').isoformat(), 'data': {}}

    for cripto in CRIPTOS:
        symbol = cripto.replace('/', '')
        rsi_resultado['data'][symbol] = {}
        fvg_resultado['data'][symbol] = {}
        sr_resultado['data'][symbol] = {}
        gp_resultado['data'][symbol] = {}
        liq_resultado['data'][symbol] = {}

        for tf in TEMPORALIDADES:
            rsi = calcular_rsi_par(exchange, cripto, tf)
            rsi_resultado['data'][symbol][tf] = rsi
            print(f"{symbol} {tf}: RSI = {rsi}")

        for tf in FVG_TEMPORALIDADES:
            fvgs = calcular_fvgs_par(exchange, cripto, tf)
            fvg_resultado['data'][symbol][tf] = fvgs
            print(f"{symbol} {tf}: {len(fvgs)} FVG")

        for tf in SR_TEMPORALIDADES:
            sr = calcular_sr_par(exchange, cripto, tf)
            sr_resultado['data'][symbol][tf] = sr
            print(f"{symbol} {tf}: {len(sr['resistencias'])} R / {len(sr['soportes'])} S")

        for tf in GP_TEMPORALIDADES:
            gp = calcular_golden_pocket_par(exchange, cripto, tf)
            gp_resultado['data'][symbol][tf] = gp
            if gp:
                print(f"{symbol} {tf}: GP {gp['tipo']} ({gp['accion']})")

        liq = calcular_liquidaciones_par(exchange, cripto)
        liq_resultado['data'][symbol] = liq
        print(f"{symbol}: {len(liq['short_zones'])} short / {len(liq['long_zones'])} long")

    with open('rsi_data.json', 'w') as f:
        json.dump(rsi_resultado, f, indent=2)
    with open('fvg_data.json', 'w') as f:
        json.dump(fvg_resultado, f, indent=2)
    with open('sr_data.json', 'w') as f:
        json.dump(sr_resultado, f, indent=2)
    with open('gp_data.json', 'w') as f:
        json.dump(gp_resultado, f, indent=2)
    with open('liq_data.json', 'w') as f:
        json.dump(liq_resultado, f, indent=2)

    print("\n✅ Datos guardados")


if __name__ == '__main__':
    main()
