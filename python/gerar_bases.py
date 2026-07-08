# -*- coding: utf-8 -*-
"""
=====================================================================
 FINGERPRINTING WiFi (RSSI)
=====================================================================
Trabalho Final - Comunicação Sem Fio
Localização Indoor por Fingerprinting WiFi (RSSI)

Este script gera bases de dados FICTÍCIAS, porém fisicamente
coerentes, para o experimento de localização indoor. Como a coleta
real depende do ESP32 (scan.ino) e da leitura da porta serial
(csv_formatado-v2.py), aqui reproduzimos esse processo através de um
modelo de propagação log-distância (log-distance path loss).

Saídas:
  - wifi_train_50_por_rp_com_local.csv  (8 RPs x 50 fingerprints)
  - wifi_test_50_por_rp_com_local.csv   (8 TPs x 50 fingerprints)

Formato das colunas (idêntico ao gerado por csv_formatado-v2.py):
  ap1, ap2, ap3, ap4, ap5, ap6, ap7, ap8, tipo, ponto, local
=====================================================================
"""

import numpy as np
import pandas as pd

# Semente para reprodutibilidade dos dados fictícios
RNG = np.random.default_rng(42)

NUM_LEITURAS = 50            # 50 fingerprints por ponto (RP e TP)
RSSI_NAO_DETECTADO = -100    # valor usado quando o AP não é "visto"
SENSIBILIDADE = -92          # abaixo disso o AP é considerado ausente
RSSI_MIN, RSSI_MAX = -100, -30

# =====================================================================
# APs MONITORADOS
# Mesmos BSSIDs usados no csv_formatado-v2.py (redes reais observadas
# no arquivo-rssi_dados.txt). A ordem define as colunas ap1..ap8.
# =====================================================================
APS_MONITORADOS = [
    "3E:64:CF:2C:34:3A",   # ap1 - TFMARQUES
    "3E:64:CF:4C:34:3A",   # ap2 - REDE TESTE AMAZON ON
    "E8:D2:FF:A5:B5:9B",   # ap3 - Celso_2G
    "EA:D2:FF:A5:B7:9C",   # ap4 - #CLARO-WIFI
    "08:C7:F5:2F:2B:64",   # ap5 - RANDY2G
    "84:0B:BB:E4:E2:B2",   # ap6 - VIVOFIBRA
    "3A:1F:48:54:4C:50",   # ap7 - #CLARO-WIFI
    "30:1F:48:54:4C:50",   # ap8 - fernandaRafael
]
N_AP = len(APS_MONITORADOS)

# Posição (x, y) de cada AP no plano do ambiente (em metros).
# Espalhados ao redor da planta para gerar assinaturas distinguíveis.
AP_POS = np.array([
    [-1.0,  8.0],   # ap1
    [ 8.0,  9.0],   # ap2
    [-1.0, -1.0],   # ap3
    [ 8.0, -1.0],   # ap4
    [16.0,  2.0],   # ap5
    [16.0,  7.0],   # ap6
    [12.0, 12.0],   # ap7
    [ 5.0, 12.0],   # ap8
])

# Potência de referência a 1 m (dBm) e expoente de perda por AP.
# Pequenas variações simulam diferenças de potência de transmissão.
P0 = np.array([-38, -40, -41, -39, -42, -40, -43, -41], dtype=float)
N_PATHLOSS = np.array([2.6, 2.7, 2.8, 2.7, 2.9, 2.8, 3.0, 2.9])

# =====================================================================
# PONTOS DE REFERÊNCIA (RPs) - 8 pontos
# Layout: sala dividida em 4 quadrantes + corredor + varanda + banheiro
# =====================================================================
RP_POS = {
    "RP1": (2.0, 6.0),   # sala - quadrante 1 (sup. esquerdo)
    "RP2": (6.0, 6.0),   # sala - quadrante 2 (sup. direito)
    "RP3": (2.0, 2.0),   # sala - quadrante 3 (inf. esquerdo)
    "RP4": (6.0, 2.0),   # sala - quadrante 4 (inf. direito)
    "RP5": (10.0, 4.0),  # corredor - local 1
    "RP6": (13.0, 4.0),  # corredor - local 2
    "RP7": (10.0, 9.0),  # varanda
    "RP8": (13.0, 9.0),  # banheiro
}

# =====================================================================
# PONTOS DE TESTE (TPs) - 8 pontos
#   4 coincidentes com RPs   (mesma posição do RP)
#   4 não coincidentes       (entre RPs adjacentes, mais próximos de um)
# 'local' = classe/região verdadeira (o RP dominante que se quer prever)
# =====================================================================
TP_DEF = {
    # --- Coincidentes ---
    "TP1": {"pos": RP_POS["RP1"],        "local": "RP1", "coinc": True},
    "TP2": {"pos": RP_POS["RP2"],        "local": "RP2", "coinc": True},
    "TP3": {"pos": RP_POS["RP3"],        "local": "RP3", "coinc": True},
    "TP4": {"pos": RP_POS["RP4"],        "local": "RP4", "coinc": True},
    # --- Não coincidentes (entre regiões adjacentes, mais perto de um RP) ---
    # TP5 entre RP1 e RP2, porém mais próximo de RP1
    "TP5": {"pos": (3.2, 6.0),           "local": "RP1", "coinc": False},
    # TP6 entre RP2 e RP3 (diagonal da sala), porém mais próximo de RP2
    "TP6": {"pos": (5.0, 5.2),           "local": "RP2", "coinc": False},
    # TP7 entre RP5 e RP6 (corredor), porém mais próximo de RP5
    "TP7": {"pos": (11.2, 4.0),          "local": "RP5", "coinc": False},
    # TP8 entre RP7 e RP8, porém mais próximo de RP7
    "TP8": {"pos": (11.2, 9.0),          "local": "RP7", "coinc": False},
}


def rssi_medio(ponto_xy):
    """Calcula o RSSI médio (dBm) de cada AP em uma posição, via
    modelo log-distância: RSSI = P0 - 10*n*log10(d)."""
    dx = AP_POS[:, 0] - ponto_xy[0]
    dy = AP_POS[:, 1] - ponto_xy[1]
    dist = np.sqrt(dx**2 + dy**2)
    dist = np.maximum(dist, 1.0)  # evita log(0); campo próximo = 1 m
    rssi = P0 - 10.0 * N_PATHLOSS * np.log10(dist)
    return rssi


def gerar_fingerprints(ponto_xy, n_amostras, sigma=3.0):
    """Gera n fingerprints ruidosos (shadowing gaussiano) para a posição."""
    base = rssi_medio(ponto_xy)
    amostras = []
    for _ in range(n_amostras):
        ruido = RNG.normal(0.0, sigma, size=N_AP)
        valores = base + ruido
        # APs abaixo da sensibilidade viram "não detectado" (-100)
        valores = np.where(valores < SENSIBILIDADE, RSSI_NAO_DETECTADO, valores)
        valores = np.clip(valores, RSSI_MIN, RSSI_MAX)
        amostras.append(np.round(valores).astype(int))
    return np.array(amostras)


def montar_dataframe(pontos, tipo):
    """Monta o DataFrame no formato de saída para um conjunto de pontos."""
    colunas = [f"ap{i+1}" for i in range(N_AP)] + ["tipo", "ponto", "local"]
    linhas = []
    for nome, info in pontos.items():
        pos = info["pos"]
        local = info["local"]
        fps = gerar_fingerprints(pos, NUM_LEITURAS)
        for fp in fps:
            linhas.append(list(fp) + [tipo, nome, local])
    return pd.DataFrame(linhas, columns=colunas)


def main():
    # --- Base de TREINAMENTO (RPs) ---
    pontos_rp = {nome: {"pos": pos, "local": nome} for nome, pos in RP_POS.items()}
    df_train = montar_dataframe(pontos_rp, tipo="RP")
    df_train.to_csv("wifi_train_50_por_rp_com_local.csv", index=False)

    # --- Base de TESTE (TPs) ---
    df_test = montar_dataframe(TP_DEF, tipo="TP")
    df_test.to_csv("wifi_test_50_por_rp_com_local.csv", index=False)

    # --- Resumo no terminal ---
    print("=" * 60)
    print("BASES GERADAS COM SUCESSO")
    print("=" * 60)
    print(f"Treino: {df_train.shape[0]} linhas "
          f"({df_train['ponto'].nunique()} RPs x {NUM_LEITURAS} fingerprints)")
    print(f"Teste : {df_test.shape[0]} linhas "
          f"({df_test['ponto'].nunique()} TPs x {NUM_LEITURAS} fingerprints)")
    print(f"APs monitorados: {N_AP}")
    print()
    print("RSSI médio (dBm) por RP:")
    cols_ap = [f"ap{i+1}" for i in range(N_AP)]
    resumo = df_train.groupby("ponto")[cols_ap].mean().round(1)
    print(resumo.to_string())
    print()
    print("Mapeamento TP -> local verdadeiro (RP dominante):")
    for nome, info in TP_DEF.items():
        tag = "coincidente" if info["coinc"] else "NÃO coincidente"
        print(f"  {nome} -> {info['local']:>4}  ({tag})")


if __name__ == "__main__":
    main()
