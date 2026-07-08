import serial
import time
import csv
import os
from collections import defaultdict


# =====================================================
# CONFIGURAÇÕES DA COMUNICAÇÃO SERIAL
# =====================================================


PORTA_SERIAL = "COM6"
BAUDRATE = 115200


# =====================================================
# CONFIGURAÇÃO DO PONTO DE COLETA
# =====================================================


# Tipo do ponto:
# "RP" = Reference Point
# ponto de referência usado para treinamento.
#
# "TP" = Test Point
# ponto usado para testar o modelo.
TIPO_PONTO = "RP"


# Identificador do ponto físico onde a coleta está sendo feita.
#
# Exemplos:
# PONTO_ID = "RP1"
# PONTO_ID = "RP2"
# PONTO_ID = "TP1"
# PONTO_ID = "TP2"
PONTO_ID = "RP1"


# Classe/região correta esperada para esta coleta.
#
# Esta é a informação usada pelo algoritmo como resposta correta.
#
# Exemplos:
# Se a coleta está no RP1:
# LOCAL_REAL = "RP1"
#
# Se a coleta está em um TP mais próximo da região RP2:
# LOCAL_REAL = "RP2"
#
# O modelo tentará prever esse valor.
LOCAL_REAL = "RP1"


# Quantidade de fingerprints a serem coletados neste ponto.
NUM_LEITURAS = 50


# =====================================================
# APs MONITORADOS
# =====================================================


# Lista fixa de BSSIDs dos Access Points monitorados.
# A ordem desta lista define a ordem das colunas:
# ap1, ap2, ap3, ...
APS_MONITORADOS = [
    "3E:64:CF:2C:34:3A",
    "3E:64:CF:4C:34:3A",
    "E8:D2:FF:A5:B5:9B",
    "EA:D2:FF:A5:B7:9C",
    "08:C7:F5:2F:2B:64",
    "84:0B:BB:E4:E2:B2",
    "3A:1F:48:54:4C:50",
    "30:1F:48:54:4C:50"
]


# =====================================================
# ARQUIVO DE SAÍDA
# =====================================================


# Para coletar dados de treinamento:
ARQUIVO_CSV = "wifi_train_dataset.csv"


# Para coletar dados de teste, altere para:
# ARQUIVO_CSV = "wifi_test_dataset.csv"


# =====================================================
# CONEXÃO SERIAL
# =====================================================


ser = serial.Serial(PORTA_SERIAL, BAUDRATE, timeout=1)
time.sleep(2)


print(f"Conectado em {PORTA_SERIAL}")
print(f"Arquivo de saída: {ARQUIVO_CSV}")
print(f"Tipo do ponto: {TIPO_PONTO}")
print(f"Ponto de coleta: {PONTO_ID}")
print(f"Classe/região correta: {LOCAL_REAL}")


# Verifica se o arquivo já existe.
# Se não existir, será criado com cabeçalho.
arquivo_existe = os.path.exists(ARQUIVO_CSV)


# Usamos modo "a" para acrescentar novas coletas sem apagar as anteriores.
with open(ARQUIVO_CSV, mode="a", newline="", encoding="utf-8") as csvfile:


    writer = csv.writer(csvfile)


    # Cria o cabeçalho apenas se o arquivo ainda não existir.
    if not arquivo_existe:
        cabecalho = []


        for i in range(len(APS_MONITORADOS)):
            cabecalho.append(f"ap{i+1}")


        cabecalho += ["tipo", "ponto", "local"]


        writer.writerow(cabecalho)


        print("Cabeçalho CSV criado.")


    fingerprints_coletados = 0


    # Dicionário usado para armazenar os RSSIs do scan atual.
    # Se algum AP monitorado não for encontrado, será usado -100.
    scan_atual = defaultdict(lambda: -100)


    # =====================================================
    # LOOP DE COLETA
    # =====================================================


    while fingerprints_coletados < NUM_LEITURAS:


        linha = ser.readline().decode(
            "utf-8",
            errors="ignore"
        ).strip()


        if not linha:
            continue


        print(linha)


        # O ESP32 envia END_SCAN ao final de cada varredura.
        # Cada END_SCAN representa 1 fingerprint completo.
        if linha == "END_SCAN":


            vetor_rssi = []


            # Monta o vetor RSSI seguindo a ordem fixa dos BSSIDs.
            for bssid in APS_MONITORADOS:
                vetor_rssi.append(scan_atual[bssid])


            # Adiciona informações do ponto coletado.
            vetor_rssi += [
                TIPO_PONTO,
                PONTO_ID,
                LOCAL_REAL
            ]


            writer.writerow(vetor_rssi)


            fingerprints_coletados += 1


            print(
                f"[{fingerprints_coletados}/{NUM_LEITURAS}] "
                f"fingerprint salvo para {PONTO_ID}."
            )


            # Limpa o scan atual para iniciar nova leitura.
            scan_atual = defaultdict(lambda: -100)


            continue


        # Ignora o cabeçalho enviado pelo ESP32.
        if linha.startswith("SCAN_ID"):
            continue


        # =====================================================
        # PROCESSAMENTO DAS LINHAS ENVIADAS PELO ESP32
        # Formato esperado:
        # SCAN_ID,SSID,BSSID,CHANNEL,RSSI
        # =====================================================


        try:
            partes = linha.split(",")


            scan_id = partes[0]
            ssid = partes[1]
            bssid = partes[2]
            channel = partes[3]
            rssi = int(partes[4])


            # Salva somente os APs que estão na lista monitorada.
            if bssid in APS_MONITORADOS:
                scan_atual[bssid] = rssi


        except Exception as e:
            print(f"Erro ao processar linha: {e}")


print("Coleta finalizada.")
ser.close()
