# -*- coding: utf-8 -*-
"""
=====================================================================
 CLASSIFICAÇÃO DE LOCALIZAÇÃO INDOOR POR FINGERPRINTING WiFi
 Algoritmos: KNN, WKNN e SVM
=====================================================================
Trabalho Final - Comunicação Sem Fio

Treina os modelos com a base de RPs e avalia sobre a base de TPs,
considerando 10, 30 e 50 amostras (fingerprints) por TP.

Parâmetros exigidos no trabalho:
  KNN  : K = 1,3,5,7 | distância Euclidiana | peso uniforme
  WKNN : K = 1,3,5,7 | distância Euclidiana | peso = inverso da distância
  SVM  : kernel = linear e RBF | C = 1,10 | gamma = scale

Métricas: Accuracy, Precision, Recall, F1-Score e Matriz de Confusão.

Saídas:
  - resultados_metricas.csv  (todas as configurações e cenários)
  - fig_matriz_confusao_*.png (matrizes de confusão dos melhores modelos)
  - acuracia_coinc_vs_naocoinc.csv (comparação coincidente x não coincidente)
=====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sem display
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

TRAIN_CSV = "wifi_train_50_por_rp_com_local.csv"
TEST_CSV = "wifi_test_50_por_rp_com_local.csv"

AP_COLS = [f"ap{i+1}" for i in range(8)]
CENARIOS = [10, 30, 50]          # amostras por TP
K_VALORES = [1, 3, 5, 7]
SVM_C = [1, 10]
SVM_KERNELS = ["linear", "rbf"]

# TPs coincidentes e não coincidentes (para análise comparativa)
TP_COINCIDENTES = ["TP1", "TP2", "TP3", "TP4"]
TP_NAO_COINCIDENTES = ["TP5", "TP6", "TP7", "TP8"]

RNG = np.random.default_rng(7)


def carregar_dados():
    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)
    return df_train, df_test


def subset_por_tp(df_test, n_amostras):
    """Seleciona n_amostras fingerprints de cada TP (amostragem fixa)."""
    partes = []
    for tp, grupo in df_test.groupby("ponto", sort=False):
        n = min(n_amostras, len(grupo))
        partes.append(grupo.iloc[:n])
    return pd.concat(partes, ignore_index=True)


def metricas(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def acuracia_por_grupo(df_eval, y_pred):
    """Acurácia separada para TPs coincidentes e não coincidentes."""
    df_eval = df_eval.copy()
    df_eval["pred"] = y_pred
    df_eval["acerto"] = (df_eval["pred"] == df_eval["local"]).astype(int)
    coinc = df_eval[df_eval["ponto"].isin(TP_COINCIDENTES)]["acerto"].mean()
    naoc = df_eval[df_eval["ponto"].isin(TP_NAO_COINCIDENTES)]["acerto"].mean()
    return coinc, naoc


def main():
    df_train, df_test = carregar_dados()

    X_train = df_train[AP_COLS].values.astype(float)
    y_train = df_train["local"].values   # rótulos RP1..RP8

    # Padronização: ajustada no treino e aplicada ao teste.
    # Essencial para o SVM-RBF; para KNN mantém as escalas comparáveis.
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    resultados = []          # métricas globais por configuração/cenário
    comparacao_grupo = []    # acurácia coincidente x não coincidente
    melhores_cm = {}         # guarda matriz de confusão do melhor por algoritmo

    for n in CENARIOS:
        df_eval = subset_por_tp(df_test, n)
        X_test = df_eval[AP_COLS].values.astype(float)
        y_test = df_eval["local"].values
        X_test_s = scaler.transform(X_test)

        # ---------------- KNN (peso uniforme) ----------------
        for k in K_VALORES:
            clf = KNeighborsClassifier(n_neighbors=k, weights="uniform",
                                       metric="euclidean")
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
            m = metricas(y_test, y_pred)
            coinc, naoc = acuracia_por_grupo(df_eval, y_pred)
            resultados.append({"algoritmo": "KNN", "config": f"K={k}",
                               "amostras": n, **m})
            comparacao_grupo.append({"algoritmo": "KNN", "config": f"K={k}",
                                     "amostras": n, "acc_coincidente": coinc,
                                     "acc_nao_coincidente": naoc})

        # ---------------- WKNN (peso = inverso da distância) ----------------
        for k in K_VALORES:
            clf = KNeighborsClassifier(n_neighbors=k, weights="distance",
                                       metric="euclidean")
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
            m = metricas(y_test, y_pred)
            coinc, naoc = acuracia_por_grupo(df_eval, y_pred)
            resultados.append({"algoritmo": "WKNN", "config": f"K={k}",
                               "amostras": n, **m})
            comparacao_grupo.append({"algoritmo": "WKNN", "config": f"K={k}",
                                     "amostras": n, "acc_coincidente": coinc,
                                     "acc_nao_coincidente": naoc})

        # ---------------- SVM (kernel linear e RBF) ----------------
        for kernel in SVM_KERNELS:
            for c in SVM_C:
                clf = SVC(kernel=kernel, C=c, gamma="scale")
                clf.fit(X_train_s, y_train)
                y_pred = clf.predict(X_test_s)
                m = metricas(y_test, y_pred)
                coinc, naoc = acuracia_por_grupo(df_eval, y_pred)
                cfg = f"{kernel},C={c},gamma=scale"
                resultados.append({"algoritmo": "SVM", "config": cfg,
                                   "amostras": n, **m})
                comparacao_grupo.append({"algoritmo": "SVM", "config": cfg,
                                         "amostras": n, "acc_coincidente": coinc,
                                         "acc_nao_coincidente": naoc})

    df_res = pd.DataFrame(resultados)
    df_grp = pd.DataFrame(comparacao_grupo)
    df_res.to_csv("resultados_metricas.csv", index=False)
    df_grp.to_csv("acuracia_coinc_vs_naocoinc.csv", index=False)

    # ================= RELATÓRIO NO TERMINAL =================
    pd.set_option("display.width", 120)
    pd.set_option("display.max_rows", None)

    print("=" * 70)
    print("RESULTADOS - MÉTRICAS GLOBAIS (todas as configurações)")
    print("=" * 70)
    df_show = df_res.copy()
    for c in ["accuracy", "precision", "recall", "f1"]:
        df_show[c] = (df_show[c] * 100).round(2)
    print(df_show.to_string(index=False))

    print("\n" + "=" * 70)
    print("MELHOR CONFIGURAÇÃO POR ALGORITMO (por accuracy, cenário 50 amostras)")
    print("=" * 70)
    df50 = df_res[df_res["amostras"] == 50]
    melhores = {}
    for alg in ["KNN", "WKNN", "SVM"]:
        sub = df50[df50["algoritmo"] == alg].sort_values("accuracy", ascending=False)
        best = sub.iloc[0]
        melhores[alg] = best
        print(f"{alg:5s} -> {best['config']:22s} "
              f"acc={best['accuracy']*100:.2f}%  prec={best['precision']*100:.2f}%  "
              f"rec={best['recall']*100:.2f}%  f1={best['f1']*100:.2f}%")

    print("\n" + "=" * 70)
    print("COMPARAÇÃO: TPs COINCIDENTES x NÃO COINCIDENTES")
    print("=" * 70)
    df_grp_show = df_grp.copy()
    df_grp_show["acc_coincidente"] = (df_grp_show["acc_coincidente"] * 100).round(2)
    df_grp_show["acc_nao_coincidente"] = (df_grp_show["acc_nao_coincidente"] * 100).round(2)
    print(df_grp_show.to_string(index=False))

    # ================= MATRIZES DE CONFUSÃO (melhores modelos, 50 amostras) =================
    df_eval = subset_por_tp(df_test, 50)
    X_test = scaler.transform(df_eval[AP_COLS].values.astype(float))
    y_test = df_eval["local"].values
    labels = sorted(df_train["local"].unique())

    def treinar_predizer(alg, cfg_row):
        if alg == "KNN":
            k = int(cfg_row["config"].split("=")[1])
            clf = KNeighborsClassifier(n_neighbors=k, weights="uniform", metric="euclidean")
        elif alg == "WKNN":
            k = int(cfg_row["config"].split("=")[1])
            clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")
        else:
            partes = cfg_row["config"].split(",")
            kernel = partes[0]
            c = int(partes[1].split("=")[1])
            clf = SVC(kernel=kernel, C=c, gamma="scale")
        clf.fit(X_train_s, y_train)
        return clf.predict(X_test)

    for alg in ["KNN", "WKNN", "SVM"]:
        y_pred = treinar_predizer(alg, melhores[alg])
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(f"Matriz de Confusão - {alg} ({melhores[alg]['config']})\n"
                     f"50 amostras/TP  |  acc={melhores[alg]['accuracy']*100:.1f}%")
        plt.tight_layout()
        fname = f"fig_matriz_confusao_{alg}.png"
        plt.savefig(fname, dpi=130)
        plt.close(fig)
        print(f"\n[figura salva] {fname}")

    # ================= GRÁFICO COMPARATIVO DE ACURÁCIA x AMOSTRAS =================
    fig, ax = plt.subplots(figsize=(7, 5))
    for alg in ["KNN", "WKNN", "SVM"]:
        accs = []
        for n in CENARIOS:
            sub = df_res[(df_res["algoritmo"] == alg) & (df_res["amostras"] == n)]
            accs.append(sub["accuracy"].max() * 100)  # melhor config no cenário
        ax.plot(CENARIOS, accs, marker="o", label=alg)
    ax.set_xlabel("Amostras por TP")
    ax.set_ylabel("Acurácia (%) - melhor configuração")
    ax.set_title("Acurácia x nº de amostras por TP")
    ax.set_xticks(CENARIOS)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("fig_acuracia_x_amostras.png", dpi=130)
    plt.close(fig)
    print("[figura salva] fig_acuracia_x_amostras.png")

    print("\nArquivos gerados: resultados_metricas.csv, "
          "acuracia_coinc_vs_naocoinc.csv, fig_matriz_confusao_*.png, "
          "fig_acuracia_x_amostras.png")


if __name__ == "__main__":
    main()
