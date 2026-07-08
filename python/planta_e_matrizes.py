# -*- coding: utf-8 -*-
"""
Gera o esboço da planta baixa do cenário (RPs, TPs e APs) e imprime
as matrizes de confusão em texto (para inclusão no relatório).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Reaproveita as posições do gerador
from gerar_bases import AP_POS, RP_POS, TP_DEF, APS_MONITORADOS

AP_COLS = [f"ap{i+1}" for i in range(8)]

# =================== PLANTA BAIXA ===================
fig, ax = plt.subplots(figsize=(9, 8))

# Cômodos (retângulos ilustrativos)
comodos = [
    ("Sala (quadrantes 1-4)", (0, 0, 8, 8), "#eaf4ff"),
    ("Corredor", (8.5, 2, 6, 4), "#fff5e6"),
    ("Varanda", (8.5, 6.5, 3, 5), "#eafbea"),
    ("Banheiro", (11.7, 6.5, 3, 5), "#fdeaea"),
]
for nome, (x, y, w, h), cor in comodos:
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=cor,
                               edgecolor="gray", lw=1.5, zorder=1))
    ax.text(x + w / 2, y + h - 0.4, nome, ha="center", va="top",
            fontsize=9, style="italic", color="dimgray")

# Divisão dos quadrantes da sala
ax.plot([4, 4], [0, 8], "--", color="lightgray", lw=1, zorder=1)
ax.plot([0, 8], [4, 4], "--", color="lightgray", lw=1, zorder=1)

# RPs
for nome, (x, y) in RP_POS.items():
    ax.scatter(x, y, marker="s", s=220, color="#1f6feb", zorder=3, edgecolors="k")
    ax.text(x, y + 0.45, nome, ha="center", fontsize=9, fontweight="bold",
            color="#0b3d91")

# TPs
for nome, info in TP_DEF.items():
    x, y = info["pos"]
    cor = "#2ca02c" if info["coinc"] else "#d62728"
    ax.scatter(x, y, marker="^", s=170, color=cor, zorder=4, edgecolors="k")
    dy = -0.75 if info["coinc"] else 0.5
    ax.text(x + 0.35, y + dy, nome, ha="left", fontsize=8, color=cor,
            fontweight="bold")

# APs
for i, (x, y) in enumerate(AP_POS):
    ax.scatter(x, y, marker="*", s=320, color="#ff7f0e", zorder=3, edgecolors="k")
    ax.text(x, y + 0.5, f"ap{i+1}", ha="center", fontsize=8, color="#a15c00")

# Legenda manual
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#1f6feb",
           markersize=12, markeredgecolor="k", label="RP (referência)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c",
           markersize=12, markeredgecolor="k", label="TP coincidente"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="#d62728",
           markersize=12, markeredgecolor="k", label="TP não coincidente"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="#ff7f0e",
           markersize=15, markeredgecolor="k", label="AP (Access Point)"),
]
ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.01, 1.0),
          fontsize=9, frameon=True)

ax.set_xlim(-2.5, 16.5)
ax.set_ylim(-2, 12.5)
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("Planta baixa do cenário - RPs, TPs e APs")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("fig_planta_baixa.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("[figura salva] fig_planta_baixa.png")

# =================== MATRIZES DE CONFUSÃO EM TEXTO ===================
df_train = pd.read_csv("wifi_train_50_por_rp_com_local.csv")
df_test = pd.read_csv("wifi_test_50_por_rp_com_local.csv")

X_train = df_train[AP_COLS].values.astype(float)
y_train = df_train["local"].values
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
labels = sorted(df_train["local"].unique())

# usa 50 amostras por TP
X_test = scaler.transform(df_test[AP_COLS].values.astype(float))
y_test = df_test["local"].values

modelos = {
    "KNN (K=5)": KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="euclidean"),
    "WKNN (K=5)": KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean"),
    "SVM (rbf, C=1)": SVC(kernel="rbf", C=1, gamma="scale"),
}

for nome, clf in modelos.items():
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    dfcm = pd.DataFrame(cm, index=[f"real {l}" for l in labels],
                        columns=[f"pred {l}" for l in labels])
    print("\n" + "=" * 60)
    print(f"MATRIZ DE CONFUSÃO - {nome}  (50 amostras/TP)")
    print("=" * 60)
    print(dfcm.to_string())
