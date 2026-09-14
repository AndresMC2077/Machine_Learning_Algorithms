"""
generar_graficas_diagnostico.py
--------------------------------
Genera las 4 gráficas de diagnóstico usadas en el reporte de análisis del árbol ID3:

  1. comparacion_accuracy.png   -> Accuracy en Train/Validación/Test: modelo BASE vs REGULARIZADO
  2. heatmap_hiperparametros.png -> Accuracy en validación para cada combinación (max_depth, min_samples_split)
  3. curva_complejidad.png      -> Accuracy en validación vs max_depth (para varios min_samples_split)
  4. curva_aprendizaje.png      -> Accuracy train/validación vs tamaño del set de entrenamiento
                                    (una versión para el modelo base y otra para el regularizado)

Requiere: pandas, numpy, matplotlib, scikit-learn (solo para accuracy_score/classification_report)
          y el archivo id3_lib.py en la misma carpeta.

USO:
    python generar_graficas_diagnostico.py --dataset dataset_compra_coche_aumentado.csv --target Comprar

Ver instrucciones detalladas en INSTRUCCIONES_USO.md
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin ventana, para poder correr en servidor/consola
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

from id3_lib import ID3, predict, tree_depth, tree_num_nodes

plt.rcParams.update({"font.size": 11})


def get_acc(tree, data, target):
    """Calcula accuracy y devuelve también las predicciones."""
    preds = [predict(tree, row) for _, row in data.iterrows()]
    return accuracy_score(data[target], preds), preds


def main(dataset_path, target, out_dir, seed):
    df = pd.read_csv(dataset_path)
    features = [c for c in df.columns if c != target]

    # Separación Train / Validation / Test (70% / 15% / 15%)
    df_train = df.sample(frac=0.7, random_state=seed)
    remaining = df.drop(df_train.index)
    df_val = remaining.sample(frac=0.5, random_state=seed)
    df_test = remaining.drop(df_val.index)

    print("Train:", df_train.shape, df_train[target].value_counts().to_dict())
    print("Val:  ", df_val.shape, df_val[target].value_counts().to_dict())
    print("Test: ", df_test.shape, df_test[target].value_counts().to_dict())

    # Modelo BASE: sin regularizar (max_depth=None, min_samples_split=2)
    #    Entrenado SOLO con df_train.
    base_model = ID3(max_depth=None, min_samples_split=2)
    base_tree = base_model.fit(df_train, target, features)

    acc_train_base, _ = get_acc(base_tree, df_train, target)
    acc_val_base, _ = get_acc(base_tree, df_val, target)
    acc_test_base, pred_test_base = get_acc(base_tree, df_test, target)

    print("\n--- MODELO BASE ---")
    print(f"Profundidad={tree_depth(base_tree)}  Nodos={tree_num_nodes(base_tree)}")
    print(f"Acc train={acc_train_base:.4f}  val={acc_val_base:.4f}  test={acc_test_base:.4f}")

    # Grid search de hiperparámetros usando SOLO el set de validación
    #    (el test no se toca en esta etapa).
    candidate_max_depths = [1, 2, 3, 4, 5, 6, None]
    candidate_min_samples_split = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    results_grid = []  # matriz [max_depth x min_samples_split] de accuracy en validación
    best_params, best_val_acc = None, -1

    for md in candidate_max_depths:
        row_acc = []
        for mss in candidate_min_samples_split:
            m = ID3(md, mss)
            t = m.fit(df_train, target, features)
            acc_v, _ = get_acc(t, df_val, target)
            row_acc.append(acc_v)
            if acc_v > best_val_acc:
                best_val_acc, best_params = acc_v, (md, mss)
        results_grid.append(row_acc)

    results_grid = np.array(results_grid)
    print("\nMejores hiperparámetros (según validación):", best_params)
    print(f"Accuracy en validación (mejor combinación): {best_val_acc:.4f}")

    # Modelo REGULARIZADO con los mejores hiperparámetros,
    #    entrenado SOLO con df_train 
    reg_model = ID3(*best_params)
    reg_tree = reg_model.fit(df_train, target, features)

    acc_train_reg, _ = get_acc(reg_tree, df_train, target)
    acc_val_reg, _ = get_acc(reg_tree, df_val, target)
    acc_test_reg, pred_test_reg = get_acc(reg_tree, df_test, target)

    print("\n--- MODELO REGULARIZADO", best_params, "---")
    print(f"Profundidad={tree_depth(reg_tree)}  Nodos={tree_num_nodes(reg_tree)}")
    print(f"Acc train={acc_train_reg:.4f}  val={acc_val_reg:.4f}  test={acc_test_reg:.4f}")

    print("\nReporte clasificación TEST - modelo base:")
    print(classification_report(df_test[target], pred_test_base, zero_division=0))
    print("\nReporte clasificación TEST - modelo regularizado:")
    print(classification_report(df_test[target], pred_test_reg, zero_division=0))

  
    # GRÁFICA 1: Comparación de accuracy Train/Val/Test - Base vs Regularizado
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Train", "Validación", "Test"]
    base_vals = [acc_train_base, acc_val_base, acc_test_base]
    reg_vals = [acc_train_reg, acc_val_reg, acc_test_reg]
    x = np.arange(len(labels))
    w = 0.35
    b1 = ax.bar(x - w / 2, base_vals, w, label="Base (sin regularizar)", color="#e0764a")
    b2 = ax.bar(x + w / 2, reg_vals, w,
                label=f"Regularizado (depth={best_params[0]}, mss={best_params[1]})", color="#4a90c4")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Comparación de Accuracy: Modelo Base vs Regularizado")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/comparacion_accuracy.png", dpi=160)
    plt.close()

    # GRÁFICA 2: Heatmap de accuracy en validación (max_depth x min_samples_split)
    depth_labels = [str(d) if d is not None else "Sin límite" for d in candidate_max_depths]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(results_grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(candidate_min_samples_split)))
    ax.set_xticklabels(candidate_min_samples_split)
    ax.set_yticks(range(len(depth_labels)))
    ax.set_yticklabels(depth_labels)
    ax.set_xlabel("min_samples_split")
    ax.set_ylabel("max_depth")
    ax.set_title("Accuracy en validación según hiperparámetros")
    for i in range(results_grid.shape[0]):
        for j in range(results_grid.shape[1]):
            color = "white" if results_grid[i, j] < results_grid.max() * 0.75 else "black"
            ax.text(j, i, f"{results_grid[i, j]:.2f}", ha="center", va="center",
                     color=color, fontsize=7)
    fig.colorbar(im, ax=ax, label="Accuracy validación")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/heatmap_hiperparametros.png", dpi=160)
    plt.close()

    # GRÁFICA 3: Curva de complejidad (accuracy validación vs max_depth,
    # para algunos valores representativos de min_samples_split)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for j, mss in enumerate(candidate_min_samples_split):
        if mss in (2, 4, 6, 10):
            ax.plot(depth_labels, results_grid[:, j], marker="o", label=f"min_samples_split={mss}")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy en validación")
    ax.set_title("Curva de complejidad: Accuracy en validación vs max_depth")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/curva_complejidad.png", dpi=160)
    plt.close()

    # GRÁFICA 4: Curvas de aprendizaje (accuracy train/val vs tamaño de train)
    # Se generan dos paneles: modelo base y modelo regularizado.
    fracs = np.linspace(0.1, 1.0, 10)
    train_accs_base, val_accs_base = [], []
    train_accs_reg, val_accs_reg = [], []

    for frac in fracs:
        sub_train = df_train.sample(frac=frac, random_state=1)

        mb = ID3(max_depth=None, min_samples_split=2)
        tb = mb.fit(sub_train, target, features)
        a_tr, _ = get_acc(tb, sub_train, target)
        a_va, _ = get_acc(tb, df_val, target)
        train_accs_base.append(a_tr); val_accs_base.append(a_va)

        mr = ID3(*best_params)
        tr = mr.fit(sub_train, target, features)
        a_tr2, _ = get_acc(tr, sub_train, target)
        a_va2, _ = get_acc(tr, df_val, target)
        train_accs_reg.append(a_tr2); val_accs_reg.append(a_va2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    axes[0].plot(fracs * len(df_train), train_accs_base, marker="o", label="Train")
    axes[0].plot(fracs * len(df_train), val_accs_base, marker="s", label="Validación")
    axes[0].set_title("Curva de aprendizaje - Modelo BASE\n(sin regularizar)")
    axes[0].set_xlabel("Tamaño del conjunto de entrenamiento")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(fracs * len(df_train), train_accs_reg, marker="o", label="Train")
    axes[1].plot(fracs * len(df_train), val_accs_reg, marker="s", label="Validación")
    axes[1].set_title(f"Curva de aprendizaje - Modelo REGULARIZADO\n(max_depth={best_params[0]}, mss={best_params[1]})")
    axes[1].set_xlabel("Tamaño del conjunto de entrenamiento")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/curva_aprendizaje.png", dpi=160)
    plt.close()

    print(f"\nListo. Las 4 gráficas se guardaron en: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera las gráficas de diagnóstico del árbol ID3.")
    parser.add_argument("--dataset", default="dataset_compra_coche_aumentado.csv",
                         help="Ruta al CSV con los datos (default: dataset_compra_coche_aumentado.csv)")
    parser.add_argument("--target", default="Comprar",
                         help="Nombre de la columna objetivo (default: Comprar)")
    parser.add_argument("--out_dir", default=".",
                         help="Carpeta donde se guardarán las imágenes .png (default: carpeta actual)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Semilla aleatoria para la partición train/val/test (default: 42)")
    args = parser.parse_args()
    main(args.dataset, args.target, args.out_dir, args.seed)
