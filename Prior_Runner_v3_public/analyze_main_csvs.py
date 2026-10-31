#!/usr/bin/env python3
"""Resume los CSV que genera tu main (uno por imagen: <img>_combined_metrics_summary.csv).

Uso:
  python analyze_main_csvs.py "enhanced_results/*/*_combined_metrics_summary.csv" --ref HPHH --out-tex tabla_main.tex

1) Promedia las 30 corridas DENTRO de cada imagen.
2) Reporta media y desviacion entre imagenes por optimizador.
3) Wilcoxon (rangos con signo, pareado por imagen) del optimizador --ref contra cada otro.
   Con menos de 5 imagenes no se calcula (no habria poder estadistico).
"""
import argparse, glob, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

COLS = ["SSIM", "PSNR", "Entropy", "Avg_Gradient", "CII", "Time (s)"]

ap = argparse.ArgumentParser()
ap.add_argument("pattern")
ap.add_argument("--ref", default="HPHH")
ap.add_argument("--out-tex", default="tabla_main.tex")
ap.add_argument("--out-csv", default="resumen_main.csv")
a = ap.parse_args()

files = sorted(glob.glob(a.pattern))
if not files:
    raise SystemExit("No encontre CSV con ese patron.")
df = pd.concat([pd.read_csv(f, dtype={"Image": str}) for f in files], ignore_index=True)
cols = [c for c in COLS if c in df.columns]
per_img = df.groupby(["Optimizer", "Image"])[cols].mean().reset_index()
n_img = per_img["Image"].nunique()
print(f"{len(files)} archivos, {n_img} imagenes, {df['Optimizer'].nunique()} optimizadores\n")

g = per_img.groupby("Optimizer")[cols]
summ = g.agg(["mean", "std"])
summ.columns = [f"{m}_{s}" for m, s in summ.columns]
summ = summ.reset_index()
summ.to_csv(a.out_csv, index=False)
print(summ.round(4).to_string(index=False))

if n_img >= 5 and a.ref in per_img["Optimizer"].unique():
    print(f"\n== Wilcoxon por imagen: {a.ref} vs cada optimizador (p<0.05 con *) ==")
    ref = per_img[per_img.Optimizer == a.ref].set_index("Image")
    for o in sorted(per_img.Optimizer.unique()):
        if o == a.ref:
            continue
        ot = per_img[per_img.Optimizer == o].set_index("Image")
        common = ref.index.intersection(ot.index)
        line = []
        for m in cols:
            d = ref.loc[common, m] - ot.loc[common, m]
            if np.allclose(d, 0):
                line.append(f"{m}: igual"); continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = wilcoxon(d).pvalue
            line.append(f"{m}: p={p:.3f}{'*' if p < 0.05 else ''}")
        print(f"  {a.ref} vs {o:<11} " + " | ".join(line))
else:
    print(f"\n[aviso] Wilcoxon por imagen requiere >=5 imagenes (hay {n_img}).")

def cell(r, m):
    s = "" if pd.isna(r[m + "_std"]) else f" $\\pm$ {r[m + '_std']:.3f}"
    return f"{r[m + '_mean']:.3f}{s}"
show = [c for c in ("SSIM", "PSNR", "Entropy", "Avg_Gradient", "Time (s)") if c in cols]
head = " & ".join(["Optimizador"] + [c.replace("_", " ") for c in show])
rows = [" & ".join([r["Optimizer"]] + [cell(r, m) for m in show]) + r" \\" for _, r in summ.iterrows()]
tex = "\\begin{tabular}{l" + "c" * len(show) + "}\n\\hline\n" + head + " \\\\\n\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n"
open(a.out_tex, "w").write(tex)
print(f"\n[ok] {a.out_csv} y {a.out_tex}")
