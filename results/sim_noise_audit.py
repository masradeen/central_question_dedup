# sim_noise_audit.py
# Cara pakai:
# 1) letakkan similarity_pairs.csv di folder yang sama
# 2) jalankan: python sim_noise_audit.py --sample_out sampel_to_label.csv
# 3) lakukan labeling manual pada sampel_to_label.csv (tambah kolom 'label' dengan 1/0)
# 4) jalankan: python sim_noise_audit.py --labelled labelled_sample.csv

import argparse, os, math
import pandas as pd
import numpy as np
from scipy import stats

def detect_similarity_col(df):
    for c in ['similarity','score','cosine','sim','similarity_score']:
        if c in df.columns:
            return c
    # fallback: the numeric column with values in [0,1] and not id cols
    numeric = df.select_dtypes(include=np.number)
    for col in numeric.columns:
        vmin, vmax = numeric[col].min(), numeric[col].max()
        if 0.0 <= vmin and vmax <= 1.0 and col not in ['id','year']:
            return col
    raise ValueError("Kolom similarity tidak ditemukan. Pastikan file mengandung kolom 'similarity' atau sejenis.")

def stratified_sample(df, sim_col, bins, per_bin):
    df['bin'] = pd.cut(df[sim_col], bins=bins, include_lowest=True, right=False)
    samples = []
    for b in df['bin'].cat.categories:
        bucket = df[df['bin'] == b]
        n = min(len(bucket), per_bin)
        if n <= 0:
            continue
        samples.append(bucket.sample(n=n, random_state=42))
    out = pd.concat(samples).drop(columns=['bin'])
    return out

def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return (0,0)
    z = stats.norm.ppf(1 - alpha/2)
    phat = k/n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = z * math.sqrt((phat*(1-phat)/n + z*z/(4*n*n))) / denom
    return (max(0, center-half), min(1, center+half))

def evaluate_labelled(df_labelled, sim_col):
    assert 'label' in df_labelled.columns, "File label harus mengandung kolom 'label' (0/1)."
    df = df_labelled.copy()
    df['label'] = df['label'].astype(int)
    # precision overall (on the sampled set)
    tp = df['label'].sum()
    fp = len(df) - tp
    prec = tp / (tp + fp) if (tp+fp)>0 else 0.0
    ci_low, ci_high = wilson_ci(tp, tp+fp)
    # per-bin
    bins = np.linspace(df[sim_col].min(), df[sim_col].max(), 6)  # 5 bins
    df['bin'] = pd.cut(df[sim_col], bins=bins, include_lowest=True, right=False)
    perbin = []
    for b in df['bin'].cat.categories:
        bucket = df[df['bin']==b]
        k = bucket['label'].sum()
        n = len(bucket)
        p = k/n if n>0 else None
        l,h = wilson_ci(k,n)
        perbin.append({'bin': str(b), 'n': n, 'precision': p, 'ci_low': l, 'ci_high': h})
    perbin_df = pd.DataFrame(perbin)
    return {'precision': prec, 'ci': (ci_low,ci_high), 'tp':tp,'fp':fp,'perbin': perbin_df}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='similarity_pairs.csv', help='CSV dengan pasangan similarity')
    ap.add_argument('--sample_out', help='Path output CSV untuk sample yang perlu di-label')
    ap.add_argument('--labelled', help='Jika sudah dilabel, berikan path ke CSV labelled untuk evaluasi')
    ap.add_argument('--per_bin', type=int, default=80, help='Jumlah sample per bin similarity saat sampling')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"File input tidak ditemukan: {args.input}")

    df = pd.read_csv(args.input)
    sim_col = detect_similarity_col(df)
    print("Detected similarity column:", sim_col)

    if args.sample_out:
        # buat bins from threshold 0.0..1.0 atau spesifik 0.78..1.0
        bins = np.linspace(0.0, 1.0, 10)  # 9 bins
        sample = stratified_sample(df, sim_col, bins=bins, per_bin=args.per_bin)
        sample.to_csv(args.sample_out, index=False)
        print(f"Sample ditulis ke {args.sample_out} (total rows: {len(sample)})")
        print("Silakan tambahkan kolom 'label' (1 = duplicate, 0 = not) lalu jalankan --labelled")
        return

    if args.labelled:
        df_lab = pd.read_csv(args.labelled)
        stats = evaluate_labelled(df_lab, sim_col)
        print("=== EVALUATION ===")
        print(f"Precision (sample) = {stats['precision']:.3f}  (95% CI: {stats['ci'][0]:.3f} - {stats['ci'][1]:.3f})")
        print(f"TP = {stats['tp']}  FP = {stats['fp']}  noise_rate ≈ {1-stats['precision']:.3f}")
        print("\nPrecision per bin:")
        print(stats['perbin'].to_string(index=False))
        # save perbin
        stats['perbin'].to_csv('precision_per_bin.csv', index=False)
        print("\nPer-bin saved to precision_per_bin.csv")
        return

    print("Tidak ada action. Gunakan --sample_out atau --labelled.")

if __name__ == '__main__':
    main()
