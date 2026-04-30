"""
Training Pipeline — CICIDS 2017
Trains RandomForest + XGBoost, picks best, saves model + scaler.
Usage: python train.py --data ./data/cicids2017/
"""

import os
import sys
import glob
import pickle
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.utils import resample

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CICIDS 2017 column mapping (raw CSV → our names)
# The dataset has inconsistent column names across files — normalize them
# ─────────────────────────────────────────────
COLUMN_MAP = {
    " Flow Duration":           "flow_duration",
    " Total Fwd Packets":       "total_fwd_pkts",
    " Total Backward Packets":  "total_bwd_pkts",
    " Fwd Packet Length Max":   "fwd_pkt_len_max",
    " Fwd Packet Length Min":   "fwd_pkt_len_min",
    " Fwd Packet Length Mean":  "fwd_pkt_len_mean",
    " Fwd Packet Length Std":   "fwd_pkt_len_std",
    " Bwd Packet Length Max":   "bwd_pkt_len_max",
    " Bwd Packet Length Min":   "bwd_pkt_len_min",
    " Bwd Packet Length Mean":  "bwd_pkt_len_mean",
    " Bwd Packet Length Std":   "bwd_pkt_len_std",
    " Flow Bytes/s":            "flow_bytes_per_s",
    " Flow Packets/s":          "flow_pkts_per_s",
    " Fwd IAT Mean":            "fwd_iat_mean",
    " Fwd IAT Std":             "fwd_iat_std",
    " Fwd IAT Max":             "fwd_iat_max",
    " Fwd IAT Min":             "fwd_iat_min",
    " Bwd IAT Mean":            "bwd_iat_mean",
    " Bwd IAT Std":             "bwd_iat_std",
    " Bwd IAT Max":             "bwd_iat_max",
    " Bwd IAT Min":             "bwd_iat_min",
    " Min Packet Length":       "pkt_len_min",
    " Max Packet Length":       "pkt_len_max",
    " Packet Length Mean":      "pkt_len_mean",
    " Packet Length Std":       "pkt_len_std",
    " Packet Length Variance":  "pkt_len_var",
    " FIN Flag Count":          "fin_flag_cnt",
    " SYN Flag Count":          "syn_flag_cnt",
    " RST Flag Count":          "rst_flag_cnt",
    " PSH Flag Count":          "psh_flag_cnt",
    " ACK Flag Count":          "ack_flag_cnt",
    " URG Flag Count":          "urg_flag_cnt",
    " Down/Up Ratio":           "down_up_ratio",
    " Average Packet Size":     "avg_pkt_size",
    " Fwd Header Length":       "fwd_header_len",
    " Bwd Header Length":       "bwd_header_len",
    " Init_Win_bytes_forward":  "init_win_bytes_fwd",
    " Init_Win_bytes_backward": "init_win_bytes_bwd",
    " Active Min":              "active_min",
    " Active Max":              "active_max",
    " Active Mean":             "active_mean",
    " Idle Min":                "idle_min",
    " Idle Max":                "idle_max",
    " Idle Mean":               "idle_mean",
    " Label":                   "label",
}

FEATURE_COLUMNS = [v for v in COLUMN_MAP.values() if v != "label"]

# CICIDS 2017 attack label → integer class
LABEL_MAP = {
    "BENIGN":                0,
    "DDoS":                  1,
    "PortScan":              2,
    "FTP-Patator":           3,
    "SSH-Patator":           3,   # Both are brute force
    "DoS Hulk":              8,
    "DoS GoldenEye":         8,
    "DoS slowloris":         8,
    "DoS Slowhttptest":      8,
    "Heartbleed":            7,
    "Web Attack – Brute Force": 6,
    "Web Attack – XSS":      6,
    "Web Attack – Sql Injection": 6,
    "Infiltration":          4,
    "Bot":                   5,
}

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_cicids_csvs(data_dir: str) -> pd.DataFrame:
    """Load all CICIDS 2017 CSV files from a directory."""
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        log.error(f"No CSV files found in {data_dir}")
        log.info("Download CICIDS 2017 from: https://www.unb.ca/cic/datasets/ids-2017.html")
        sys.exit(1)

    dfs = []
    for f in csv_files:
        log.info(f"Loading {os.path.basename(f)} ...")
        try:
            df = pd.read_csv(f, low_memory=False)
            df.rename(columns=COLUMN_MAP, inplace=True)
            # Keep only mapped columns
            cols = [c for c in FEATURE_COLUMNS + ["label"] if c in df.columns]
            dfs.append(df[cols])
        except Exception as e:
            log.warning(f"  Skipped {f}: {e}")

    if not dfs:
        log.error("No valid CSVs loaded.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"Total rows loaded: {len(combined):,}")
    return combined

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    log.info("Preprocessing ...")

    # Normalize label strings
    df["label"] = df["label"].str.strip()
    df["label"] = df["label"].map(LABEL_MAP)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

    # Fill missing feature columns with 0
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].copy()

    # Replace inf and NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    X = X.clip(-1e9, 1e9)

    log.info(f"Class distribution:\n{y.value_counts().to_string()}")
    log.info(f"Features: {X.shape[1]} | Samples: {X.shape[0]:,}")
    return X, y

# ─────────────────────────────────────────────
# CLASS BALANCING — undersample majority, oversample minority
# ─────────────────────────────────────────────
def balance_classes(X: pd.DataFrame, y: pd.Series,
                    max_per_class: int = 30_000,
                    min_per_class: int = 1_000) -> tuple:
    log.info("Balancing classes ...")
    dfs = []
    for cls in y.unique():
        mask = y == cls
        X_c = X[mask]
        y_c = y[mask]
        n = len(X_c)
        if n > max_per_class:
            X_c, y_c = resample(X_c, y_c, replace=False, n_samples=max_per_class, random_state=42)
        elif n < min_per_class:
            X_c, y_c = resample(X_c, y_c, replace=True, n_samples=min_per_class, random_state=42)
        dfs.append(pd.concat([X_c, y_c], axis=1))

    combined = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    return combined[FEATURE_COLUMNS], combined["label"]

# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
def train(X_train, y_train):
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        ),
    }

    best_model  = None
    best_score  = 0.0
    best_name   = ""

    for name, m in models.items():
        log.info(f"Training {name} ...")
        m.fit(X_train, y_train)

        # Quick CV score on training data
        cv_scores = cross_val_score(m, X_train, y_train, cv=3,
                                    scoring="f1_weighted", n_jobs=-1)
        mean_f1 = cv_scores.mean()
        log.info(f"  {name} CV F1 (weighted): {mean_f1:.4f} ± {cv_scores.std():.4f}")

        if mean_f1 > best_score:
            best_score = mean_f1
            best_model = m
            best_name  = name

    log.info(f"Best model: {best_name} (F1={best_score:.4f})")
    return best_model, best_name

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
ATTACK_NAMES = {
    0: "BENIGN", 1: "DDoS", 2: "PortScan",
    3: "BruteForce", 4: "Infiltration", 5: "BotNet",
    6: "WebAttack", 7: "Heartbleed", 8: "DoS"
}

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")

    log.info(f"\nAccuracy : {acc:.4f}")
    log.info(f"F1 Score : {f1:.4f}")
    log.info("\nClassification Report:")
    target_names = [ATTACK_NAMES.get(c, str(c)) for c in sorted(y_test.unique())]
    log.info("\n" + classification_report(y_test, y_pred,
             target_names=target_names, zero_division=0))

    # Feature importances (if RF)
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
        log.info("\nTop 10 Important Features:")
        log.info(fi.nlargest(10).to_string())

    return acc, f1

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
def save_artifacts(model, scaler, model_path, scaler_path):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info(f"Saved: {model_path}, {scaler_path}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train IDS model on CICIDS 2017")
    parser.add_argument("--data",   default="./data/", help="Path to CICIDS CSV directory")
    parser.add_argument("--model",  default="traffic_model.pkl")
    parser.add_argument("--scaler", default="scaler.pkl")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--no-balance", action="store_true",
                        help="Skip class balancing (not recommended)")
    args = parser.parse_args()

    log.info("=" * 55)
    log.info("  EtherWatch IDS — Training Pipeline")
    log.info(f"  Data dir : {args.data}")
    log.info("=" * 55)

    # Load
    df = load_cicids_csvs(args.data)

    # Preprocess
    X, y = preprocess(df)

    # Balance
    if not args.no_balance:
        X, y = balance_classes(X, y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size,
        random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    X_train_sc = pd.DataFrame(X_train_sc, columns=FEATURE_COLUMNS)
    X_test_sc  = pd.DataFrame(X_test_sc,  columns=FEATURE_COLUMNS)

    # Train
    best_model, best_name = train(X_train_sc, y_train)

    # Evaluate
    log.info(f"\nEvaluating {best_name} on test set ...")
    acc, f1 = evaluate(best_model, X_test_sc, y_test)

    # Save
    save_artifacts(best_model, scaler, args.model, args.scaler)

    log.info("\n✅ Training complete!")
    log.info(f"   Model: {args.model}")
    log.info(f"   Scaler: {args.scaler}")
    log.info(f"   Accuracy: {acc:.2%} | F1: {f1:.4f}")

if __name__ == "__main__":
    main()
