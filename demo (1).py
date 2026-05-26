#!/usr/bin/env python3
"""
============================================================
  Fake Review Detection — Class Demo
  SVM Classifier | UCI ML Repository Dataset
============================================================

Usage:
  python demo.py                     Run full interactive demo
  python demo.py --retrain           Force retrain and overwrite saved model
  python demo.py --quick             Train on 8K-sample subset (faster, ~30s)

The first run will train the model and save it as fake_review_model.pkl.
Every run after that loads instantly.
============================================================
"""

import sys
import os
import time
import textwrap
import argparse

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour helpers
# ─────────────────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
BG_RED   = "\033[41m"
BG_GREEN = "\033[42m"
BG_BLUE  = "\033[44m"
BG_DARK  = "\033[100m"

def c(text, *codes):
    return "".join(codes) + str(text) + RESET

def banner():
    print()
    print(c("╔══════════════════════════════════════════════════════════════╗", CYAN, BOLD))
    print(c("║", CYAN, BOLD) + c("       FAKE REVIEW DETECTION  —  CLASS DEMO               ", WHITE, BOLD) + c("   ║", CYAN, BOLD))
    print(c("║", CYAN, BOLD) + c("       SVM Classifier  |  UCI ML Repository Dataset        ", DIM)         + c("   ║", CYAN, BOLD))
    print(c("╚══════════════════════════════════════════════════════════════╝", CYAN, BOLD))
    print()

def section(title):
    print()
    print(c(f"  ▶  {title}", CYAN, BOLD))
    print(c("  " + "─" * 58, DIM))

def spinner_wait(msg, secs=0.8):
    frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    end_time = time.time() + secs
    i = 0
    while time.time() < end_time:
        print(f"\r  {c(frames[i % len(frames)], CYAN)} {msg}", end="", flush=True)
        time.sleep(0.1)
        i += 1
    print(f"\r  {c('✓', GREEN)} {msg}" + " " * 10)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATASET_PATH = "fake reviews dataset.csv"
MODEL_PATH   = "fake_review_model.pkl"
BEST_PARAMS  = dict(C=10, kernel="rbf", gamma="scale")

# Pre-canned demo examples (label, review_text, category, rating)
DEMO_EXAMPLES = [
    (
        "FAKE",
        "This product is absolutely amazing! Best purchase I've ever made. "
        "Five stars all the way! Everyone should buy this immediately. Perfect in every way!",
        "Home_and_Kitchen_5", 5
    ),
    (
        "REAL",
        "Bought this for my kitchen. It works as advertised but the handle gets warm "
        "after about 10 minutes of use. Not a dealbreaker, but worth knowing. "
        "Shipping was fast and packaging was solid.",
        "Home_and_Kitchen_5", 4
    ),
    (
        "FAKE",
        "I love love love this book! It changed my life completely. "
        "The author is a genius and every single page is pure gold. "
        "I have recommended it to all my friends and family. Truly life-changing!",
        "Kindle_Store_5", 5
    ),
    (
        "REAL",
        "Decent read, though the pacing drags in the second half. "
        "The first few chapters are gripping but the ending felt rushed. "
        "I'd still recommend it to fans of the genre, just manage expectations.",
        "Kindle_Store_5", 3
    ),
    (
        "FAKE",
        "This toy is the best toy ever made for children. My kids absolutely love it "
        "and play with it every single day. Amazing quality, amazing price, amazing everything!",
        "Toys_and_Games_5", 5
    ),
]

# ─────────────────────────────────────────────────────────────────────────────
# BUILD PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline(svc_params):
    rating_pipeline = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler()
    )
    category_pipeline = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="missing"),
        OneHotEncoder(handle_unknown="ignore")
    )
    text_pipeline = make_pipeline(
        TfidfVectorizer(stop_words="english", max_features=5000)
    )
    preprocessing = ColumnTransformer([
        ("ratings",    rating_pipeline,    ["rating"]),
        ("categories", category_pipeline,  ["category"]),
        ("text",       text_pipeline,      "text_"),
    ])
    return make_pipeline(preprocessing, SVC(**svc_params))

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(quick=False):
    section("Loading Dataset")

    if not os.path.exists(DATASET_PATH):
        print(c(f"\n  ERROR: Cannot find '{DATASET_PATH}'", RED, BOLD))
        print(c("  Make sure the CSV is in the same folder as this script.", DIM))
        sys.exit(1)

    df = pd.read_csv(DATASET_PATH)
    df.dropna(subset=["text_"], inplace=True)
    df["rating"] = df["rating"].astype(int)

    le = LabelEncoder()
    le.fit(["CG", "OR"])
    df["label"] = le.transform(df["label"])

    if quick:
        # Stratified sample for quick mode
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), 4000), random_state=42),
            include_groups=False
        ).reset_index(drop=True)
        print(c(f"  [Quick mode] Sampled 8,000 reviews for faster training.", YELLOW))

    print(f"  {c('Dataset:', BOLD)} {c(DATASET_PATH, WHITE)}")
    print(f"  {c('Total reviews:', BOLD)} {c(f'{len(df):,}', CYAN)}")
    print(f"  {c('Fake (CG):', BOLD)} {c(f'{(df.label==0).sum():,}', RED)}"
          f"  {c('Real (OR):', BOLD)} {c(f'{(df.label==1).sum():,}', GREEN)}")
    print(f"  {c('Categories:', BOLD)} {c(df.category.nunique(), CYAN)}")

    # Train / test split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in splitter.split(df, df["label"]):
        train_df = df.loc[train_idx]
        test_df  = df.loc[test_idx]

    x_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    x_test  = test_df.drop(columns=["label"])
    y_test  = test_df["label"]

    print(f"\n  {c('Train:', BOLD)} {len(x_train):,} samples  "
          f"{c('Test:', BOLD)} {len(x_test):,} samples  "
          f"{c('(80/20 stratified split)', DIM)}")
    return x_train, y_train, x_test, y_test

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN OR LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
def get_model(x_train, y_train, retrain=False, quick=False):
    section("Model")

    if not retrain and os.path.exists(MODEL_PATH):
        print(f"  {c('Saved model found →', GREEN)} {MODEL_PATH}")
        spinner_wait("Loading model...", 0.5)
        model = joblib.load(MODEL_PATH)
        print(f"  {c('Model ready!', GREEN, BOLD)}")
        return model

    tag = " (quick/subset)" if quick else ""
    print(f"  {c('No saved model found. Training SVM now' + tag + '...', YELLOW)}")
    print(f"  {c('Parameters:', BOLD)} C={BEST_PARAMS['C']},  "
          f"kernel={BEST_PARAMS['kernel']},  gamma={BEST_PARAMS['gamma']}")
    print(c("\n  This may take 2–5 minutes on the full dataset. Please wait...\n", DIM))

    model = build_pipeline(BEST_PARAMS)

    start = time.time()
    frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    # Fit in foreground; show a simple elapsed-time spinner
    import threading
    done_event = threading.Event()

    def fit_thread():
        model.fit(x_train, y_train)
        done_event.set()

    t = threading.Thread(target=fit_thread, daemon=True)
    t.start()

    i = 0
    while not done_event.is_set():
        elapsed = time.time() - start
        print(f"\r  {c(frames[i % len(frames)], CYAN)} Training...  "
              f"{c(f'{elapsed:.0f}s elapsed', DIM)}", end="", flush=True)
        time.sleep(0.12)
        i += 1

    elapsed = time.time() - start
    print(f"\r  {c('✓', GREEN, BOLD)} Training complete in {c(f'{elapsed:.1f}s', CYAN)}!" + " " * 20)

    joblib.dump(model, MODEL_PATH)
    print(f"  {c('Model saved →', GREEN)} {MODEL_PATH}  "
          f"{c('(loads instantly next time)', DIM)}")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# SHOW PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def show_metrics(model, x_test, y_test):
    section("Performance on Test Set")

    y_pred = model.predict(x_test)
    acc  = accuracy_score(y_test, y_pred)
    cm   = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Fake","Real"], output_dict=True)

    # Accuracy bar
    bar_len = 40
    filled  = int(round(acc * bar_len))
    bar = c("█" * filled, GREEN, BOLD) + c("░" * (bar_len - filled), DIM)
    print(f"\n  {c('Accuracy:', BOLD)}  [{bar}]  {c(f'{acc*100:.1f}%', GREEN, BOLD)}\n")

    # Metrics table
    header = f"  {'':12}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'Support':>8}"
    print(c(header, DIM))
    print(c("  " + "─" * 50, DIM))
    for cls in ["Fake", "Real"]:
        row = report[cls]
        prec    = f"{row['precision']:.2%}"
        rec     = f"{row['recall']:.2%}"
        f1      = f"{row['f1-score']:.2%}"
        support = str(int(row['support']))
        print(f"  {c(cls, BOLD):21}  "
              f"{c(prec, CYAN):20}  "
              f"{c(rec, CYAN):19}  "
              f"{c(f1, CYAN):19}  "
              f"{c(support, WHITE):>7}")
    print(c("  " + "─" * 50, DIM))

    # Confusion matrix
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])
    print(f"\n  {c('Confusion Matrix:', BOLD)}")
    print(f"  {c('              Pred FAKE   Pred REAL', DIM)}")
    print(f"  {c('Actual FAKE', BOLD)}   {c(f'{tp:>6,}', GREEN)}      {c(f'{fn:>6,}', RED)}")
    print(f"  {c('Actual REAL', BOLD)}   {c(f'{fp:>6,}', RED)}      {c(f'{tn:>6,}', GREEN)}")
    print()
    print(f"  {c('✓ True Positive (Fake detected):',  DIM)} {c(f'{tp:>5,}', GREEN)}")
    print(f"  {c('✗ False Negative (Fake missed):',   DIM)} {c(f'{fn:>5,}', RED)}")

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT SINGLE REVIEW
# ─────────────────────────────────────────────────────────────────────────────
def predict_review(model, text, category="Home_and_Kitchen_5", rating=5):
    row = pd.DataFrame([{"category": category, "rating": rating, "text_": text}])
    pred = model.predict(row)[0]
    score = model.decision_function(row)[0]   # + = fake (class 0), depends on label order
    label  = "FAKE" if pred == 0 else "REAL"
    confidence = abs(score)
    return label, confidence, score

def print_prediction(label, confidence, score, true_label=None, text=None):
    if text:
        wrapped = textwrap.fill(text, width=60, initial_indent="  ", subsequent_indent="  ")
        print(c(wrapped, DIM))
        print()

    if label == "FAKE":
        verdict_str = f"  {BG_RED}{BOLD}   FAKE REVIEW   {RESET}"
    else:
        verdict_str = f"  {BG_GREEN}{BOLD}   REAL REVIEW   {RESET}"

    # Sigmoid-based confidence: maps SVM decision score → 0–100%
    conf_pct = round((1 / (1 + np.exp(-confidence))) * 100 - 50) * 2
    conf_bar_len = 30
    conf_filled  = int(round(conf_pct / 100 * conf_bar_len))
    conf_bar = c("█" * conf_filled, CYAN) + c("░" * (conf_bar_len - conf_filled), DIM)

    print(f"{verdict_str}   {c('Confidence:', DIM)} [{conf_bar}]  {c(f'{conf_pct:.0f}%', CYAN)}")

    if true_label is not None:
        match = label == true_label
        status = c("✓  Correct", GREEN, BOLD) if match else c("✗  Incorrect", RED, BOLD)
        print(f"  {c('True label:', DIM)} {c(true_label, WHITE, BOLD)}   {status}")

# ─────────────────────────────────────────────────────────────────────────────
# DEMO EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────
def run_demo_examples(model):
    section("Pre-loaded Demo Examples")
    print(c("  Classifying 5 example reviews...", DIM))

    correct = 0
    for i, (true_label, text, cat, rating) in enumerate(DEMO_EXAMPLES, 1):
        print(f"\n  {c(f'Example {i}/{len(DEMO_EXAMPLES)}', BOLD)}  "
              f"{c(f'[Category: {cat}  |  Rating: {rating}/5]', DIM)}")
        print(c("  " + "─" * 58, DIM))
        label, conf, score = predict_review(model, text, cat, rating)
        print_prediction(label, conf, score, true_label=true_label, text=text)
        if label == true_label:
            correct += 1
        input(c("\n  Press Enter for next example...", DIM))

    print(f"\n  {c('Demo examples:', BOLD)} {c(f'{correct}/{len(DEMO_EXAMPLES)} correct', GREEN, BOLD)}")

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────────────────────────────────────
CATEGORIES = [
    "Home_and_Kitchen_5", "Electronics_5", "Kindle_Store_5",
    "Books_5", "Toys_and_Games_5", "Sports_and_Outdoors_5",
    "Clothing_Shoes_and_Jewelry_5", "Movies_and_TV_5",
    "Pet_Supplies_5", "Tools_and_Home_Improvement_5"
]

def interactive_mode(model):
    section("Interactive Demo  —  Type Any Review")
    print(c("  Type or paste a review, then press Enter twice to classify it.", DIM))
    print(c("  Type  'quit'  or press Ctrl+C to exit.\n", DIM))

    while True:
        try:
            # Get review text
            print(c("  ┌── Your review (press Enter twice when done) ──", CYAN))
            lines = []
            while True:
                line = input("  │ ")
                if line.strip().lower() in ("quit", "exit", "q"):
                    print(c("\n  Goodbye!\n", CYAN))
                    return
                if line == "" and lines:
                    break
                lines.append(line)

            text = " ".join(lines).strip()
            if not text:
                continue

            # Get star rating
            print()
            rating_str = input(c("  Star rating (1–5, or press Enter for 5): ", CYAN)).strip()
            rating = int(rating_str) if rating_str.isdigit() and 1 <= int(rating_str) <= 5 else 5

            # Get category
            print()
            print(c("  Categories:", DIM))
            for idx, cat in enumerate(CATEGORIES, 1):
                print(c(f"    {idx:>2}. {cat}", DIM))
            cat_str = input(c("  Category number (or press Enter for default): ", CYAN)).strip()
            category = CATEGORIES[int(cat_str) - 1] if cat_str.isdigit() and 1 <= int(cat_str) <= len(CATEGORIES) else CATEGORIES[0]

            print()
            print(c("  ─" * 30, DIM))
            label, conf, score = predict_review(model, text, category, rating)
            print_prediction(label, conf, score)
            print(c("  ─" * 30, DIM))
            print()

        except (KeyboardInterrupt, EOFError):
            print(c("\n\n  Goodbye!\n", CYAN))
            break

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fake Review Detection Demo")
    parser.add_argument("--retrain", action="store_true", help="Force retrain the model")
    parser.add_argument("--quick",   action="store_true", help="Train on 8K-sample subset (faster)")
    args = parser.parse_args()

    banner()

    x_train, y_train, x_test, y_test = load_dataset(quick=args.quick)
    model = get_model(x_train, y_train, retrain=args.retrain, quick=args.quick)
    show_metrics(model, x_test, y_test)
    run_demo_examples(model)
    interactive_mode(model)

if __name__ == "__main__":
    main()
