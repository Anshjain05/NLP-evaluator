# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import csv
from urllib.parse import urljoin
import time
import random
import matplotlib.pyplot as plt
import io
import logging
from typing import Optional, Tuple, List
import os
from ftfy import fix_text

API_KEY = st.secrets["GOOGLE_FACTCHECK_API_KEY"]

# ----------------------------------------------
# Small cleaner using ftfy + whitespace collapse
# ----------------------------------------------
def clean(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    try:
        s = fix_text(s)
    except Exception:
        pass
    return " ".join(s.split()).strip()

# ----------------------------------------
# Create a helper to normalize text labels
# ----------------------------------------

def normalize_label(label_text: str) -> str:
    """Convert label text into simplified True/False/Mixed."""
    if not label_text:
        return "Unknown"
    text = label_text.lower()
    if any(x in text for x in ["true", "accurate", "fact", "real"]):
        return "True"
    if any(x in text for x in ["false", "fake", "pants", "incorrect"]):
        return "False"
    if any(x in text for x in ["half", "mixed", "partly"]):
        return "Mixed"
    return "Unknown"

# ---------------------------
# Google Fact Check
# ---------------------------
def get_fact_check_results(query):
    """Fetch fact-check results for the given query from Google Fact Check Tools API."""
    if not API_KEY:
        return []
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {"query": query, "key": API_KEY}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        claims = data.get("claims", [])
        results = []
        for claim in claims:
            reviews = claim.get("claimReview", [])
            for r in reviews:
                results.append({
                    "publisher": r.get("publisher", {}).get("name", "Unknown"),
                    "title": r.get("title", ""),
                    "rating": r.get("textualRating", "No Rating"),
                    "url": r.get("url", "")
                })
        return results
    except Exception as e:
        return [{"publisher": "Error", "title": str(e), "rating": "", "url": ""}]

# Imbalanced learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# NLP & ML
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ---------------------------
# CONFIG
# ---------------------------
SCRAPED_DATA_PATH = "politifact_data.csv"
N_SPLITS = 5
MAX_PAGES = 100  # safety
REQUEST_RETRIES = 3
REQUEST_BACKOFF = 2  # seconds base

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# SpaCy loader (cached)
# ---------------------------
@st.cache_resource
def load_spacy_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError as e:
        st.error("SpaCy model 'en_core_web_sm' not found. Add the wheel URL to requirements.txt in your deploy environment.")
        st.code("""
# Example to add in requirements.txt (adapt version if needed):
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
imbalanced-learn
        """, language="text")
        raise e

try:
    NLP_MODEL = load_spacy_model()
except Exception:
    st.stop()

stop_words = STOP_WORDS
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

# ---------------------------
# Robust GET with retries
# ---------------------------
def safe_get(url: str, timeout: int = 15) -> Optional[requests.Response]:
    backoff = REQUEST_BACKOFF
    for attempt in range(REQUEST_RETRIES):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            logger.warning(f"Request error ({attempt+1}/{REQUEST_RETRIES}) for {url}: {e}")
            time.sleep(backoff)
            backoff *= 2
    return None

# ---------------------------
# 1) SCRAPER with ftfy cleaning
# ---------------------------
@st.cache_data(ttl=60*60*24)  # cache for a day per arg set
def scrape_data_by_date_range(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    base_url = "https://www.politifact.com/factchecks/list/"
    current_url = base_url
    seen_urls = set()
    rows = []
    page_count = 0

    while current_url and page_count < MAX_PAGES:
        page_count += 1
        if current_url in seen_urls:
            logger.info("Detected repeated page, stopping to avoid infinite loop.")
            break
        seen_urls.add(current_url)

        resp = safe_get(current_url, timeout=15)
        if resp is None:
            st.warning(f"Failed to fetch {current_url} after retries; stopping scraper.")
            break

        #  Force correct text decoding
        try:
            raw_html = resp.content  # get bytes, not pre-decoded text
            decoded_html = raw_html.decode("utf-8", errors="replace")
        except Exception:
            decoded_html = resp.text  # fallback

        # Parse with BeautifulSoup using manually decoded HTML
        soup = BeautifulSoup(decoded_html, "html.parser")

        items = soup.find_all("li", class_="o-listicle__item")
        if not items:
            logger.info("No items found on page; stopping.")
            break

        stop_if_older = False
        for card in items:
            # date extraction
            date_div = card.find("div", class_="m-statement__desc")
            date_text = date_div.get_text(" ", strip=True) if date_div else ""
            claim_date = None
            if date_text:
                match = re.search(r"stated on ([A-Za-z]+\s+\d{1,2},\s+\d{4})", date_text)
                if match:
                    try:
                        claim_date = pd.to_datetime(match.group(1), format="%B %d, %Y")
                    except Exception:
                        claim_date = pd.to_datetime(match.group(1), errors='coerce')

            if claim_date is None:
                continue

            if claim_date < start_date:
                stop_if_older = True
                break

            if not (start_date <= claim_date <= end_date):
                continue

            # statement
            statement = None
            statement_block = card.find("div", class_="m-statement__quote")
            if statement_block:
                a = statement_block.find("a", href=True)
                if a:
                    statement = clean(a.get_text(" ", strip=True))

            # source / speaker
            source = None
            source_a = card.find("a", class_="m-statement__name")
            if source_a:
                source = clean(source_a.get_text(" ", strip=True))

            # author
            author = None
            footer = card.find("footer", class_="m-statement__footer")
            if footer:
                text = footer.get_text(" ", strip=True)
                m = re.search(r"By\s+([^•\n\r]+)", text)
                if m:
                    author = clean(m.group(1).strip())
                else:
                    parts = text.split("•")
                    if parts:
                        author = clean(parts[0].replace("By", "").strip())

            # label
            label = None
            label_img = card.find("img", alt=True)
            if label_img and 'alt' in label_img.attrs:
                label = clean(label_img['alt'].replace('-', ' ').title())

            rows.append({
                "author": author,
                "statement": statement,
                "source": source,
                "date": claim_date.strftime("%Y-%m-%d"),
                "label": label
            })

        if stop_if_older:
            break

        # next page
        next_link = soup.find("a", class_="c-button c-button--hollow", string=re.compile(r"Next", re.I))
        if next_link and next_link.get("href"):
            current_url = urljoin(base_url, next_link['href'])
        else:
            break

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["statement", "label"])
    if not df.empty:
        df.to_csv(SCRAPED_DATA_PATH, index=False)
    return df

# ---------------------------
# 2) Feature functions (batch)
# ---------------------------
def lexical_features_batch(texts: List[str], nlp) -> List[str]:
    processed = []
    for doc in nlp.pipe(texts, disable=["ner", "parser"]):
        toks = [token.lemma_.lower() for token in doc if token.is_alpha and token.lemma_.lower() not in stop_words]
        processed.append(" ".join(toks))
    return processed

def syntactic_features_batch(texts: List[str], nlp) -> List[str]:
    processed = []
    for doc in nlp.pipe(texts, disable=["ner"]):
        pos = " ".join([token.pos_ for token in doc])
        processed.append(pos)
    return processed

def semantic_features_batch(texts: List[str]) -> pd.DataFrame:
    out = []
    for t in texts:
        b = TextBlob(t)
        out.append([b.sentiment.polarity, b.sentiment.subjectivity])
    return pd.DataFrame(out, columns=["polarity", "subjectivity"])

def discourse_features_batch(texts: List[str], nlp) -> List[str]:
    processed = []
    for doc in nlp.pipe(texts, disable=["ner"]):
        sents = [sent.text.strip() for sent in doc.sents]
        first_words = " ".join([s.split()[0].lower() for s in sents if len(s.split()) > 0])
        processed.append(f"{len(sents)} {first_words}")
    return processed

def pragmatic_features_batch(texts: List[str]) -> pd.DataFrame:
    rows = []
    for t in texts:
        tl = t.lower()
        rows.append([tl.count(w) for w in pragmatic_words])
    return pd.DataFrame(rows, columns=pragmatic_words)

# ---------------------------
# 3) Feature extraction dispatcher
# ---------------------------
def apply_feature_extraction(X_series: pd.Series, phase: str, nlp) -> Tuple[np.ndarray, Optional[object]]:
    X_texts = X_series.astype(str).tolist()
    if phase == "Lexical & Morphological":
        X_proc = lexical_features_batch(X_texts, nlp)
        # ✅ dynamically adjust min_df to avoid "max_df < min_df" error
        min_df_value = 1 if len(X_proc) < 3 else 2
        vect = CountVectorizer(binary=True, ngram_range=(1, 2), min_df=min_df_value)
        X_feat = vect.fit_transform(X_proc)
        return X_feat, vect

    if phase == "Syntactic":
        X_proc = syntactic_features_batch(X_texts, nlp)
        vect = TfidfVectorizer(max_features=5000)
        X_feat = vect.fit_transform(X_proc)
        return X_feat, vect

    if phase == "Semantic":
        df = semantic_features_batch(X_texts)
        return df.values, None

    if phase == "Discourse":
        X_proc = discourse_features_batch(X_texts, nlp)
        vect = CountVectorizer(ngram_range=(1,2), max_features=5000)
        X_feat = vect.fit_transform(X_proc)
        return X_feat, vect

    if phase == "Pragmatic":
        df = pragmatic_features_batch(X_texts)
        return df.values, None

    raise ValueError("Unknown phase")

# ---------------------------
# 4) Model helpers & evaluation
# ---------------------------
def get_models_dict():
    return {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, class_weight='balanced'),
        "SVM": SVC(kernel='linear', C=0.5, random_state=42, class_weight='balanced', probability=False)
    }

def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    REAL_LABELS = ["True", "No Flip", "Mostly True", "Half Flip", "Half True"]
    FAKE_LABELS = ["False", "Barely True", "Pants On Fire", "Full Flop"]

    def map_label(l):
        if pd.isna(l):
            return np.nan
        l = str(l).strip()
        if l in REAL_LABELS:
            return 1
        if l in FAKE_LABELS:
            return 0
        low = l.lower()
        if "true" in low and "mostly" not in low and "half" not in low:
            return 1
        if "false" in low or "pants" in low or "fire" in low:
            return 0
        return np.nan

    df = df.copy()
    df["target_label"] = df["label"].apply(map_label)
    return df

def evaluate_models(df: pd.DataFrame, selected_phase: str, nlp) -> pd.DataFrame:
    df = create_binary_target(df)
    df = df.dropna(subset=["target_label"])
    df = df[df["statement"].astype(str).str.len() > 10]

    X_raw = df["statement"].astype(str)
    y_raw = df["target_label"].astype(int)

    if len(np.unique(y_raw)) < 2:
        st.error("Only one class present after mapping — adjust data or date range.")
        return pd.DataFrame()

    X_features_full, vectorizer = apply_feature_extraction(X_raw, selected_phase, nlp)

    if isinstance(X_features_full, np.ndarray):
        X_full = X_features_full
    else:
        X_full = X_features_full

    models = get_models_dict()
    results = []

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    X_list = X_raw.tolist()

    for name, model in models.items():
        st.caption(f"Training {name}...")
        fold_acc, fold_f1, fold_prec, fold_rec = [], [], [], []
        train_times, infer_times = [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y_raw)), y_raw)):
            X_train_raw = pd.Series([X_list[i] for i in train_idx])
            X_test_raw = pd.Series([X_list[i] for i in test_idx])
            y_train = y_raw.values[train_idx]
            y_test = y_raw.values[test_idx]

            if vectorizer is not None:
                if selected_phase == "Lexical & Morphological":
                    X_train_proc = lexical_features_batch(X_train_raw.tolist(), nlp)
                    X_test_proc  = lexical_features_batch(X_test_raw.tolist(), nlp)
                elif selected_phase == "Syntactic":
                    X_train_proc = syntactic_features_batch(X_train_raw.tolist(), nlp)
                    X_test_proc  = syntactic_features_batch(X_test_raw.tolist(), nlp)
                elif selected_phase == "Discourse":
                    X_train_proc = discourse_features_batch(X_train_raw.tolist(), nlp)
                    X_test_proc  = discourse_features_batch(X_test_raw.tolist(), nlp)
                else:
                    X_train_proc = X_train_raw.tolist()
                    X_test_proc  = X_test_raw.tolist()

                X_train = vectorizer.transform(X_train_proc)
                X_test  = vectorizer.transform(X_test_proc)
            else:
                if selected_phase == "Semantic":
                    X_train = semantic_features_batch(X_train_raw.tolist()).values
                    X_test  = semantic_features_batch(X_test_raw.tolist()).values
                elif selected_phase == "Pragmatic":
                    X_train = pragmatic_features_batch(X_train_raw.tolist()).values
                    X_test  = pragmatic_features_batch(X_test_raw.tolist()).values
                else:
                    X_train = X_train_raw.values.reshape(-1, 1)
                    X_test  = X_test_raw.values.reshape(-1, 1)

            start_train = time.time()
            try:
                if name == "Naive Bayes":
                    Xt_fit = np.abs(X_train).astype(float)
                    model.fit(Xt_fit, y_train)
                    clf = model
                else:
                    pipeline = ImbPipeline([("smote", SMOTE(random_state=42, k_neighbors=3)), ("clf", model)])
                    pipeline.fit(X_train, y_train)
                    clf = pipeline

                train_time = time.time() - start_train
                start_inf = time.time()
                y_pred = clf.predict(X_test)
                infer_time = (time.time() - start_inf) * 1000.0

                fold_acc.append(accuracy_score(y_test, y_pred))
                fold_f1.append(f1_score(y_test, y_pred, average="weighted", zero_division=0))
                fold_prec.append(precision_score(y_test, y_pred, average="weighted", zero_division=0))
                fold_rec.append(recall_score(y_test, y_pred, average="weighted", zero_division=0))
                train_times.append(train_time)
                infer_times.append(infer_time)
            except Exception as e:
                st.warning(f"Fold {fold+1} failed for {name}: {e}")
                fold_acc.append(0); fold_f1.append(0); fold_prec.append(0); fold_rec.append(0)
                train_times.append(0); infer_times.append(9999)

        results.append({
            "Model": name,
            "Accuracy": np.mean(fold_acc) * 100,
            "F1-Score": np.mean(fold_f1),
            "Precision": np.mean(fold_prec),
            "Recall": np.mean(fold_rec),
            "Training Time (s)": round(np.mean(train_times), 3),
            "Inference Latency (ms)": round(np.mean(infer_times), 3)
        })

    return pd.DataFrame(results)
# -----------------------------------------
# Creating a simple model prediction function
# -----------------------------------------

def ai_predict_label(statement: str, nlp) -> str:
    """Use your existing NLP + trained model to predict if statement is True/False."""
    try:
        # Use Lexical & Morphological phase as baseline
        phase = "Lexical & Morphological"

        # Check if vectorizer and model exist in session
        if "ai_vectorizer" not in st.session_state or "ai_model" not in st.session_state:
            # Train once and save
            if st.session_state.get("scraped_df", pd.DataFrame()).empty:
                st.sidebar.warning("Please scrape some data first to train the AI model.")
                return "Unknown"

            df_train = create_binary_target(st.session_state["scraped_df"])
            df_train = df_train.dropna(subset=["target_label"])

            X_train, vect = apply_feature_extraction(df_train["statement"], phase, nlp)
            y_train = df_train["target_label"]

            model = LogisticRegression(max_iter=1000, solver="liblinear")
            model.fit(X_train, y_train)

            # Save for reuse
            st.session_state["ai_vectorizer"] = vect
            st.session_state["ai_model"] = model

        # Now use stored vectorizer + model
        vect = st.session_state["ai_vectorizer"]
        model = st.session_state["ai_model"]

        # Transform input statement using SAME vectorizer
        X_feat = vect.transform(lexical_features_batch([statement], nlp))

        # Predict
        y_pred = model.predict(X_feat)
        return "True" if y_pred[0] == 1 else "False"

    except Exception as e:
        st.sidebar.warning(f"AI prediction error: {e}")
        return "Unknown"

# ---------------------------
# Humor & critique
# ---------------------------
def get_phase_critique(best_phase: str) -> str:
    critiques = {
        "Lexical & Morphological": ["Ah, the Lexical phase. Proving that sometimes, all you need is raw vocabulary and minimal effort. It's the high-school dropout that won the Nobel Prize.", "Just words, nothing fancy. This phase decided to ditch the deep thought and focus on counting. Turns out, quantity has a quality all its own.", "The Lexical approach: when in doubt, just scream the words louder. It lacks elegance but gets the job done."],
        "Syntactic": ["Syntactic features won? So grammar actually matters! We must immediately inform Congress. This phase is the meticulous editor who corrects everyone's texts.", "The grammar police have prevailed. This model focused purely on structure, proving that sentence construction is more important than meaning... wait, is that how politics works?", "It passed the grammar check! This phase is the sensible adult in the room, refusing to process any nonsense until the parts of speech align."],
        "Semantic": ["The Semantic phase won by feeling its feelings. It's highly emotional, heavily relying on vibes and tone. Surprisingly effective, just like a good political ad.", "It turns out sentiment polarity is the secret sauce! This model just needed to know if the statement felt 'good' or 'bad.' Zero complex reasoning required.", "Semantic victory! The model simply asked, 'Are they being optimistic or negative?' and apparently that was enough to crush the competition."],
        "Discourse": ["Discourse features won! This phase is the over-analyzer, counting sentences and focusing on the rhythm of the argument. It knows the debate structure better than the content.", "The long-winded champion! This model cared about how the argument was *structured*—the thesis, the body, the conclusion. It's basically the high school debate team captain.", "Discourse is the winner! It successfully mapped the argument's flow, proving that presentation beats facts."],
        "Pragmatic": ["The Pragmatic phase won by focusing on keywords like 'must' and '?'. It just needed to know the speaker's intent. It's the Sherlock Holmes of NLP.", "It's all about intent! This model ignored the noise and hunted for specific linguistic tells. It’s concise, ruthless, and apparently correct.", "Pragmatic features for the win! The model knows that if someone uses three exclamation marks, they're either lying or selling crypto. Either way, it's a clue."],
    }
    return random.choice(critiques.get(best_phase, ["The results are in, and the system is speechless. It seems we need to hire a better comedian."]))

def get_model_critique(best_model: str) -> str:
    critiques = {
        "Naive Bayes": ["Naive Bayes: It's fast, it's simple, and it assumes every feature is independent. The model is either brilliant or blissfully unaware, but hey, it works!", "The Simpleton Savant has won! Naive Bayes brings zero drama and just counts things. It’s the least complicated tool in the box, which is often the best.", "NB pulled off a victory. It’s the 'less-is-more' philosopher who manages to outperform all the complex math majors."],
        "Decision Tree": ["The Decision Tree won by asking a series of simple yes/no questions until it got tired. It's transparent, slightly judgmental, and surprisingly effective.", "The Hierarchical Champion! It built a beautiful, intricate set of if/then statements. It's the most organized person in the office, and the accuracy shows it.", "Decision Tree victory! It achieved success by splitting the data until it couldn't be split anymore. A classic strategy in science and divorce."],
        "Logistic Regression": ["Logistic Regression: The veteran politician of ML. It draws a clean, straight line to victory. Boring, reliable, and hard to beat.", "The Straight-Line Stunner. It uses simple math to predict complex reality. It's predictable, efficient, and definitely got tenure.", "LogReg prevails! The model's philosophy is: 'Probability is all you need.' It's the safest bet, and the accuracy score agrees."],
        "SVM": ["SVM: It found the biggest, widest gap between the truth and the lies, and parked its hyperplane right there. Aggressive but effective boundary enforcement.", "The Maximizing Margin Master! SVM doesn't just separate classes; it builds a fortress between them. It's the most dramatic and highly paid algorithm here.", "SVM crushed it! It’s the model that believes in extreme boundaries. No fuzzy logic, just a hard, clean, dividing line."],
    }
    return random.choice(critiques.get(best_model, ["This model broke the simulation, so we have nothing funny to say."]))

def generate_humorous_critique(df_results: pd.DataFrame, selected_phase: str) -> str:
    if df_results.empty:
        return "The system failed to train anything. We apologize; our ML models are currently on strike demanding better data and less existential dread."
    df_results = df_results.copy()
    df_results['F1-Score'] = pd.to_numeric(df_results['F1-Score'], errors='coerce').fillna(0)
    best_idx = df_results['F1-Score'].idxmax()
    best_model_row = df_results.loc[best_idx]
    best_model = best_model_row['Model']
    max_f1 = best_model_row['F1-Score']
    max_acc = best_model_row['Accuracy']
    phase_critique = get_phase_critique(selected_phase)
    model_critique = get_model_critique(best_model)
    headline = f"👑 The Golden Snitch Award goes to the {best_model}!"
    summary = (
        f"**Accuracy Report Card:** {headline}\n\n"
        f"This absolute unit achieved a **{max_acc:.2f}% Accuracy** (and {max_f1:.2f} F1-Score) on the `{selected_phase}` feature set. "
        f"It beat its rivals, proving that when faced with political statements, the winning strategy was to rely on: **{selected_phase} features!**\n\n"
    )
    roast = (
        f"### The AI Roast (Certified by a Data Scientist):\n"
        f"**Phase Performance:** {phase_critique}\n\n"
        f"**Model Personality:** {model_critique}\n\n"
        f"*(Disclaimer: All models were equally confused by the 'Mostly True' label, which they collectively deemed an existential threat.)*"
    )
    return summary + roast

# ---------------------------
# STREAMLIT APP UI
# ---------------------------
def app():
    st.set_page_config(page_title="AI vs. Fact: Intelligent Verifier", layout="wide")

    # === Custom Styling ===
    st.markdown("""
        <style>
        * {
            font-family: 'Poppins', sans-serif;
        }
        /* Global background */
        .stApp {
            background: linear-gradient(145deg, #101820 0%, #1A2634 100%);
            color: #E6E6E6;
        }
        /* Header Container */
        .hero {
            padding: 3rem 2rem;
            border-radius: 20px;
            background: linear-gradient(120deg, rgba(0,200,255,0.15), rgba(0,120,255,0.05));
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        }
        .hero h1 {
            font-size: 3rem;
            font-weight: 800;
            color: #00E5FF;
            margin-bottom: 0.3em;
        }
        .hero h3 {
            font-weight: 400;
            color: #A8B2C1;
            font-size: 1.2rem;
        }
        /* Divider Styling */
        hr {
            border: none;
            height: 1px;
            background: linear-gradient(to right, transparent, #00E5FF, transparent);
            margin: 30px 0;
        }
        /* Streamlit buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #00bcd4, #2196f3);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #26c6da, #42a5f5);
            transform: translateY(-2px);
        }
        /* Cards */
        .card {
            background: rgba(255,255,255,0.07);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.3);
        }
        /* Highlight badges */
        .highlight {
            background: #00E5FF;
            color: #000;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 700;
        }
        </style>
    """, unsafe_allow_html=True)

    # === Hero Section ===
    st.markdown("""
        <div class="hero">
            <h1>AI vs. Fact</h1>
            <h3>Unmasking misinformation through intelligent linguistic reasoning.</h3>
        </div>
    """, unsafe_allow_html=True)

    # === Layout ===
    col1, col2, col3 = st.columns([1.2, 2, 2])

    if 'scraped_df' not in st.session_state:
        st.session_state['scraped_df'] = pd.DataFrame()
    if 'df_results' not in st.session_state:
        st.session_state['df_results'] = pd.DataFrame()

    # ==== LEFT COLUMN: Data & Config ====
    with col1:
        st.markdown("### 🧠 Data Source & Configuration", unsafe_allow_html=True)
        with st.container():
            start_date = st.date_input("Start Date", pd.to_datetime('2023-01-01'))
            end_date = st.date_input("End Date", pd.to_datetime('today'))

            if st.button("Scrape Politifact Data ⛏️"):
                if start_date > end_date:
                    st.error("Start Date must be before End Date.")
                else:
                    with st.spinner("Scraping fresh data..."):
                        scraped_df = scrape_data_by_date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
                        if scraped_df.empty:
                            st.warning("No data scraped — adjust your date range.")
                        else:
                            st.session_state['scraped_df'] = scraped_df
                            st.success(f"✅ {len(scraped_df)} records scraped successfully.")
                            st.download_button("⬇️ Download CSV", scraped_df.to_csv(index=False).encode('utf-8'), "politifact_scraped.csv")

        st.divider()

        st.markdown("### ⚙️ Model Configuration")
        selected_phase = st.selectbox("Select Feature Phase:", ["Lexical & Morphological", "Syntactic", "Semantic", "Discourse", "Pragmatic"])

        if st.button("Run Model Showdown 🥊"):
            if st.session_state['scraped_df'].empty:
                st.error("Please scrape data first!")
            else:
                with st.spinner(f"Training models using {selected_phase} features..."):
                    df_results = evaluate_models(st.session_state['scraped_df'], selected_phase, NLP_MODEL)
                    st.session_state['df_results'] = df_results
                    st.session_state['selected_phase_run'] = selected_phase
                    if not df_results.empty:
                        st.success("✅ Analysis complete.")
                    else:
                        st.warning("No results — check dataset quality.")

    # ==== CENTER COLUMN: Results & Charts ====
    with col2:
        st.markdown("### 📊 Model Performance Dashboard")
        if st.session_state['df_results'].empty:
            st.info("No results yet — run model showdown first.")
        else:
            df_results = st.session_state['df_results']
            st.markdown(f"#### Phase: <span class='highlight'>{st.session_state['selected_phase_run']}</span>", unsafe_allow_html=True)
            st.dataframe(df_results[['Model','Accuracy','F1-Score','Training Time (s)','Inference Latency (ms)']], use_container_width=True)

            metric = st.selectbox("Select Metric for Comparison:", ['Accuracy','F1-Score','Precision','Recall'])
            df_plot = df_results[['Model', metric]].set_index('Model')
            st.bar_chart(df_plot)

            st.download_button("📥 Export Results CSV", df_results.to_csv(index=False).encode('utf-8'), "model_results.csv")

    # ==== RIGHT COLUMN: Critique + Scatter Plot ====
    with col3:
        st.markdown("### 🎭 AI Critique & Trade-off Visuals")
        if st.session_state['df_results'].empty:
            st.info("Awaiting model run results.")
        else:
            critique_text = generate_humorous_critique(st.session_state['df_results'], st.session_state['selected_phase_run'])
            st.markdown(critique_text, unsafe_allow_html=True)
            st.divider()

            x_axis = st.selectbox("X-Axis (Speed/Cost):", ['Training Time (s)', 'Inference Latency (ms)'])
            y_axis = st.selectbox("Y-Axis (Quality):", ['Accuracy','F1-Score','Precision','Recall'])
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(st.session_state['df_results'][x_axis], st.session_state['df_results'][y_axis], s=150, alpha=0.8)
            for i, row in st.session_state['df_results'].iterrows():
                ax.annotate(row['Model'], (row[x_axis], row[y_axis]*0.98), fontsize=9)
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title("Speed vs Quality")
            ax.grid(True, linestyle="--", alpha=0.4)
            st.pyplot(fig)

    st.divider()
    st.markdown("<center>🚀 <i>Built to decode truth — powered by NLP, data, and logic.</i></center>", unsafe_allow_html=True)

    # ================================================
    # 🔎 Google Fact Check Explorer (Integrated Tool)
    # ================================================
    st.divider()
    st.markdown("## 🌐 Google Fact Check Explorer")

    st.markdown("""
        <div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:15px;">
            Enter any <b>news headline</b> or <b>claim</b> below to verify it against
            Google's official <a href='https://toolbox.google.com/factcheck/explorer' target='_blank' style='color:#00E5FF; text-decoration:none;'>Fact Check Tools</a> database.
        </div>
    """, unsafe_allow_html=True)

    query = st.text_input("🧠 Enter a headline or claim for verification:")

    if st.button("🔍 Check Fact (Google API)"):
        if not query.strip():
            st.warning("Please enter a news headline or statement.")
        else:
            api_key = API_KEY  # already loaded from st.secrets
            url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key={api_key}"

            with st.spinner("Fetching verified results..."):
                response = requests.get(url)
                data = response.json()

            if "claims" in data:
                st.success(f"✅ Found {len(data['claims'])} related fact-checks.")
                for claim in data["claims"]:
                    st.markdown("---")
                    st.markdown(f"### 🗞️ {claim.get('text', 'No headline provided')}")
                    if "claimReview" in claim:
                        for review in claim["claimReview"]:
                            publisher = review.get("publisher", {}).get("name", "Unknown Source")
                            rating = review.get("textualRating", "No Rating")
                            link = review.get("url", "#")

                            # color code ratings
                            if "true" in rating.lower():
                                color = "#00c853"
                            elif "false" in rating.lower():
                                color = "#e53935"
                            elif "half" in rating.lower() or "mixed" in rating.lower():
                                color = "#ffb300"
                            else:
                                color = "#9e9e9e"

                            st.markdown(f"""
                                <div style="background:rgba(255,255,255,0.07); border-radius:10px; padding:10px; margin-bottom:10px;">
                                    <div><b>Source:</b> {publisher}</div>
                                    <div><b>Rating:</b> 
                                        <span style="background:{color}; color:white; border-radius:5px; padding:3px 8px; font-weight:600;">
                                            {rating}
                                        </span>
                                    </div>
                                    <div style="margin-top:6px;">
                                        <a href="{link}" target="_blank" style="color:#00E5FF; text-decoration:none;">View Full Article →</a>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No reviews found for this claim.")
            else:
                st.error("❌ No fact-check results found for this query.")

if __name__ == "__main__":
    app()
