# Phase-wise NLP Analysis with Model Comparison

## 📌 Project Overview

This project is a **Streamlit-based NLP analysis tool** that allows users to **upload a CSV dataset** and automatically evaluate which machine learning model performs best for a **specific NLP task**.  

Traditionally, evaluating multiple models for text classification requires **running each model separately**, which is time-consuming. This app solves that problem by running **all supported models at once**, extracting phase-wise NLP features, and providing a **clear comparison of model performance**.

---

## 🎯 Aim

The main aim of this project is to:

1. Perform **phase-wise NLP feature extraction** on text data:
   - **Phase 1: Lexical & Morphological** → Tokenization, stopword removal, lemmatization.
   - **Phase 2: Syntactic** → Part-of-Speech (POS) tagging.
   - **Phase 3: Semantic** → Sentiment polarity & subjectivity.
   - **Phase 4: Discourse** → Sentence-level features such as length and connectors.
   - **Phase 5: Pragmatic** → Contextual cues (modality words, questions, exclamations).

2. Evaluate **multiple machine learning models** on the extracted features **in one go**:
   - Naive Bayes
   - Decision Tree
   - Logistic Regression
   - Support Vector Machine (SVM)

3. Show **phase-wise model accuracy in percentages**, rounded, with **descending order ranking**.  
4. Provide **visualizations** (bar charts) for easy comparison.  

This allows users to quickly identify **which model fits their dataset best** for a specific NLP task without multiple repetitive runs.

---

## 🛠️ Technologies & Libraries Used

- **Python 3.10+**  
- **Streamlit** – for interactive web UI  
- **Pandas & NumPy** – for data manipulation  
- **NLTK** – for NLP tasks (tokenization, stopwords, lemmatization)  
- **spaCy** – for syntactic parsing (POS tagging)  
- **TextBlob** – for semantic analysis (sentiment polarity & subjectivity)  
- **scikit-learn** – for machine learning models (Naive Bayes, Decision Tree, Logistic Regression, SVM)  
- **Matplotlib** – for visualization of model performance  

---

## ⚙️ Features

1. **Upload CSV** – Users can select the text column and target column for analysis.  
2. **Phase Selection** – Choose which NLP phase to analyze.  
3. **Automatic Multi-Model Evaluation** – All supported ML models are run automatically.  
4. **Descending Accuracy Table** – Accuracy displayed as **rounded percentages with “%” symbol**, sorted from best to worst.  
5. **Visualization** – Bar chart shows accuracy of all models for the selected phase with numeric labels.  
6. **Single-Click Analysis** – No need to run models separately; get results for all models in **one go**.  

---

## 📂 How to Use

1. Clone the repository or upload your files to Streamlit Cloud.  
2. Ensure dependencies are installed:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
````

3. Run the Streamlit app:

   ```bash
   streamlit run app.py --server.fileWatcherType none
   ```
4. Upload your CSV file.
5. Select **Text Column** and **Target Column**.
6. Choose the **NLP Phase** you want to analyze.
7. Click **Run Comparison** to see **all models evaluated at once**.
8. View the **accuracy table and bar chart** to identify the best-performing model.

---

## 📊 Output Example

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 85%      |
| Naive Bayes         | 82%      |
| SVM                 | 80%      |
| Decision Tree       | 78%      |

* The table is sorted in **descending order**, making it easy to see the **best model**.
* The bar chart visually reinforces the ranking.

---

## ✅ Advantages

* **Time-saving:** No need to run each model separately.
* **Easy-to-use UI:** Works directly in the browser via Streamlit.
* **Comprehensive:** Supports multiple NLP phases and models.
* **Data-driven decision:** Helps choose the **best-fit model** for your dataset quickly.

---

## 📌 Notes

* Ensure your CSV has a **text column** and a **target label column**.
* NLTK resources (`punkt`, `stopwords`, `wordnet`, `omw-1.4`) are required. These are downloaded automatically in the app.
* Works with both **binary and multi-class classification tasks**.

---

## 📂 Project Structure

```
project/
│
├─ app.py               # Streamlit main app
├─ requirements.txt     # Python dependencies
└─ README.md            # Project documentation
```

---

