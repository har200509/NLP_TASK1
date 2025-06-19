# 📰 BBC News Document Classification

This notebook demonstrates a process for classifying BBC News articles into different categories — **sport**, **business**, **politics**, **tech**, and **entertainment** — using a **Support Vector Machine (SVM)** model. The process includes data loading, preprocessing, feature extraction using TF-IDF, and model training and evaluation.

---

## ⚙️ How to Use

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/har200509/NLP_TASK1.git
   ```

2. **Open in Google Colab:**

   * Upload the cloned notebook (`.ipynb`) to your Google Drive.
   * Open the notebook using [Google Colab](https://colab.research.google.com/).

3. **Run the Notebook:**

   * Run all cells sequentially.
   * Ensure your `kaggle.json` API key is uploaded to Colab for downloading the dataset.

---

## 🧪 Notebook Steps

### 1. 📥 Setup and Data Download

* Sets up Kaggle API credentials.
* Downloads the **BBC Full Text Document Classification** dataset.
* Unzips the dataset.

### 2. 📊 Data Loading and Exploration

* Loads the `bbc_data.csv` into a DataFrame using pandas.
* Displays sample rows and checks class distribution.
* Verifies data types and null entries.

### 3. 🧹 Data Preprocessing and Vocabulary Creation

* Splits the data into separate category DataFrames.
* Converts each document column into lists.
* Uses **NLTK** for:

  * Tokenization (`punkt`)
  * Stopword removal
  * Lemmatization (via `wordnet`)
* Removes punctuation.
* Builds vocabulary sets for each category by removing stopwords and punctuation.

### 4. ✂️ Vocabulary Refinement

* Finds common words across all category vocabularies using `set.intersection`.
* Removes these shared terms to keep only discriminative words.
* Combines all refined vocabularies into one set.
* Displays the final vocabulary length.

---

### 5. 📌 TF-IDF Vectorization

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert raw text data into a format suitable for ML models.

#### 🔄 Process Overview:

* The `'data'` column is converted into a list of documents.
* `TfidfVectorizer` (from `scikit-learn`) is used with:

  * English stopwords removed
  * Tokenization
* The result is a **sparse TF-IDF matrix**.

#### 🧠 What is TF-IDF?

TF-IDF scores indicate how **important a word is to a specific document**, relative to the entire dataset.

* **TF (Term Frequency):** Frequency of the word in a document.
* **IDF (Inverse Document Frequency):** Penalizes common words across all docs.

> **TF-IDF Score = TF × IDF**

A high score means the word is **frequent in this document**, but **rare overall** — useful for identifying the document's topic.

#### 🧬 Sparse Matrix Output Example:

```
<1x32548 sparse matrix of type '<class 'numpy.float64'>'
    with 49 stored elements in Compressed Sparse Row format>

  Coords      Values
  (0, 1470)   0.0679
  (0, 3534)   0.0945
  (0, 3880)   0.0735
  (0, 6155)   0.1679
  (0, 10840)  0.1812
  (0, 19775)  0.5515
  ...
```

* `32548`: Vocabulary size.
* Only `49` words had non-zero scores for this document.
* Example: word at index `19775` has a score of `0.5515`.

---

### 6. 🔢 Label Encoding

* Uses `LabelEncoder` to convert string labels (like `"business"`) into integers.
* Saves the label mapping for decoding predictions later.

### 7. 📂 Data Splitting

* Uses `train_test_split` with:

  * `test_size=0.3` (30% test set)
  * `stratify=y` to preserve class proportions
  * `random_state=42` for reproducibility

### 8. 🤖 Model Training

* Trains a **Linear Support Vector Classifier (SVC)**:

  * Uses `kernel='linear'` (effective for text classification)
  * Fits the model using `X_train` and `y_train`

### 9. 📈 Model Evaluation

* Calculates **accuracy** on the test set using `accuracy_score`.
* Builds a **confusion matrix** to evaluate per-class performance.
* Uses **seaborn heatmap** for visual display of classification results.

---

## 📚 Libraries Used

| Library          | Purpose                                                                                                                                              |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **pandas**       | Data manipulation and DataFrame operations                                                                                                           |
| **nltk**         | Text cleaning (tokenizing, stopword removal, lemmatization)                                                                                          |
| **scikit-learn** | Machine learning tasks:<br> • `TfidfVectorizer`<br> • `LabelEncoder`<br> • `SVC`<br> • `train_test_split`<br> • `accuracy_score`, `confusion_matrix` |
| **seaborn**      | For plotting heatmaps of confusion matrix                                                                                                            |

---
