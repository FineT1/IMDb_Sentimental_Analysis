# IMDb Movie Review Sentiment Analysis

This project reads movie reviews from IMDb and decides whether each one is **positive** or **negative**. It does this with classic machine learning models, and it compares several models to see which one works best.

---

## What's in this project

```
finet1-imdb_sentimental_analysis/
├── README.md              ← you are here
├── requirements.txt       ← list of Python libraries to install
├── data/                  ← the movie review dataset goes here
└── src/                   ← all the Python scripts
    ├── load_dataset.py
    ├── preprocess.py
    ├── vectorize.py
    ├── train_member1.py
    ├── train_member2.py
    ├── evaluate_results.py
    └── eval_member2.py
```

---

## How it works (the big picture)

Think of the project as an assembly line. Each script does one job and hands its output to the next one.

1. **Split the data.** The reviews are divided into two halves: one for training the models and one for testing them.
2. **Clean the text.** Remove HTML tags, punctuation, numbers, and common filler words like "the" and "is". Convert words to their base form (for example, "movies" becomes "movie").
3. **Turn text into numbers.** Computers can't read words directly, so each review is converted into a list of numbers that describe which words and word pairs appear in it.
4. **Train the models.** Several machine learning models learn from the training half to tell positive reviews from negative ones.
5. **Test and compare.** Each model is checked on the test half, and its results are saved as tables and charts.

---

## Before you start

You need:

- **Python 3** (version 3.9 or newer is a safe choice)
- **At least 8 GB of RAM** (16 GB is better for the second training script)
- **An internet connection** for the first run, so the text-cleaning tools can download their word lists

---

## Setup (one time only)

**1. Open a terminal in the project folder.** This is the folder that contains `src/`, `data/`, and this README.

**2. (Recommended) Create a virtual environment.** This keeps the project's libraries separate from everything else on your computer.

```bash
python -m venv venv
```

Activate it:

- On Mac or Linux: `source venv/bin/activate`
- On Windows: `venv\Scripts\activate`

**3. Install the libraries.**

```bash
pip install -r requirements.txt
```

If that fails or the file is incomplete, install the main libraries directly:

```bash
pip install pandas numpy scikit-learn nltk joblib matplotlib
```

**4. Add the dataset.**

Download the **IMDB Dataset of 50K Movie Reviews** (it's on Kaggle and is usually named `IMDB Dataset.csv`). Save it inside the `data/` folder so the path is:

```
data/IMDB Dataset.csv
```

The file must have two columns named `review` and `sentiment`, and the sentiment values must be spelled exactly `positive` and `negative`.

**5. Create the output folders.**

```bash
mkdir -p data models results
```

The scripts save their work into `models/` and `results/`. Creating them now prevents errors later.

---

## Running the project

Run the scripts **one at a time, in this order**. Each one must finish before you start the next. All commands are run from the project folder.

| Step | Command | What it does | What you get |
|------|---------|--------------|--------------|
| 1 | `python src/load_dataset.py` | Splits the 50,000 reviews into a training half and a testing half (25,000 each). | `data/train.csv`, `data/test.csv` |
| 2 | `python src/preprocess.py` | Cleans the text in both halves. The first run downloads some word lists, so it needs internet. | `data/train_processed.csv`, `data/test_processed.csv` |
| 3 | `python src/vectorize.py` | Converts the cleaned text into numbers in three different ways. | Three files in `models/` ending in `_vectorizer.pkl` |
| 4 | `python src/train_member1.py` | Trains **Logistic Regression** and **Naive Bayes** on each of the three versions. This is fairly quick. | `results/member1_results.csv` and model files in `models/` |
| 5 | `python src/train_member2.py` | Trains **SVM**, **Random Forest**, **Boosting**, and a **neural network (DNN)**. **This is the slowest step**, so be patient. | `results/member2_results.csv` and model files in `models/` |
| 6 | `python src/evaluate_results.py` | Draws charts comparing the Member 1 models. | Charts in `results/` |
| 7 | `python src/eval_member2.py` | Draws charts comparing the Member 2 models. If Step 4's results exist, it also makes one combined chart with every model. | Charts and `all_models_summary.csv` in `results/` |

When everything is finished, the `results/` folder holds the performance tables and charts. Open the PNG files to see the comparisons.

### Tips for the slow step (Step 5)

Near the top of `src/train_member2.py` there are a few numbers you can lower if training takes too long:

- `K_FEATURES_BOOST` (currently 10,000)
- `K_FEATURES_RF_DNN` (currently 20,000)

Lowering them makes training faster but may slightly reduce accuracy.

### If you're on a computer without a screen (like a remote server)

The chart scripts normally pop up a window. To avoid that, add `MPLBACKEND=Agg` before the command:

```bash
MPLBACKEND=Agg python src/evaluate_results.py
MPLBACKEND=Agg python src/eval_member2.py
```

---

## Understanding the results

Each model is scored on the same test reviews using four measures:

- **Accuracy:** the share of all reviews that were labelled correctly.
- **Precision:** of the reviews the model called positive, how many really were.
- **Recall:** of the reviews that really were positive, how many the model found.
- **F1 Score:** a single number that balances precision and recall.

For a quick answer, look at **Accuracy** in the comparison charts. The higher the bar, the better the model. The results files also list the best model and the best settings for each one.

---

## Good to know

- **Run from the project folder.** The scripts look for `data/` and `models/` relative to where you run them. Running from inside `src/` will cause "file not found" errors.
- **Common words are removed.** The cleaning step removes words like "not" and "no". This can make the models miss negative phrases such as "not good". It's a good idea to try keeping those words as an improvement.
- **The split is 50/50.** Many projects use 80% for training, but this one uses half the data for testing, which is a deliberate choice.
- **Some models can't be reused on new text as-is.** The Random Forest, Boosting, and DNN models were trained on a selected subset of features, and that selection step isn't saved. The SVM, Logistic Regression, and Naive Bayes models can be reused directly.

---

## Troubleshooting

| Problem | What to do |
|---------|-----------|
| "File not found" for `IMDB Dataset.csv` | Check the file is inside `data/` with the exact name, and that you're running from the project folder. |
| "File not found" when saving files | Run `mkdir -p data models results` first. |
| Error about `stopwords` or `wordnet` | Connect to the internet and run Step 2 again, or run `python -m nltk.downloader stopwords wordnet omw-1.4`. |
| `MemoryError` or the computer slows to a crawl | Lower the `K_FEATURES` numbers in `train_member2.py`, or close other programs. |
| Errors about `NaN` or labels | Make sure the sentiment column says exactly `positive` or `negative`, in lowercase. |
| The chart script seems stuck | It's waiting for you to close a chart window. Close it, or use the `MPLBACKEND=Agg` option above. |
| Warnings about convergence | These are usually harmless and can be ignored. |
