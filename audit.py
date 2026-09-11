"""
Reproducible Data Integrity Audit
Fake News Detection Dataset

Uses:
    Pandas
    NumPy

Run:
    python audit.py
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

FILE_NAME = "compressed_data (4).csv"

TARGET = "label"

FAKE = "fake"
REAL = "real"


# ============================================================
# LOAD DATASET
# ============================================================

def load_data():
    """Load the compressed CSV dataset."""

    try:
        df = pd.read_csv(
            FILE_NAME,
            compression="gzip"
        )

        print("\nDataset loaded successfully.")
        return df

    except Exception as e:
        print("\nError while loading dataset:")
        print(e)
        return None


# ============================================================
# DATASET SHAPE
# ============================================================

def dataset_shape(df):
    """Return number of rows and columns."""

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1])
    }


# ============================================================
# COLUMN INFORMATION
# ============================================================

def column_information(df):
    """Return dataset column names."""

    return {
        "columns": list(df.columns)
    }


# ============================================================
# MISSING VALUES
# ============================================================

def missing_values(df):
    """Check missing values in every column."""

    missing = df.isnull().sum()

    return {
        str(column): int(value)
        for column, value in missing.items()
        if value > 0
    }


# ============================================================
# DUPLICATE DATA
# ============================================================

def duplicate_data(df):
    """Check duplicate rows and duplicate articles."""

    result = {
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_rows": int(df.drop_duplicates().shape[0])
    }

    if "text" in df.columns:

        result["unique_articles"] = int(
            df["text"].nunique(dropna=True)
        )

        result["duplicate_articles"] = int(
            df["text"].duplicated().sum()
        )

    if "title" in df.columns:

        result["unique_titles"] = int(
            df["title"].nunique(dropna=True)
        )

        result["duplicate_titles"] = int(
            df["title"].duplicated().sum()
        )

    return result


# ============================================================
# DATA TYPES
# ============================================================

def data_types(df):
    """Return data type of every column."""

    return {
        str(column): str(dtype)
        for column, dtype in df.dtypes.items()
    }


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

def label_distribution(df):
    """Calculate Fake and Real news distribution."""

    if TARGET not in df.columns:

        return {
            "error": "label column not found"
        }

    labels = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    fake_count = int(
        np.sum(labels == FAKE)
    )

    real_count = int(
        np.sum(labels == REAL)
    )

    total = fake_count + real_count

    if total == 0:

        return {
            "fake_count": 0,
            "real_count": 0,
            "fake_percentage": 0,
            "real_percentage": 0
        }

    fake_percentage = (
        fake_count / total * 100
    )

    real_percentage = (
        real_count / total * 100
    )

    return {
        "fake_count": fake_count,
        "real_count": real_count,
        "fake_percentage": round(
            fake_percentage, 2
        ),
        "real_percentage": round(
            real_percentage, 2
        )
    }


# ============================================================
# CLASS BALANCE
# ============================================================

def class_balance(df):
    """Check whether Fake and Real classes are balanced."""

    if TARGET not in df.columns:

        return {
            "error": "label column not found"
        }

    labels = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    fake_count = int(
        np.sum(labels == FAKE)
    )

    real_count = int(
        np.sum(labels == REAL)
    )

    if fake_count == 0:

        ratio = 0

    else:

        ratio = real_count / fake_count

    return {
        "fake_count": fake_count,
        "real_count": real_count,
        "real_to_fake_ratio": round(
            ratio, 2
        )
    }


# ============================================================
# MAJORITY CLASS BASELINE
# ============================================================

def majority_baseline(df):
    """
    Calculate the accuracy obtained by always
    predicting the most common class.
    """

    if TARGET not in df.columns:

        return {
            "error": "label column not found"
        }

    counts = df[TARGET].value_counts()

    if len(counts) == 0:

        return {
            "error": "Dataset is empty"
        }

    majority_class = str(
        counts.index[0]
    )

    accuracy = (
        counts.iloc[0] / len(df) * 100
    )

    return {
        "majority_class": majority_class,
        "majority_baseline_accuracy": round(
            float(accuracy), 2
        )
    }


# ============================================================
# TEXT STATISTICS
# ============================================================

def text_statistics(df):
    """Analyze news article text."""

    if "text" not in df.columns:

        return {
            "error": "text column not found"
        }

    text = (
        df["text"]
        .fillna("")
        .astype(str)
    )

    word_counts = (
        text
        .str.split()
        .str.len()
        .to_numpy()
    )

    character_counts = (
        text
        .str.len()
        .to_numpy()
    )

    if len(word_counts) == 0:

        return {
            "error": "No text data found"
        }

    return {
        "empty_articles": int(
            np.sum(word_counts == 0)
        ),

        "non_empty_articles": int(
            np.sum(word_counts > 0)
        ),

        "average_words_per_article": round(
            float(np.mean(word_counts)), 2
        ),

        "median_words_per_article": round(
            float(np.median(word_counts)), 2
        ),

        "standard_deviation_words": round(
            float(np.std(word_counts)), 2
        ),

        "minimum_words": int(
            np.min(word_counts)
        ),

        "maximum_words": int(
            np.max(word_counts)
        ),

        "average_characters_per_article": round(
            float(np.mean(character_counts)), 2
        )
    }


# ============================================================
# TITLE STATISTICS
# ============================================================

def title_statistics(df):
    """Analyze news article titles."""

    if "title" not in df.columns:

        return {
            "error": "title column not found"
        }

    titles = (
        df["title"]
        .fillna("")
        .astype(str)
    )

    word_counts = (
        titles
        .str.split()
        .str.len()
        .to_numpy()
    )

    return {
        "empty_titles": int(
            np.sum(word_counts == 0)
        ),

        "average_title_words": round(
            float(np.mean(word_counts)), 2
        ),

        "median_title_words": round(
            float(np.median(word_counts)), 2
        ),

        "minimum_title_words": int(
            np.min(word_counts)
        ),

        "maximum_title_words": int(
            np.max(word_counts)
        )
    }


# ============================================================
# AUTHOR ANALYSIS
# ============================================================

def author_statistics(df):
    """Analyze author information."""

    if "author" not in df.columns:

        return {
            "error": "author column not found"
        }

    authors = (
        df["author"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return {
        "unique_authors": int(
            authors[authors != ""].nunique()
        ),

        "missing_or_empty_authors": int(
            np.sum(authors == "")
        )
    }


# ============================================================
# CATEGORY ANALYSIS
# ============================================================

def category_statistics(df):
    """Analyze news categories."""

    if "category" in df.columns:

        column = "category"

    elif "subject" in df.columns:

        column = "subject"

    else:

        return {
            "error": "category or subject column not found"
        }

    counts = df[column].value_counts(
        dropna=False
    )

    return {
        str(category): int(count)
        for category, count in counts.items()
    }


# ============================================================
# DATE ANALYSIS
# ============================================================

def date_statistics(df):
    """Analyze publication dates."""

    date_columns = [
        "date",
        "publication_date",
        "published_date"
    ]

    date_column = None

    for column in date_columns:

        if column in df.columns:

            date_column = column
            break

    if date_column is None:

        return {
            "error": "date column not found"
        }

    dates = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    valid_dates = dates.dropna()

    if len(valid_dates) == 0:

        return {
            "valid_dates": 0,
            "invalid_or_missing_dates":
                int(dates.isna().sum())
        }

    return {
        "valid_dates": int(
            dates.notna().sum()
        ),

        "invalid_or_missing_dates": int(
            dates.isna().sum()
        ),

        "earliest_date": str(
            valid_dates.min().date()
        ),

        "latest_date": str(
            valid_dates.max().date()
        )
    }


# ============================================================
# LABEL QUALITY
# ============================================================

def label_quality(df):
    """Check Fake/Real label quality."""

    if TARGET not in df.columns:

        return {
            "error": "label column not found"
        }

    labels = (
        df[TARGET]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return {
        "missing_labels": int(
            df[TARGET].isna().sum()
        ),

        "fake_labels": int(
            np.sum(labels == FAKE)
        ),

        "real_labels": int(
            np.sum(labels == REAL)
        ),

        "unexpected_labels": int(
            np.sum(
                ~labels.isin(
                    [FAKE, REAL]
                )
            )
        )
    }


# ============================================================
# SAME TEXT WITH DIFFERENT LABELS
# ============================================================

def label_text_conflicts(df):
    """
    Check whether identical news articles
    have different labels.
    """

    if "text" not in df.columns:

        return {
            "checked": False,
            "conflicting_articles": 0
        }

    if TARGET not in df.columns:

        return {
            "checked": False,
            "conflicting_articles": 0
        }

    temp = df[
        ["text", TARGET]
    ].dropna()

    conflicts = (
        temp
        .groupby("text")[TARGET]
        .nunique()
    )

    return {
        "checked": True,
        "conflicting_articles": int(
            np.sum(conflicts > 1)
        )
    }


# ============================================================
# NUMERICAL SUMMARY
# ============================================================

def numerical_summary(df):
    """Generate summary statistics for numerical columns."""

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    result = {}

    for column in numeric_columns:

        values = (
            df[column]
            .dropna()
            .to_numpy()
        )

        if len(values) == 0:
            continue

        result[str(column)] = {
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "mean": round(
                float(np.mean(values)), 2
            ),
            "median": round(
                float(np.median(values)), 2
            ),
            "standard_deviation": round(
                float(np.std(values)), 2
            )
        }

    return result


# ============================================================
# RUN ALL AUDITS
# ============================================================

def run_all(df=None):
    """Run all data-integrity audits."""

    if df is None:

        df = load_data()

    if df is None:

        return {
            "error": "Dataset could not be loaded"
        }

    return {

        "dataset_shape":
            dataset_shape(df),

        "column_information":
            column_information(df),

        "data_types":
            data_types(df),

        "missing_values":
            missing_values(df),

        "duplicate_data":
            duplicate_data(df),

        "label_distribution":
            label_distribution(df),

        "class_balance":
            class_balance(df),

        "majority_baseline":
            majority_baseline(df),

        "text_statistics":
            text_statistics(df),

        "title_statistics":
            title_statistics(df),

        "author_statistics":
            author_statistics(df),

        "category_statistics":
            category_statistics(df),

        "date_statistics":
            date_statistics(df),

        "label_quality":
            label_quality(df),

        "label_text_conflicts":
            label_text_conflicts(df),

        "numerical_summary":
            numerical_summary(df)
    }


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    import pprint

    print("=" * 60)
    print("FAKE NEWS DETECTION DATASET")
    print("DATA INTEGRITY AUDIT")
    print("=" * 60)

    results = run_all()

    print("\nAUDIT RESULTS")
    print("=" * 60)

    pprint.pp(results)

    print("\n" + "=" * 60)
    print("AUDIT COMPLETED")
    print("=" * 60)