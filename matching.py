from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

import pandas as pd


def load_data(cv_path, job_path):
    """
    Loads candidate CV and job description datasets.
    """

    cv_df = pd.read_csv(cv_path, sep=";")
    job_df = pd.read_csv(job_path, sep=";")

    return cv_df, job_df


def vectorize_texts(
    cv_texts,
    job_texts,
    stop_words=None,
    ngram_range=(1, 1)
):
    """
    Transforms textual data into TF-IDF vector representations.
    """

    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        ngram_range=ngram_range
    )

    all_texts = cv_texts + job_texts

    vectors = vectorizer.fit_transform(all_texts)

    cv_vectors = vectors[:len(cv_texts)]
    job_vectors = vectors[len(cv_texts):]

    return cv_vectors, job_vectors


def calculate_cosine_similarity(cv_vectors, job_vectors):
    """
    Calculates cosine similarity between candidate and job vectors.
    """

    return cosine_similarity(cv_vectors, job_vectors)


def calculate_euclidean_distance(cv_vectors, job_vectors):
    """
    Calculates Euclidean distance between candidate and job vectors.
    """

    return euclidean_distances(cv_vectors, job_vectors)


def recruiter_decision(score):
    """
    Converts similarity scores into recruiter-oriented decisions
    using predefined heuristic thresholds.
    """

    if score >= 0.70:
        return "Invite to interview"
    elif score >= 0.40:
        return "Further evaluation required"
    else:
        return "Reject"


if __name__ == "__main__":

    cv_df, job_df = load_data(
        "cv_data.csv",
        "job_data.csv"
    )

    print(cv_df.head())
    print(job_df.head())

    print("matching.py works successfully")