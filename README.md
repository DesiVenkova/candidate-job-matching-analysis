# Candidate – Job Matching using Similarity Metrics

## Project Overview

This project explores how mathematical similarity measures can support automated recruitment screening and candidate-job matching.

The project compares:
- Cosine Similarity
- Euclidean Distance

using TF-IDF vectorization on synthetic CV and job description datasets.

The primary goal is to evaluate whether text similarity analysis can effectively identify relevant and non-relevant candidates for technical job roles.

---

## Dataset

The project uses two independent datasets:

- Synthetic candidate CV profiles
- Simulated real-world job descriptions

The datasets include multiple software engineering and technical domains such as:

- .NET Development
- Python Development
- Frontend Development
- QA Automation
- DevOps
- Cybersecurity
- Data Science
- Project Management

---

## Technologies

The project was developed using:

- Python
- pandas
- scikit-learn
- matplotlib
- Jupyter Notebook

---

## Methodology

The project follows the following workflow:

1. Load and preprocess textual data
2. Clean and normalize candidate and job description text
3. Transform text into TF-IDF vectors
4. Calculate cosine similarity and Euclidean distance
5. Compare similarity approaches
6. Visualize candidate-job matching results
7. Generate recruiter-oriented recommendations

---

## Key Features

- TF-IDF vectorization
- Candidate-job similarity analysis
- Cosine similarity comparison
- Euclidean distance analysis
- Recruiter decision support
- Data visualization and statistical analysis

---

## Project Structure

```text
candidate_job_matching_analysis.ipynb
matching.py
cv_data.csv
job_data.csv
README.md
```

---

## How to Run

1. Install required libraries:

```bash
pip install pandas scikit-learn matplotlib
```

2. Open the Jupyter Notebook:

```bash
jupyter notebook
```

3. Run all notebook cells sequentially.

---

## Results

The analysis demonstrates that cosine similarity performs more effectively for text-based recruitment matching tasks compared to Euclidean distance.

The model successfully identifies strong candidate-job matches while distinguishing unrelated candidate profiles.

---

## Limitations

- The model relies primarily on keyword overlap
- Semantic meaning and contextual understanding are limited
- The dataset is synthetic and relatively small
- Soft skills and non-technical experience are not considered

---

## Conclusion

This project demonstrates how text analysis and mathematical similarity measures can support recruiter-oriented candidate screening and ranking processes.
