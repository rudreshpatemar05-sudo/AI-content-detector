# AI Content Detector

A Python-based AI content detector that uses TF-IDF features, stylometric signals, and logistic regression to estimate whether text is likely human-written or AI-generated.

This project includes data preprocessing, model training, evaluation, and a Streamlit web interface for interactive text analysis.

## Features

- TF-IDF word and character n-gram features
- Stylometric feature extraction
- Logistic regression classifier
- Calibration and probability thresholding
- Evaluation reports and confusion matrices
- Streamlit app for live analysis
- Support for document upload (TXT, PDF, image OCR)

## Project Structure

```text
ai-content-detector/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── results/
├── src/
│   ├── baseline.py
│   ├── create_evasion_test.py
│   ├── document_utils.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── prepare_data.py
│   ├── prepare_raid_sample.py
│   ├── style_features.py
│   ├── train.py
│   ├── train_raid_robust.py
│   ├── train_robust.py
│   ├── train_tfidf.py
│   └── utils.py
├── tests/
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
└── tmp_test_examples.json
```

## Tech Stack

- Python 3.11+
- scikit-learn
- pandas
- NumPy
- Streamlit
- PyMuPDF
- PyTesseract
- RapidOCR
- PyArrow
- FSSpec

## Setup

1. Clone the repository
2. Create a virtual environment
3. Install dependencies

    ```bash
    cd ai-content-detector
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Run the app

```bash
streamlit run app/streamlit_app.py
```

## Train and evaluate model

From the project root:

```bash
python src/prepare_data.py
python src/create_evasion_test.py
python src/train_robust.py
python src/evaluate.py
```

## Notes

- The project expects some dataset files under `data/raw/` and `data/processed/`.
- Model outputs and evaluation artifacts are stored in `models/` and `results/`.
- This is a research/demo project and should be used as an assistive detection tool rather than sole evidence in high-stakes decisions.

## License

This project is provided for educational and research purposes.

## Author

Rudresh Patemar
