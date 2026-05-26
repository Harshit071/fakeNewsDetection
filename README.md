# Fake News Detection Benchmark

## Project Overview

This repository contains a benchmark prompt, final verdict justification, and a complete runnable golden response for an end-to-end fake news detection system.

The executable implementation is in `golden_response/`. It trains a TF-IDF plus Logistic Regression classifier, evaluates validation and test performance, saves model/report artifacts, serves predictions with FastAPI, and provides a Streamlit frontend for single and batch predictions.

## Repository Structure

- `prompt.md`: detailed benchmark prompt and acceptance criteria.
- `justification.md`: final verdict comparing Response A and Response B.
- `golden_response/`: complete executable reference implementation.
- `golden_response/run_pipeline.py`: one-command training and evaluation pipeline.
- `golden_response/data_preprocessing.py`: shared preprocessing for training and inference.
- `golden_response/ml_core.py`: ingestion, training, metrics, artifacts, and orchestration logic.
- `golden_response/api.py`: FastAPI backend.
- `golden_response/streamlit_app.py`: Streamlit frontend.
- `golden_response/schemas.py`: Pydantic v2 API schemas.
- `golden_response/requirements.txt`: pinned dependencies.
- `golden_response/data/raw/`: included raw dataset files used by the pipeline.

## Running And Testing

From the repository root:

```bash
cd golden_response
pip install -r requirements.txt
python run_pipeline.py
```

After the pipeline finishes, run the API:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

In another terminal, run the frontend:

```bash
streamlit run streamlit_app.py
```

Quick syntax test:

```bash
python -m py_compile run_pipeline.py api.py streamlit_app.py ml_core.py data_preprocessing.py schemas.py
```

## Evaluation Methodology

The benchmark evaluates whether a response satisfies the explicit prompt requirements and the implicit expectations of production-quality software. The main criteria are requirement coverage, correctness, robustness, maintainability, error handling, reproducibility, and usability.

The final verdict in `justification.md` favors the response that delivers the most complete runnable system with the fewest blocking omissions.
