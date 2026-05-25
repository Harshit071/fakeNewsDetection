"""
Run instructions:
1. Place True.csv and Fake.csv in data/raw/.
2. Install the dependencies listed in requirements.txt.
3. Run: python golden_response.py
4. The script will create merged_news.csv, cleaned_news.csv, the trained model,
   the fitted vectorizer, confusion matrix plots, ROC curve plots, and metrics.
5. Optional email notifications can be enabled with SMTP_* environment variables.
"""

from __future__ import annotations

from ml_core import format_stage_table, run_pipeline


def main() -> None:
    """Execute the full machine learning pipeline and print a summary table."""

    stage_results = run_pipeline()
    print(format_stage_table(stage_results))


if __name__ == "__main__":
    main()
