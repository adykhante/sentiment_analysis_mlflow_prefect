# Sentiment Analysis Project with MLflow

🚀 Excited to share my latest project leveraging MLflow for experiment tracking, model management, and reproducibility in sentiment analysis! 🌟

🔍 Objective: Utilize MLflow to streamline experiment tracking and model management in sentiment analysis.

🔧 Project Highlights:

- Integrated MLflow for seamless experiment tracking and model management.
- Demonstrated logging of parameters, metrics, and artifacts for enhanced model transparency.
- Customized MLflow UI with run names for improved organization and clarity.
- Presented metric and hyperparameter plots for insightful analysis.
- Leveraged MLflow's tagging capabilities for efficient model versioning and management.

🔧 Bonus Achievement:

- Built a Prefect Workflow and demonstrated auto-scheduling. The Prefect Dashboard showcases relevant outputs for effective monitoring.

📊 Model Versioning:

- Archived: Retain previous model versions for reference.
- Staged: Prepare models for deployment pending final approval.
- Production: Deploy approved models for serving predictions.

## Setup Instructions

### Prerequisites

- Python 3 installed on your system.

### Creating and Activating a Virtual Environment

In order to install Prefect, create a virtual environment:
- `python -m venv venv_name`

###Installing Prefect 2.0
- `pip install prefect`
- `pip install -U prefect` (OR if you have Prefect 1, upgrade to Prefect 2 using this command)

###Running Prefect Dashboard
- `prefect server start`

###Running Python App for Prefect
(After Running prefect server, Navigate To Virtual Environment, and run python app.py)
- `python my_workflow_script.py`
