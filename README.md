# MLOps Pipeline
## Overview


MLOps pipeline for building machine learning models on streaming data and automated deployment in a production environment.

## Key Features

- **Streaming Data Processing**: Batch collection and processing of streaming data
- **End-to-End ML Pipeline**:
  - Data collection and storage
  - Automated data quality analysis, EDA and cleaning
  - Model training and incremental updates
  - Model validation and monitoring
  - Model serving and inference
- **CLI Interface** for pipeline control
- **Custom Implementation** without specialized MLOps tools

## Project Structure

```
mlops-pipeline/
├── .github/
│   └── workflows/
│       └── mlops-workflow.yml   # CI/CD workflow definition
├── doc/                         # Documentation
├── src/
│   ├── data/                    # Data processing
│   ├── model/                   # Model training and serving
│   └── run.py                   # Main CLI interface
├── runtime/                     # Runtime data storage
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Installation

1. Clone the repository:
	```bash
	git clone https://github.com/noir-nik/MLOps-Pipeline.git
	cd mlops-pipeline
	```

2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

3. Follow [instructions](https://www.kaggle.com/docs/api#authentication) for providing Kaggle API to download the dataset

## Usage

The pipeline is controlled through a command-line interface with three main modes:

### 1. Inference Mode
Apply the trained model to new data:
```bash
python src/run.py -mode "inference" -file "./path_to_data.csv"
```
Output: Path to CSV file with predictions added in a "predict" column

### 2. Update Mode
Update the model with new data:
```bash
python src/run.py -mode "update"
```
Output: "true" if successful, "false" if failed

### 3. Summary Mode
Generate report comparing the model performance:
```bash
python src/run.py -mode "summary"
```
Output: Path to generated report file

### 4. Report Mode
Generate detailed report and visualizations:
```bash
python src/run.py -mode "report"
```

## CI/CD Pipeline

This project includes a GitHub Actions workflow for continuous integration and deployment of the ML pipeline.

**CI/CD Features**:
   - Automatic model training on push/pull requests
   - Scheduled daily incremental training
   - Preservation of model state between runs
   - Training logs and summary reports as downloadable artifacts

### Customizing the Workflow

The workflow can be customized by modifying the following parameters in `.github/workflows/iterative_taining.yml`:

- `TRAINING_ITERATIONS`: Number of training iterations per workflow run
- Scheduled training frequency (cron)