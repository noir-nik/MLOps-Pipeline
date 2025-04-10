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
├── src/
│   ├── data/                   # Data processing
│   ├── model/                  # Model training and serving
│   └── run.py                  # Main CLI interface
├── runtime/                    # Runtime data storage
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
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
Generate monitoring report:
```bash
python src/run.py -mode "summary"
```
Output: Path to generated report file
