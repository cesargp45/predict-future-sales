#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = predict-future-sales
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	conda env update --name $(PROJECT_NAME) --file environment.yml --prune
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	conda env create --name $(PROJECT_NAME) -f environment.yml
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset (genera data/interim/sales_clean.csv a partir de data/raw)
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m predict_future_sales.dataset


## Ejecuta el notebook de EDA de punta a punta y guarda los resultados en el mismo archivo
.PHONY: eda
eda:
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/01_exploratory_data_analysis.ipynb


## Proyecto #2: ejecuta los notebooks 02 a 05 en orden (pipelines .pkl + submission.csv)
.PHONY: pipeline
pipeline:
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=10800 notebooks/02_feature_exploration.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=10800 notebooks/03_feature_creation.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=10800 notebooks/04_model_training_n_tunning.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=10800 notebooks/05_inference_calculation.ipynb


## Re-entrena el pipeline completo (preprocesamiento + modelo ganador) con los meses 0-33
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m predict_future_sales.modeling.train


## Genera data/processed/submission.csv con models/full_pipeline.pkl
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m predict_future_sales.modeling.predict


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
