dev_dependencies:
	pip install --upgrade pip
	pip install -r requirements-dev.txt
dependencies:
	pip install --upgrade pip
	pip install -r requirements.txt
lint:
	flake8