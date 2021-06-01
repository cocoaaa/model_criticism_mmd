clean:
	rm -rf model_criticism_mmd-*
install:
	poetry install
	poetry build --format sdist
	tar -xvf dist/*-`poetry version -s`.tar.gz
	cd model_criticism_mmd-*/ && pip install -e .
	rm -rf model_criticism_mmd-*
full:
	poetry install --extras full
	poetry build --format sdist
	tar -xvf dist/*-`poetry version -s`.tar.gz
	cd model_criticism_mmd-*/ && pip install -e .
	rm -rf model_criticism_mmd-*