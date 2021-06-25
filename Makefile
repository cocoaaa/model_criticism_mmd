clean:
	rm -rf model_criticism_mmd-*
install:
	rm -rf model_criticism_mmd-*
	poetry install
	poetry build --format sdist
	tar -xvf dist/*-`poetry version -s`.tar.gz
	cd model_criticism_mmd-*/ && pip install -e .
full:
	rm -rf model_criticism_mmd-*
	poetry install --extras full
	poetry build --format sdist
	tar -xvf dist/*-`poetry version -s`.tar.gz
	cd model_criticism_mmd-*/ && pip install -e .
test:
	pytest --workers 4 tests/
	pytest --nbmake --ignore samples/study-parameter-stability.ipynb samples/