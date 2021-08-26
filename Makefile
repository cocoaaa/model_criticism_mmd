clean:
	rm -rf model_criticism_mmd-*
build:
	rm -rf model_criticism_mmd-latest
	poetry build --format sdist
	tar -xvf dist/*-`poetry version -s`.tar.gz
	mv model_criticism_mmd-* model_criticism_mmd-latest
install:
	cd model_criticism_mmd-latest/ && pip install -e .
test:
	pytest --workers 4 tests/
	pytest --nbmake --ignore samples/study-parameter-stability.ipynb samples/