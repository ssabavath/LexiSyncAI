
.PHONY: install run test evaluate

install:
	python -m pip install -r requirements.txt

run:
	python -m streamlit run app.py

test:
	python -m pytest -q

evaluate:
	python evaluate.py
