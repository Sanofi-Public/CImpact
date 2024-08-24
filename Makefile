config:
	python ./scripts/install.py

install:
	if [ ! -d .env ]; then \
		python -m venv .env;\
	fi
	. .env/bin/activate; pip install --upgrade pip
	. .env/bin/activate; pip install -r requirements.txt
	. .env/bin/activate; python ./scripts/install.py

format:
	black *.py
	black ./src/*.py
	black .scripts/*.py

clean:
	rm -rf .env

test:
	echo "Not testing"

publish:
	git push