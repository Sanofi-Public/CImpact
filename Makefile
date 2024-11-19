config:
	echo "Nothing to configure"

install:
	if [ ! -d .env ]; then \
		python -m venv .env;\
	fi
	. .env/bin/activate; pip install --upgrade pip
	. .env/bin/activate; pip install -r requirements.txt
	

format:
	black *.py
	black ./src/*.py
	

clean:
	rm -rf .env

test:
	echo "Testing";
	. .env/bin/activate; python -m pip install --upgrade --force-reinstall dist/cimpact*.whl
	. .env/bin/activate; python -m unittest -v

build:
	. .env/bin/activate; python -m build	

publish:
	git push