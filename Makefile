run-get-data:
	python3 -m src.main --process get_data

run-preprocess-data:
	python3 -m src.main --process preprocess_data

run-batch-inference:
	python3 -m src.main --process run-batch-inference

run-postprocess-and-save-data:
	python3 -m src.main --process postprocess-and-save-data