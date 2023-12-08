## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run bloom-1b7 locally

```bash
python3 run_bloom.py
```
-> Above command will start a local flask server on port 5000 with a UI to interact with bloom-1b7 model locally

## Fine tune bloom-560m model locally with pdf

```bash
python3 train_bloom.py
```
-> Above command will train the bloom-560m model with the data from pdf and create a folder named "bloom_fine_tuned_model" in the project's root directory
-> You can also change the path to pdf file and name of fine tuned model accordingly

## Run fine tuned bloom-560m model locally

```bash
python3 run_trained_bloom.py
```
-> Above command will start a local flask server on port 5000 with a UI to interact with fine tuned bloom-560m model locally







