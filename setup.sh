git config --global credential.helper store

pip install -r src/requirements.txt

mc cp s3/apalazzolo/Deduplication/data/wi_dataset.csv data/01_raw/wi_dataset.csv  # Path s3 to be changed by the user

mc cp -r s3/apalazzolo/Deduplication/past_submissions/ data/09_past_approaches  # Path s3 to be changed by the user

pre-commit autoupdate
pre-commit install
