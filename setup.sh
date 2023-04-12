git config --global credential.helper store

pip install -r src/requirements.txt

# A new user can either:
# - Change the s3 paths below to communicate with an external database than contains at least the initial dataset wi_dataset.csv
# - Load manually the initial dataset wi_dataset.csv into the folder data/01_raw/
#   (and the possible past approaches and submissions in data/09_past_approaches/)

# mc cp s3/apalazzolo/Deduplication/data/wi_dataset.csv data/01_raw/wi_dataset.csv  # Path s3 to be changed by the user
# mc cp -r s3/apalazzolo/Deduplication/past_submissions/ data/09_past_approaches  # Path s3 to be changed by the user

pre-commit autoupdate
pre-commit install
