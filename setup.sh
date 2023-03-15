git config --global credential.helper store

mc cp s3/apalazzolo/Deduplication/data/wi_dataset.csv data/01_raw/wi_dataset.csv
# mc cp s3/projet-dedup-oja/data/wi_dataset.csv data/01_raw/wi_dataset.csv

mc cp -r s3/apalazzolo/Deduplication/past_submissions/ data/09_past_submissions
