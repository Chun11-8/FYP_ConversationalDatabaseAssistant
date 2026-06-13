from datasets import load_dataset

dataset = load_dataset("birdsql/livesqlbench-base-lite-sqlite", "livesqlbench")
dev_data = dataset["dev"]

# See all column names
print(dev_data.column_names)

# Look at first example
print(dev_data[0])