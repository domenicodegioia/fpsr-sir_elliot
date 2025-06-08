import pandas as pd

root = "data/ml-1m/"

train_df = pd.read_csv(root + "train_new.tsv", sep="\t", names=["user_id", "item_id"], header=None)
valid_df = pd.read_csv(root + "valid_new.tsv", sep="\t", names=["user_id", "item_id"], header=None)
test_df = pd.read_csv(root + "test_new.tsv", sep="\t", names=["user_id", "item_id"], header=None)

combined_df = pd.concat([train_df, valid_df, test_df], ignore_index=True).drop_duplicates()

combined_df = combined_df.sort_values(by=["user_id", "item_id"]).reset_index(drop=True)

combined_df.to_csv(root + "dataset.tsv", sep="\t", index=False, header=False)
