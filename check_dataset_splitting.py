import pandas as pd


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, help='Name of the dataset')
# args = parser.parse_args()
# dataset = args.dataset if args.dataset else 'douban'


train = pd.read_csv(f'train.tsv', sep="\t", header=None)
val = pd.read_csv(f'valid.tsv', sep="\t", header=None)
test = pd.read_csv(f'test.tsv', sep="\t", header=None)
df = pd.concat([train, val, test], axis=0)

if (val.equals(test)):
    print("val e test uguali")
else:
    print("val e test diversi")



# print(f"SHAPES for {dataset}")
print(f"train:\t{train.shape}, {train.shape[0] / df.shape[0] * 100} %")
print(f"val:\t{val.shape}, {val.shape[0] / df.shape[0] * 100} %")
print(f"test:\t{test.shape}, {test.shape[0] / df.shape[0] * 100} %")
print(f"total:\t{df.shape}, {(train.shape[0] + val.shape[0] + test.shape[0]) / df.shape[0] * 100} %")

print(f"\ntrain and test only:")
df2 = pd.concat([train, test], axis=0)
print(f"train: {train.shape[0] / (train.shape[0] + test.shape[0]) * 100} %")
print(f"test: {test.shape[0] / (train.shape[0] + test.shape[0]) * 100} %")


# verificare nello split che in ogni set ci siano tutti gli user unique
print("\nCHECK USER UNIQUE NEI SET")
print(f"unique users in train:\t{train[0].unique().__len__()}")
print(f"unique users in val:\t{val[0].unique().__len__()}")
print(f"unique users in test:\t{test[0].unique().__len__()}")
print(f"total users:\t{df[0].unique().__len__()}")
train_users = set(train[0].unique())
val_users = set(val[0].unique())
test_users = set(test[0].unique())
total_users = set(df[0].unique())
train_check = (train_users == total_users) #train_users.issubset(total_users)
val_check = val_users.issubset(train_users)
test_check = test_users.issubset(train_users)
print(f"Tutti gli user del training set sono nel dataset totale: {train_check}")
print(f"Tutti gli user del validation set sono nel training set: {val_check}")
print(f"Tutti gli user del test set sono nel training set: {test_check}")


# verificare che tutti gli item in val (e test) siano presenti anche nel train (quindi conosciuti)
print("\nCHECK ITEM UNIQUE NEI SET")
print(f"unique items in train:\t{train[1].unique().__len__()}")
print(f"unique items in val:\t{val[1].unique().__len__()}")
print(f"unique items in test:\t{test[1].unique().__len__()}")
print(f"total items:\t{df[1].unique().__len__()}")
train_items = set(train[1].unique())
val_items = set(val[1].unique())
test_items = set(test[1].unique())
total_items = set(df[1].unique())
train_check = (train_items == total_items) #train_items.issubset(total_items)
val_check = val_items.issubset(train_items)  # difference
test_check = test_items.issubset(train_items)
print(f"Tutti gli item del training set sono nel dataset totale: {train_check}")
print(f"Tutti gli item del validation set sono nel training set: {val_check}")
print(f"Tutti gli item del test set sono nel training set: {test_check}")
