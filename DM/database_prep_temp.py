from sklearn.datasets import fetch_openml
ds = fetch_openml("covertype", as_frame=True)
df = ds.frame

print("Columns:", df.columns.tolist())
print(df.head())
print("Label unique:", df.iloc[:, -1].unique())
print(df.iloc[:, -1].value_counts())

ds = fetch_openml(1169, as_frame=True)   # Airlines
df = ds.frame

print("Columns:", df.columns.tolist())
print(df.head())
print("Label unique:", df.iloc[:, -1].unique())
print(df.iloc[:, -1].value_counts())

ds = fetch_openml("higgs", as_frame=True)
df = ds.frame

print("Columns:", df.columns.tolist())
print(df.head())
print("Label unique:", df.iloc[:, -1].unique())
print(df.iloc[:, -1].value_counts())
