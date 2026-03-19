import pandas as pd
from sklearn.model_selection import train_test_split


# 1. Load dataset
df = pd.read_csv("Labels.csv")
if "Unnamed: 4" in df.columns:
    df = df.drop(columns=["Unnamed: 4"])


print("Original dataset size:", len(df))


# 2. Encode labels
df["label_numeric"] = df["Label"].map({"GON+": 1, "GON-": 0})


# 3. Check class distribution
print("\nOverall label distribution:")
print(df["Label"].value_counts())
print("\nOverall label percentages:")
print(df["Label"].value_counts(normalize=True).round(3))


# 4. Verify no patient has mixed labels
patients_with_mixed_labels = df.groupby("Patient")["Label"].nunique()
mixed = patients_with_mixed_labels[patients_with_mixed_labels > 1]
print("\nPatients with mixed labels:", len(mixed))  # must be 0


# 5. Stratified patient-level split
patient_labels = df.groupby("Patient")["Label"].first().reset_index()


train_patients, test_patients = train_test_split(
    patient_labels["Patient"],
    test_size=0.2,
    random_state=42,
    stratify=patient_labels["Label"]
)


train_df = df[df["Patient"].isin(train_patients)]
test_df  = df[df["Patient"].isin(test_patients)]


# 6. Verify results
print("\nTraining images:", len(train_df))
print("Testing images:", len(test_df))
print("Training patients:", train_df["Patient"].nunique())
print("Testing patients:", test_df["Patient"].nunique())


print("\nTraining label distribution:")
print(train_df["Label"].value_counts())
print(train_df["Label"].value_counts(normalize=True).round(3))


print("\nTesting label distribution:")
print(test_df["Label"].value_counts())
print(test_df["Label"].value_counts(normalize=True).round(3))


# 7. Sanity check — confirm no patient appears in both sets
overlap = set(train_df["Patient"]) & set(test_df["Patient"])
print("\nPatient overlap between train and test:", len(overlap))  # must be 0


# 8. Save datasets
train_df.to_csv("train_dataset.csv", index=False)
test_df.to_csv("test_dataset.csv", index=False)
print("\nDatasets saved successfully.")




