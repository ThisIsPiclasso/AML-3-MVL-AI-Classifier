from MVL_AI_Classifier.data.builder import DatasetBuilder  # noqa: E402

builder = DatasetBuilder(size=140000)
# builder.scan()
# print(f"Length of dataset: {len(builder.data)}")
# builder.filter()
# print(f"Length of filtered dataset: {len(builder.data)}")
builder()
print(builder.data.head())
print(f"Length of dataset: {len(builder.data)}")
print("Total rows in file:", len(builder.data))
print("Value counts for splits:\n", builder.data["split"].value_counts())
