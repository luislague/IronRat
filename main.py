# Importing standard libraries
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from collections import Counter
import seaborn as sns


# Importing dataset and tokenization tools
from transformers import (
    AutoTokenizer,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    SentenceTransformer,
    pipeline,
)

# Importing scikit-learn tools
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Importing additional libraries
from torch.utils.data import Dataset

# Importing NLP tools
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")


# CLASSIFICATION - SENTIMENT ANALYSIS

# Load the dataset using pandas
file_path = "/notebooks/1429_1.csv"

# Read the CSV file into a pandas DataFrame
reviews = pd.read_csv(file_path)

# View the first few rows of the dataset
print(reviews.head())

# Get a summary of the dataset
print(reviews.info())

# Check for missing values in each column
print(reviews.isnull().sum())

# Get the count of each rating value in the 'reviews.rating'
# column to see what is the proportion of the reviews rating
rating_counts = reviews["reviews.rating"].value_counts()

# Print the result
print(rating_counts)

# Combine 'reviews.title' and 'reviews.text' into a single text field
reviews["title_text"] = reviews["reviews.title"] + " " + reviews["reviews.text"]

# Clean the data: replace NaN values with empty strings and ensure
# all text is string type
reviews["title_text"] = reviews["title_text"].fillna("").astype(str)

# Check the combined text
print(reviews["title_text"].head())

# Check the data
print(reviews.head())


# Assign labels based on ratings
def label_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"


# Apply function to dataset
reviews["reviews.rating"] = reviews["reviews.rating"].apply(label_sentiment)

# Check the distribution of these labels
print(reviews["reviews.rating"].value_counts())

# check if GPU is available
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# initialize the pipeline
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device,
    use_fast=True,
)

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")

# Define candidate labels
candidate_labels = ["positive", "neutral", "negative"]
print(f"Candidate Labels: {candidate_labels}")

# Prepare lists
reviews_tile_text_list = reviews["title_text"].tolist()
reviews_ratings_list = reviews["reviews.rating"].tolist()

# Apply the classification function to the dataset
batch_size = 16
classified_labels = []

for i in range(0, len(reviews_tile_text_list), batch_size):
    batch = reviews_tile_text_list[i : i + batch_size]
    if not batch:
        continue
    try:
        results = classifier(batch, candidate_labels)
        for result in results:
            classified_labels.append(result["labels"][0])
    except Exception as e:
        print(f"An error occurred at batch index {i}: {e}")
        for _ in batch:
            classified_labels.append("error")

# Count how many labels are 'error'
error_count = classified_labels.count("error")

# Total number of classified reviews
total_reviews = len(classified_labels)

# Calculate the percentage of errors
error_percentage = (error_count / total_reviews) * 100

# Print the results
print(f"Total reviews classified: {total_reviews}")
print(f"Number of 'error' labels: {error_count}")
print(f"Percentage of reviews classified as 'error': {error_percentage:.2f}%")

# Count the occurrences of each label
positive_count = classified_labels.count("positive")
neutral_count = classified_labels.count("neutral")
negative_count = classified_labels.count("negative")
error_count = classified_labels.count("error")

# Calculate the percentages for each label
positive_percentage = (positive_count / total_reviews) * 100
neutral_percentage = (neutral_count / total_reviews) * 100
negative_percentage = (negative_count / total_reviews) * 100
error_percentage = (error_count / total_reviews) * 100

print(f"Number of 'positive' labels: {positive_count} ({positive_percentage:.2f}%)")
print(f"Number of 'neutral' labels: {neutral_count} ({neutral_percentage:.2f}%)")
print(f"Number of 'negative' labels: {negative_count} ({negative_percentage:.2f}%)")
print(f"Number of 'error' labels: {error_count} ({error_percentage:.2f}%)")

# Filter out any "error" labels if they exist in the predictions
valid_idx = [
    i for i in range(len(classified_labels)) if classified_labels[i] != "error"
]

# Filtered predicted and true labels
y_pred = [classified_labels[i] for i in valid_idx]
y_true = [reviews_ratings_list[i] for i in valid_idx]

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculate precision
precision = precision_score(y_true, y_pred, average="weighted")

# Calculate recall
recall = recall_score(y_true, y_pred, average="weighted")

# Calculate F1 score
f1 = f1_score(y_true, y_pred, average="weighted")

# Print out the results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Compute confusion matrix
cm = confusion_matrix(
    reviews_ratings_list, classified_labels, labels=["positive", "neutral", "negative"]
)

# Visualize confusion matrix
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["positive", "neutral", "negative"],
    yticklabels=["positive", "neutral", "negative"],
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Save the cleaned data
reviews_with_sentiment = pd.DataFrame(
    {
        "review": reviews_tile_text_list,
        "classified_label": classified_labels,
        "reviews_ratings": reviews_ratings_list,
    }
)

# Save to a CSV file for further analysis or sharing
reviews_with_sentiment.to_csv("classified_reviews_clean.csv", index=False)


# CLUSTERING BY PRODUCT CATEGORY

# Load the dataset using pandas
file_path = "/notebooks/classified_reviews_clean.csv"

reviews = pd.read_csv(file_path)

# Convert all values in reviews to string to avoid error during embedding
reviews["review"] = reviews["review"].astype(str)

print(reviews.head())

# Load the pre-trained model. This one maps sentences & paragraphs to a
# 384 dimensional dense vector space and can be used
# for tasks like clustering or semantic search.
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(reviews["review"], show_progress_bar=True)

print(type(embeddings))
print(embeddings.shape)
print(embeddings[0])

# Save embeddings
np.save("review_embeddings.npy", embeddings)

# Dimensionality reduction with PCA

# Dtep by step selecting number of components
pca_full = PCA().fit(embeddings)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("PCA Explained Variance")
plt.grid(True)
plt.show()

# We choose 125 so we preserve around 90% of the variance
pca = PCA(n_components=125, random_state=42)
reduced_embeddings = pca.fit_transform(embeddings)

# K-means clustering
# First we use the elbow method to determine the ideal number of clusters
inertias = []
k_values = range(2, 10)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(reduced_embeddings)
    inertias.append(kmeans.inertia_)

# Plot the results
plt.plot(k_values, inertias, "bo-")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method For Optimal k")
plt.show()

# We proceed with the cluster assgingment
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(reduced_embeddings)
reviews["cluster"] = clusters

# Analize cluster contents
for i in range(optimal_k):
    cluster_reviews = reviews[reviews["cluster"] == i]
    print(f"\nCluster {i} reviews:")
    print(cluster_reviews["review"])

# Identify domintant product categories
for i in range(optimal_k):
    cluster_categories = reviews[reviews["cluster"] == i]["review"]
    category_counts = cluster_categories.value_counts()
    print(f"\nCluster {i} category distribution:")
    print(category_counts)

# Assuming 'reviews_df' is your DataFrame with 'review_text'
# and 'cluster' columns
for cluster_id in range(optimal_k):
    print(f"\nCluster {cluster_id} Sample Reviews:")
    cluster_reviews = reviews[reviews["cluster"] == cluster_id]["review"].sample(
        5, random_state=42
    )
    for idx, review in enumerate(cluster_reviews):
        print(f"{idx+1}. {review}\n")

# Define stop words and punctuation
stop_words = set(stopwords.words("english"))
punctuation = set(string.punctuation)


# Define a Function to Clean and Tokenize Text:
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


# Compute Term Frequencies for Each Cluster:
for cluster_id in range(optimal_k):
    cluster_reviews = reviews[reviews["cluster"] == cluster_id]["review"]
    all_words = []

    for review in cluster_reviews:
        tokens = preprocess_text(review)
        all_words.extend(tokens)

    word_counts = Counter(all_words)
    common_words = word_counts.most_common(10)
    print(f"\nCluster {cluster_id} Common Words:")
    for word, count in common_words:
        print(f"{word}: {count}")

# assing labels to clusters
cluster_labels = {0: "Tablet", 1: "Echo", 2: "Kindle", 3: "Customer love", 4: "SmartTV"}

# apply lables to dataframe
reviews["cluster_label"] = reviews["cluster"].map(cluster_labels)

print(reviews["cluster_label"])
print(reviews.head())

# Save to a CSV file for further analysis or sharing
reviews.to_csv("cleaned_classified_clustered_reviews.csv", index=False)

# Assuming 'reduced_embeddings' and 'reviews_df' with 'cluster' column
silhouette_avg = silhouette_score(reduced_embeddings, reviews["cluster_label"])
db_score = davies_bouldin_score(reduced_embeddings, reviews["cluster_label"])

print(f"Silhouette Coefficient: {silhouette_avg:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}")


# SUMMARIZER - WRITE REVIEWS BASED ON PRODUCT CATEGORY AND SENTIMENT

# Load the dataset
reviews = pd.read_csv("cleaned_classified_clustered_reviews.csv.csv")

# Display the first few rows
print(reviews.head())

# Check for missing values
print(reviews.isnull().sum())

# Delete the "Generic love" reviews
category_to_remove = "Generic Love"

reviews = reviews[reviews["product_category"] != category_to_remove]

# Check if it was properly erased
category_count = reviews[reviews["product_category"] == category_to_remove].shape[0]
print(f"Occurrences of '{category_to_remove}' after filtering: {category_count}")


# Create a new column that combines the necessary fields into a single string
reviews["concatenated_columns"] = reviews.apply(
    lambda row: f"Product Category: {row['product_category']}. Customer Review: {row['review']}. Review Sentiment: {row['sentiment']}.",
    axis=1,
)

print(reviews["concatenated_columns"][1])

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Add padding token to tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Prepare the data for GPT-2
def preprocess_function(examples):
    inputs = examples["concatenated_columns"]
    inputs = [tokenizer.bos_token + text + tokenizer.eos_token for text in inputs]
    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, padding="max_length"
    )
    return model_inputs


# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(reviews[["concatenated_columns"]])
train_testvalid = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_testvalid["train"]
val_dataset = train_testvalid["test"]

# Tokenize datasets
train_dataset = train_dataset.map(
    preprocess_function, batched=True, remove_columns=["concatenated_columns"]
)
val_dataset = val_dataset.map(
    preprocess_function, batched=True, remove_columns=["concatenated_columns"]
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results_gpt2",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    logging_dir="./logs_gpt2",
    logging_steps=50,
    learning_rate=5e-5,
    save_total_limit=3,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)
# Early stopping callback
early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[early_stopping],
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained("fine-tuned-gpt2")
tokenizer.save_pretrained("fine-tuned-gpt2")

# Generate articles


def generate_article(product_category):
    prompt = f"""Generate a short review for an article of the following product category:{product_category}. \
        It must be no longer than 300 characters. Make sure you finish all of the sentences. \
        It should have either a positive, neutral or negative sentiment. \
        Here is an example for product_category = Tablet, sentiment = positive: \
        Great tablet for the the price This tablet is really good for the price, especially on the holidays. \
        If you are a movie watcher or just using it for games it has great speed with a quad core. \
        Highly recommend so e-book as well. Very happy with my purchase I bought 3 already. \
        Here is an example for product_category = E-book, sentiment = neutral: \
        \n"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Create attention mask (1s for tokens that should be attended to, 0s for padding)
    attention_mask = torch.ones(input_ids.shape, device=device)

    # Generate with attention mask and explicitly set pad_token_id
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=300,
        num_beams=3,
        no_repeat_ngram_size=2,
        temperature=0.6,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    article = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return article


# Example usage
unique_categories = reviews["product_category"].unique()
for category in unique_categories:
    article = generate_article(category)
    print(f"Article for {category}:\n")
    print(article)
    print("\n" + "=" * 80 + "\n")
