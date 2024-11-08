![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project: Product Reviews

- This project aims to classify product reviews, cluster them into categories, and generate product articles based on sentiment analysis. The workflow includes cleaning the dataset, sentiment classification using zero-shot learning, product category clustering, and attempting to generate articles with a language model (LLM).


## Dataset
- The dataset contains 35,000 product reviews, with relevant columns like:

reviews.title: The title of the review.
reviews.text: The full text of the review.
reviews.rating: The product rating provided by customers.

Follow the instructions provided in the notebook.

Read the instructions for each cell and provide your answers. Make sure to test your answers in each cell and save. Jupyter Notebook should automatically save your work progress. But it's a good idea to periodically save your work manually just in case.

## Workflow
1. Data Cleaning
The first step involved cleaning the dataset by combining the review title and text into a single field. Missing values were handled, and the data was prepared for classification.

2. Sentiment Classification
The sentiment analysis was done using a zero-shot classifier based on transformers (BART-large-mnli). Reviews were categorized into:

Positive
Neutral
Negative
The classifier's performance was evaluated using accuracy, precision, recall, and F1 score metrics.

3. Clustering Product Categories
The text from the reviews was embedded using a pre-trained transformer model (sentence-transformers/all-MiniLM-L6-v2) and then clustered into product categories using K-Means. PCA was used for dimensionality reduction, and an optimal number of clusters was selected using the elbow method. Clusters were assigned labels such as "Tablet," "Smart Speaker," "Kindle," and others.

4. Article Generation (LLM)
The final step attempted to generate product articles based on sentiment and product categories using GPT-2. Although the model was fine-tuned, this step did not succeed as expected.
