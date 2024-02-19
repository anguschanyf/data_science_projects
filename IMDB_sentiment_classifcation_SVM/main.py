import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import warnings

# Download NLTK resources (if not already downloaded)
import nltk
nltk.download('stopwords')

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Step 1: Read multiple txt files into a dataframe
def read_txt_files(folder_path, label):
    data = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                data.append(content)
                labels.append(label)

    df = pd.DataFrame({'Text': data, 'Label': labels})
    return df

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    
    # Stemming (you can also experiment with lemmatization)
    stemmer = PorterStemmer()
    text = ' '.join(stemmer.stem(word) for word in text.split())
    
    return text

# Step 2: Train SVM model and predict on testing data with grid search
def train_and_predict_with_grid_search(train_df, test_df, feature_values=None):
    if feature_values is None:
        feature_values = [10, 100, 500, 1000, 5000, 10000]

    # Apply text preprocessing
    train_df['Text'] = train_df['Text'].apply(preprocess_text)
    test_df['Text'] = test_df['Text'].apply(preprocess_text)

    # Split the data into training and testing sets
    X_train, X_test = train_df['Text'], test_df['Text']
    y_train, y_test = train_df['Label'], test_df['Label']


    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf=True)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Scale your features with `with_mean=False` for sparse matrices
    scaler = StandardScaler(with_mean=False)
    X_train_tfidf_scaled = scaler.fit_transform(X_train_tfidf)
    X_test_tfidf_scaled = scaler.transform(X_test_tfidf)

    # Create the pipeline with feature selection and LinearSVC
    pipeline = Pipeline([
        ('k_best', SelectKBest(f_classif)),
        ('scaler', scaler),
        ('svm', LinearSVC(max_iter=10000, dual=False, class_weight='balanced'))  # Adjusted for class imbalance
    ])

    # Define the parameter grid for grid search
    param_grid = {
        'k_best__k': feature_values,
        'svm__C': [0.1, 1, 10],
    }

    # Perform grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train_tfidf_scaled, y_train)

    # Get the best parameters from the grid search
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    # Predict on testing data using the best model
    X_test_final_tfidf = tfidf_vectorizer.transform(test_df['Text'])
    X_test_final_tfidf_scaled = scaler.transform(X_test_final_tfidf)
    predictions = grid_search.predict(X_test_final_tfidf_scaled)

    # Evaluate the model
    print("Evaluation Metrics:")
    print("Accuracy:", accuracy_score(test_df['Label'], predictions))
    print("Precision:", precision_score(test_df['Label'], predictions))
    print("Recall:", recall_score(test_df['Label'], predictions))
    print("F1 Score:", f1_score(test_df['Label'], predictions))

    return predictions

# Example usage
train_positive_folder = 'train/pos'
train_negative_folder = 'train/neg'
test_positive_folder = 'test/pos'
test_negative_folder = 'test/neg'

# Read training data
train_positive_df = read_txt_files(train_positive_folder, label=1)
train_negative_df = read_txt_files(train_negative_folder, label=0)
train_df = pd.concat([train_positive_df, train_negative_df], ignore_index=True)

# Read testing data
test_positive_df = read_txt_files(test_positive_folder, label=1)
test_negative_df = read_txt_files(test_negative_folder, label=0)
test_df = pd.concat([test_positive_df, test_negative_df], ignore_index=True)

# Train and predict using SVM model with grid search and text preprocessing
predictions = train_and_predict_with_grid_search(train_df, test_df)
