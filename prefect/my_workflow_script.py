import pandas as pd
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from prefect import task, flow

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

#Loading Data
@task
def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)
    
# Filter Data
@task
def filter_data(data):
    """
    Filtering data loaded from dataframe
    """
    data = data.drop_duplicates()
    data = data.dropna(subset=['Review text'])
    data['Sentiment'] = data['Ratings'].apply(lambda x: 'Positive' if x >= 3 else 'Negative')
    data = data[['Review text', 'Sentiment']]
    return data
    
# Split Input & Output
@task
def split_inputs_output(data, inputs, output):
    """
    Split features and target variables.
    """
    X = data[inputs]
    y = data[output]
    return X, y
    
# Split Into Train & Test
@task
def split_train_test(X, y, test_size=0.25, random_state=0):
    """
    Split data into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
# Preprocess Text
@task
def preprocess(X_train, X_test):
    lemmatizer = WordNetLemmatizer()
    # Define a function for cleaning text
    def clean_text(text):
        # Removing special characters and digits
        sentence = re.sub("[^a-zA-Z]", " ", text)
        # Change sentence to lower case
        sentence = sentence.lower()
        # Tokenize into words
        tokens = sentence.split()
        # Remove stop words
        clean_tokens = [t for t in tokens if not t in stopwords.words("english")]
        # Lemmatization
        clean_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens]
        return " ".join(clean_tokens)

    # Apply the cleaning function to X_train and X_test
    X_train_clean = X_train.apply(clean_text)
    X_test_clean = X_test.apply(clean_text)

    return X_train_clean, X_test_clean
    
# Vectorizaion
@task
def preprocess_data(X_train_clean, X_test_clean, y_train):
    """
    Rescale the data.
    """
    scaler = CountVectorizer()
    X_train_scaled = scaler.fit_transform(X_train_clean)
    X_test_scaled = scaler.transform(X_test_clean)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)    
    return X_train_resampled, y_train_resampled, X_test_scaled
    
# Model Training
@task
def train_model(X_train_resampled, y_train_resampled, hyperparameters):
    """
    Training the machine learning model.
    """
    y_train_xgb = y_train_resampled.map({"Negative": 0, "Positive": 1})
    clf = XGBClassifier(**hyperparameters)
    clf.fit(X_train_resampled, y_train_xgb)
    return clf
    
# Evaluation
@task
def evaluate_model(model, X_train_resampled, y_train_resampled, X_test_scaled, y_test):
    """
    Evaluating the model.
    """
    y_train_pred_ = model.predict(X_train_resampled)
    y_train_pred = pd.Series(y_train_pred_).map({0: "Negative", 1: "Positive"})
    
    # Train Results
    train_accuracy = accuracy_score(y_train_resampled, y_train_pred)
    train_f1_pos = f1_score(y_train_resampled, y_train_pred, pos_label='Positive')
    train_f1_neg = f1_score(y_train_resampled, y_train_pred, pos_label='Negative')
    train_report = classification_report(y_train_resampled, y_train_pred)

    y_test_pred_ = model.predict(X_test_scaled)
    y_test_pred = pd.Series(y_test_pred_).map({0: "Negative", 1: "Positive"})
    
    # Test Results
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1_pos = f1_score(y_test, y_test_pred, pos_label='Positive')
    test_f1_neg = f1_score(y_test, y_test_pred, pos_label='Negative')
    test_report = classification_report(y_test, y_test_pred)
    
    return train_accuracy, train_f1_pos, train_f1_neg, train_report, test_accuracy, test_f1_pos, test_f1_neg, test_report 
    
# Workflow
@flow(name="Sentiment_Analsyis_XGBoost Flow")
def workflow():
    DATA_PATH = 'data.csv'
    INPUTS = 'Review text'
    OUTPUT = 'Sentiment'
    HYPERPARAMETERS = {'n_estimators': 100, 'learning_rate': 0.5, 'max_depth': 5}
    
    # Load data
    data = load_data(DATA_PATH)

    # Filter Data
    data = filter_data(data)

    # Identify Inputs and Output
    X, y = split_inputs_output(data, INPUTS, OUTPUT)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Preprocess the data
    X_train_clean, X_test_clean = preprocess(X_train, X_test)

    # Resample and scale the data
    X_train_resampled, y_train_resampled, X_test_scaled = preprocess_data(X_train_clean, X_test_clean, y_train)
    
    # Build a model
    model = train_model(X_train_resampled, y_train_resampled, HYPERPARAMETERS)
    
    # Evaluation
    train_accuracy, train_f1_pos, train_f1_neg, train_report, test_accuracy, test_f1_pos, test_f1_neg, test_report = evaluate_model(model, X_train_resampled, y_train_resampled, X_test_scaled, y_test)
    
    print("*" * 100)
    print("Train Score:")
    print("*" * 50)
    print('Accuracy on Train Data: ', train_accuracy)
    print('F1 Score (Positive Class) on Train Data: ', train_f1_pos)
    print('F1 Score (Negative Class) on Train Data: ', train_f1_neg)

    # Display Train Classification Report
    print('Classification Report For Train Data:\n', train_report)
    print("*" * 100)
    print("Test Score:")
    print("*" * 50)
    print('Accuracy on Train Data: ', test_accuracy)
    print('F1 Score (Positive Class) on Train Data: ', train_f1_pos)
    print('F1 Score (Negative Class) on Train Data: ', train_f1_neg)

    # Display Classification Report    
    print('Classification Report For Train Data:\n', test_report)


if __name__ == "__main__":
    workflow.serve(
        name="my-first-deployment",
        cron="15 * * * *"
    )