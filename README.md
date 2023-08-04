# Twitter Sentiment Analysis

Sentiment analysis is the process of determining the sentiment expressed in a piece of text. In this project, we use the Twitter dataset and a logistic regression model to classify tweets as either positive or negative. The model uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to represent text data in a numerical format suitable for machine learning algorithms.

## Installation
To run this project, you need to have the following Python libraries installed:
  * numpy
  * pandas
  * nltk
  * scikit-learn
  * matplotlib
  * seaborn

You can install the required libraries using the following command:
```
pip install numpy pandas nltk scikit-learn matplotlib seaborn
```

## Usage
1. Download the Twitter [Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) and place it in the project directory.
  
2. Clone this repository to your local machine:
```
git clone https://github.com/melita-celia/twitter-sentiment-analysis.git
cd your_folder
```

3. Run the python file:
```
python filename.py
```

## Data Preprocessing
The dataset is preprocessed to remove irrelevant information and clean the text data. It includes the following steps:
* Lowercasing: Convert all text to lowercase for standardization.
* Removing Special Characters: Remove Twitter handles, URLs, and special characters.
* Tokenization: Split sentences into individual words (tokens).
* Stopword Removal: Remove common stopwords like "the", "and", "is" that do not contribute to sentiment analysis.
* Stemming: Reduce words to their root form using Porter Stemmer.

## Model Training
The logistic regression model is used for sentiment classification. The TF-IDF vectorization method is employed to convert the preprocessed text data into numerical features.

## Model Evaluation
The trained model is evaluated on a test set using precision, recall, F1-score, and accuracy metrics. A confusion matrix is plotted to visualize the model's performance.

## Testing the Model
You can use the trained model to predict the sentiment of any custom text. Simply run the provided function with your text as input.
```
text = "I enjoy watching Formula 1."
```

## License
This project is licensed under the MIT License - see the [LICENSE]() file for details.
