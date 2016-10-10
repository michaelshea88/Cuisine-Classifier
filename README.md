# Text Classification with Recipe Data
### Summary
A supervised machine learning exercise to use recipe ingredients to categorize the cuisine. 

### Results
Achieved **96% accuracy** using a Naive Bayes classifier.



### Process
1. Obtain data in batches from an API
2. Parse JSON into pandas DataFrames
3. Store data on Amazon Web Services Relational Database Service (AWS RDS)
4. Preprocess text data using Python's NLTK package
5. Select and tune a classifier using Python's scikit-learn package
6. Obtain new unseen data from API to test model's predictive accuracy

### Background
Websites like Allrecipes, Epicurious and Yummly aggregate millions of recipes from around the web and compile them into a centralized resource that's convenient for users. Some of the recipes pulled onto these sites come with lots of details, like difficulty level, total cost of ingredients, cook time, flavor profile, cuisine, etc. Others contain only a list of ingredients and instructions for preparation. Since those extra details allow users to search and filter more effectively, many websites have developed algorithms to predict the characteristics of recipes that contain missing information using data from recipes that do.

Below is an overview of the steps I took to replicate the process of using machine learning to infer unknown recipe attributes. 

# Step 1: Get the data
The goal of this project was to replicate the work of recipe aggregation websites by attempting to develop an algorithm to predict unknown attributes of recipes with missing information, such as the cuisine and the flavor profile. I used data gathered from the [Yummly Recipe API](https://developer.yummly.com/), preprocessed it with the Python natural language processing (NLP) package [NLTK](http://www.nltk.org/), and used Bayesian statistics to predict a recipe's cuisine from a list of its ingredients.



This is the webpage for the Capstone Project I completed for General Assembly's [Data Science Immersive](https://generalassemb.ly/education/data-science-immersive). I used data from [Yummly.com](http://www.yummly.com/) to build a model that predicts a recipe's cuisine from its ingredients. 

```python
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
```
The above two modules from the the [NLTK](http://www.nltk.org/) and [scikit-learn](scikit-learn.org) packages were really important to the success of this project. I used NLTK's [WordNetLemmatizer](http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.wordnet) to normalize ingredient names. Lemmatization is a natural language processing (NLP) technique that allows you to ignore trivial differences between different forms of the same word. Stanford's [Introduction to Information Retrieval](http://nlp.stanford.edu/IR-book/) is a good resource for learning about NLP. Here they discuss the difference between lemmatization and another technique called stemming:

> Stemming usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes. Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma.

For example, compare the output of the word "sausage" when passed through a lemmatizer and a stemmer:

```python
from nltk.stem import WordNetLemmatizer, PorterStemmer

wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()

print wordnet_lemmatizer.lemmatize('sausages')
print porter_stemmer.stem('sausages')
```
sausage
sausag

af
This exact task is actually the topic of an [existing Kaggle competition](https://www.kaggle.com/c/whats-cooking), but I came up with the idea independently while brainstorming something fun to predict for my final project. 

The goal was to use text data from recipes to predict a recipe's cuisine. 


Initially, my idea for the project was to develop an algorithm to predict a recipe's user rating based on it's ingredients, cooking time, difficulty, and instructions. The goal would be to figure out which factors influence a recipe's rating the most. Do short, easy recipes get the highest ratings? Do recipes that include beets get lower ratings than recipes without beets?  However, once I found a good data source I realized I wouldn't be able to predict ratings after all due to 

I used the [Yummly Recipe API](https://developer.yummly.com/) to collect over 25,000 recipes from [Yummly.com](http://www.yummly.com/). 

Use recipe ingredients to categorize the cuisine
