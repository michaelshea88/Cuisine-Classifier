# Text Classification with Recipe Data
### Summary
A supervised machine learning exercise to use recipe ingredients to categorize the cuisine. 

### Results
Achieved **96% accuracy** using a Naive Bayes classifier.

### Process
![alt text](https://github.com/michaelshea88/Cuisine-Classifier/blob/master/images/process.png "process")

### Background
Websites like Allrecipes, Epicurious and Yummly aggregate millions of recipes from around the web and compile them into a centralized resource that allows users to search for and compare recipes. Some of the recipes pulled onto these sites come with lots of details, such as difficulty level, course, cuisine, total cost of ingredients, cook time, flavor profile, cuisine, etc. Others contain only a list of ingredients and instructions for preparation. 

Since extra details about recipes allow users to search and filter more effectively, many websites have developed algorithms to predict the characteristics of recipes that contain missing information. Below is an overview of the steps I took to replicate the process of using machine learning to infer unknown recipe attributes. 

# Step 1: Get the data
I obtained access to the [Yummly Recipe API](https://developer.yummly.com/), which contains data on over 2 million recipes. The API has very good documentation, and academic access can be requested if you're using it for educational purposes. It allows you to set a variety of search parameters, such as allowed_course, excluded_course, allowed_cuisine and excluded_cuisine. Although there was no apparent limit to the maximum number of responses you could request per API call, I found that batches of 500 worked best. Below is some of the code I used to make an API call. It uses the requests libraryIt returns the response in JSON format if the retrieval was successful.

```python
# use requests library for API call
response = requests.get(url, headers=headers, params=parameters)

# check status code (200 means all good)
print response.status_code

# decode json
new_data = response.json()
```

Once the JSON file was retrieved from the API I converted it to a Python dictionary which I then converted to a pandas DataFrame. 

# Step 2: Store on AWS
Since I was only requesting 500 recipes at a time, I had to make a series of API calls over the course of about a week. I stored the data retrieved from each batch on a Postgres instance on Amazon's Relational Database Service. I used pandas' very handy to_sql function to add slices of data to the database I created. Then when I had all the data compiled in  Postgres, I used the read_sql function to make queries directly into pandas. 

Here's how I sent data to sql. The Postgres instance I created is no longer active.

```python 
from sqlalchemy import create_engine
import pandas as pd

#establish db connection
engine = create_engine('postgresql://treytrey3:113315th3@recipeproject3.czcsc2tr7kct.us-east-1.rds.amazonaws.com:5432/dsicapstone3')

#sample dataframe 
df = pd.read_csv('../ingredients_combined/ingredients_reduced.csv')

#name it and send to sql
name = 'ingredients'
df.to_sql(name, engine, flavor='postgres', if_exists='replace')
```

# Step 3: Normalize text data

```python
from nltk.stem import WordNetLemmatizer
```
I used NLTK's [WordNetLemmatizer](http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.wordnet) to normalize ingredient names. Lemmatization is a natural language processing (NLP) technique that allows you to ignore trivial differences between different forms of the same word. Stanford's [Introduction to Information Retrieval](http://nlp.stanford.edu/IR-book/) is a good resource for learning about NLP. Here they discuss the difference between lemmatization and another technique called stemming:

> Stemming usually refers to a crude heuristic process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational affixes. Lemmatization usually refers to doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word, which is known as the lemma.

For example, compare the output of the word "sausage" when passed through a lemmatizer and a stemmer:

```python
from nltk.stem import WordNetLemmatizer, PorterStemmer

wordnet_lemmatizer = WordNetLemmatizer()
porter_stemmer = PorterStemmer()

print wordnet_lemmatizer.lemmatize('sausages')
print porter_stemmer.stem('sausages')
```
```python
sausage
sausag
```

# Step 4: Create Bag of Words
```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

The above [scikit-learn](scikit-learn.org) "feature_extraction" package was also really important to the success of this project. It contains many common NLP-related tools. One that was especially useful was the TfidfVectorizer. Tf-idf stands for *term frequency inverse document frequency*. It's not complicated; it essentially weights the importance of words based on their frequency in a document. 

So if you were going through recipes as I was, ingredients like onions and salt and pepper aren't very indicative of the cuisine of the recipe. But jalapeno or soy sauce or herring is more important, but also more uncommon. Tf-idf weighs rarer terms more heavily than common terms. [Here is an excellent overview](http://planspace.org/20150524-tfidf_is_about_what_matters/) of the concept. 

# Step 5: Deploy Naive Bayes
The beautiful thing about Naive Bayes, well there are two beautiful things about Naive Bayes. First, it is highly effective at text classification tasks. Second, it doesn't require much tuning at all, unlike other models like logistic regression. 

Here is the classification report I generated from a Multinomial Naive Bayes:
