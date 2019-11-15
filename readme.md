# Ad-hoc semantic search & retrieve using Unsupervised Learning

Hello world! The objective of this project is to create a tool that allows users to extract customer sentiment visual & summarized insights from user reviews in order to discover & leverage hidden insights at scale to better drive business decisions. Unstructured data, such as user reviews, contain a wealth of information that can prove to be very valuable if there is a mechanism to ingest, proocess, and uncover valuable data frorm it.

There are many approaches and implementationos to the same problem, the majority of such implementations focus on **supervised learning**. However, I attempted to solve this problem using **unsupervised learning**. I personally believe that a supervised learning approach can provide better results, however, supervised learning typically requires deep domain expertise especially in the annotation process; this is very labor intensive and time consuming. 

On the other hand, unsupervised learing will spare you the hassle of requiring deep domain expetise for data annotation, which will significantly reduce time and efforts. This could be at the expense of peformance, but personally I've seen the peformance to be pretty good. So the trade-off for me to switch to supervised learning hasn't been too high to force me to make a switch to solving the aforementioned probelm with supervised leaning techniques.

This capabilities of this project are deployed as RESTful APIs.

# Developed using
* Python 3.6
* [Gensim]
* [TextBlob]
* [Pandas]
* [Flask]

# Installation
To install the required libraries run the following command.

```
pip install -r requirements.txt
```

# Offered functionality

The API endpoints developed offer the following functionalities:

* Text summarization
* Semantic search & retrieval of sentances based on queried
* Categorical sentiment analysis, granular level breakdown of expressed sentiment, and keyword extraction based on queried topics
* Preprocessing text information in preparation for training
* Training Word2vec & FastText word embedding models
* Training Doc2Vec models

# Usage

First, fire up a local rest server using the following command:
```
python rest_server.py
```

To summarize text:
```
pass
```

To search & retrieve information bsed on queries
```
pass
```

To perform text analysis on user reviews
```
pass
```

To train a Word2Vec model:
```
pass
```

To train a FastTesxt model:
```
pass
```

To train a Doc2Vec model:
```
pass
```



# Next Steps

### Topic modelling using LDAs
As of now, you have to input your topic queries to retrieve insights. However, I'd like to take this a step further by adding a layer where the topic detection is automated using [LDAs].


### Deep level sentiment analysis
I've come across [IBM Watson's powerful NLU service], and more specifically its sentiment analysis capabiliities. What I particularly love about Watson's sentiment analysis tool is that it quantifies the human emotion expressed in the data, rather than just returning an overall sentiment score between -1 & 1. Watson quantifies the following emotions: Joy, Anger, Disgust, Sadness, and Fear. With that being said, I'd definitely like to explore how I can leverage such a feature to deliever deeper sentiment analysis

[Gensim]: https://radimrehurek.com/gensim/
[TextBlob]: https://textblob.readthedocs.io/en/dev/
[Pandas]: https://pandas.pydata.org/
[Flask]: https://github.com/pallets/flask
[LDAs]: https://radimrehurek.com/gensim/models/ldamodel.html
[IBM Watson's powerful NLU service]: https://cloud.ibm.com/docs/services/natural-language-understanding/getting-started.html
