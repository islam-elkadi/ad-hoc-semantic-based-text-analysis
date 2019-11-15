# Ad-hoc semantic search & retrieve engine using Unsupervised Learning

Hello world! The objective of this project is to create a tool that allows users to extract customer sentiment visual & summarized insights from user reviews in order to discover & leverage hidden insights at scale to better drive business decisions. Unstructured data, such as user reviews, contain a wealth of information that can prove to be very valuable if there is a mechanism to ingest, proocess, and uncover valuable data frorm it.

There are many approaches and implementationos to the same problem, the majority of such implementations focus on **supervised learning**. However, I attempted to solve this problem using **unsupervised learning**. I personally believe that a supervised learning approach can provide better results, however, supervised learning typically requires deep domain expertise especially in the annotation process; this is very labor intensive and time consuming. 

On the other hand, unsupervised learing will spare you the hassle of requiring deep domain expetise for data annotation, which will significantly reduce time and efforts. This could be at the expense of peformance, but personally I've seen the peformance to be pretty good. So the trade-off for me to switch to supervised learning hasn't been too high to force me to make a switch to solving the aforementioned probelm with supervised leaning techniques.

This capabilities of this project are deployed as RESTful APIs.

# Installation

This project was developed using Python 3.6. To install the required libraries run the following command.

```
pip install -r requirements.txt
```

# Offered functionality

The core capabilities of this project where developed using the wonderful [Gensim] library. The API endpoints developed using Gensim provide the following functionalities:

* Text summarization
* Semantic search & retrieval of sentances based on queried
* Categorical sentiment analysis, granular level breakdown of expressed sentiment, and keyword extraction based on queried topics
* Preprocessing text information in preparation for training
* Training Word2vec & FastText word embedding models
* Training Doc2Vec models

# Usage

Fire up a local rest server using the following command:
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


[Gensim]: https://radimrehurek.com/gensim/
