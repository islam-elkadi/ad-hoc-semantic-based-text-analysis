from sys_utils import *

#-----------------------------------------------
#                   Data Cleaning               
#-----------------------------------------------

def clean_text(data):
    """
        Cleans text data by:
            1. Removing html tags
            2. Removing multiple whitespaces
            3. Removing ascii characters
            4. 
            5. Replacing contractions into full form
            6. Removing multiple periods or multiple periods followed by comma
            7. Removing white space before or after bracket
            8. Adds a zero before a decimal number missing a zero
            9. 
            10. Capitalize the start of each sentence
            11. Replace blank space to ‘@’ separator after abbreviation and next word.
            12. Remove sentences with two words or less

        Parameters:
            data: text data
        
        Returns:
            data: cleaned data
    """
    data=strip_tags(data)
    data=strip_multiple_whitespaces(data)
    data=re.sub(r"[^\x00-\x9f]",r"",data)
    data=split_alphanum(data)
    data=remove_contractions(data)
    data=re.sub(r"\.\.+|\.+\,",r",",data)
    data=re.sub(r"\W+\)",r")",data)
    data=re.sub(r"\(\W+",r"(",data)
    data=re.sub(r"\s+\.",".",data)
    data=append_zero_to_decimal(data)
    data=replace_abbreviations(data)
    data=" ".join([sent.capitalize() for sent in split_sentences(data) if len(sent.split()) > 2])
    data=",".join(data.split(","))
    return data
    
def text_to_word_tokens(model,data):
    """
        Converts data into list of tokens to be processed for semantic search. Data is further
        cleaned by removing stop words and words not in model vocab.

        Parameters:
            Model: pretrained word embedding model
            data: preprocessed data

        Returns:
            List of processed tokenized sentences
    """
    data=remove_stopwords(data)
    def noise(x):
        if x in model.vocab:
            return True
    sentences=[list(filter(noise,simple_preprocess(sent,min_len=4))) for sent in split_sentences(data)]
    return sentences

#-----------------------------------------------
#                   Word Vectors               
#-----------------------------------------------

def get_syns(model,categories):
    """
        Generates similarity synonyms for list of queries

        Parameters:
            model: pre-trained model
            categories: list of words to generate synoyms for
        Returns:
            synonyms: list of generated synonym vectors
    """
    synonyms=[]
    for word in categories:
        vecs=model.most_similar(positive=[word])
        vecs=[vec[0] for vec in vecs if vec[0] in model.vocab]
        vecs.insert(0,word)
        synonyms.append(vecs)
    return synonyms

#-----------------------------------------------
#             Sentence Classification               
#-----------------------------------------------

def similarity_matrix(model,tokens,vecs):
    """
        Generates NxM cosine similarity matrix between filtered tokenized words and synonym vectors

        Parameters:
            model: pre-trained model
            tokens: list of preprocessed and filtered word-tokenzied sentences
            vecs: generated similarity synonym vectors

        Returns:
            sim_mat: cosine similarity matrix 
    """
    sim_mat=np.zeros([len(tokens),len(vecs)])
    for i,token in enumerate(tokens):
        for j,vector in enumerate(vecs):
            sim_mat[i,j]=model.n_similarity(vector,token)
    return sim_mat

def classify_sentences(model,token_sents,token_words,synonyms,categories):
    """
        Classifies sentences into appropriate category based on highest cosine similarity to quey

        Parameters:
            model: pre-trained model
            token_sents: list of tokenized sentences
            token_words: list of preprocessed and filtered word-tokenzied sentences
            synonyms: generated similarity synonym vectors
            categories: queries

        Return:
            classifications: dictionary of category key and corresponding classified list of sentences
    """
    token_sents=np.array(token_sents)
    sim_mat=similarity_matrix(model,token_words,synonyms)        
    pred=np.argmax(sim_mat,axis=1)

    classifications={}
    for i,category in enumerate(categories):
        idx=np.where(pred==i)[0].tolist()
        classifications[category]=token_sents[idx].tolist()
    return classifications

#-----------------------------------------------
#               Sentiment Analysis               
#-----------------------------------------------

def create_label(value):
    """
        Converts sentiment score into label

        Parameter:
            value: float decimal number 
        
        Return:
            label
    """
    if 0.3 < value <= 1:
        return "very positive"
    elif 0.1 < value <= 0.3:
        return "positive"
    elif -0.1 <= value <= 0.1:
        return "neutral"
    elif -0.3 < value < -0.1:
        return "negative"
    elif -1 <= value <= -0.3:
        return "very negative"

def sentiment_ratios(counter_item):
    """
        Generates average percentage breakdown of sentiment ratios

        Parameters:
            counter_item: dictionary of sentiment labels and frequency

        Returns:
            zipped_ratios: dictionary of sentiment average breakdown and labels
    """
    keys=counter_item.keys()
    values=counter_item.values()
    averages=["{}%".format(round(float(value/sum(values))*100)) for value in values]
    zipped_ratios=dict(zip(keys,averages))
    return zipped_ratios

def sentiment_labels_ratios(values):
    """
        Returns dictionary of overall sentiment and sentiment ratio breakdown

        Parameters:
            values: list of sentences

        Returns: {"overall_sentiment":average_sentiment,"ratios":ratios}

    """
    list_of_labels=[create_label(TextBlob(value).sentiment.polarity) for value in values]
    counter_item=Counter(list_of_labels)
    ratios=sentiment_ratios(counter_item)
    
    if len(set(counter_item.values())) != 1:
        overall_sentiment=counter_item.most_common(1)[0][0]
        return {"overall_sentiment":overall_sentiment,"ratios":ratios}
    else:
        avg_score=np.mean([TextBlob(value).sentiment.polarity for value in values])
        average_sentiment=create_label(avg_score)
        return {"overall_sentiment":average_sentiment,"ratios":ratios}

def analyze_emotions(categorized_sentence,ratio):
    dictionary={}      
    for key,values in categorized_sentence.items():
        if key != "unclassified" and values:
            dictionary[key]=defaultdict(list)
            dictionary[key]["sentences"].extend(values)
            sentiment_labels_and_ratios=sentiment_labels_ratios(values)
            dictionary[key]["sentiment_ratios"].append(sentiment_labels_and_ratios["ratios"])
            dictionary[key]["overall_sentiment"].append(sentiment_labels_and_ratios["overall_sentiment"])
            dictionary[key]["keywords"]=keywords(text=remove_stopwords(" ".join(values)),ratio=ratio,split=True)
    return dictionary

#-----------------------------------------------
#                   Summarizer               
#-----------------------------------------------

def create_summary(text,ratio):
    """
        Creates text summary using gensim text summarizer

        Parameters:
            text: text data
            ratio: size of summary

        Returns:
            summary
    """
    summary=summarize(text,ratio=ratio)
    if summary:
        summary={"summary": " ".join([sentence.capitalize() for sentence in sent_tokenize(summary)])}
    else:
        summary={"summary": "Input too short to summarize."}
    return summary
