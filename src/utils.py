from sys_utils import *

#-----------------------------------------------
#                   Data Cleaning               
#-----------------------------------------------

def clean_text(data):
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
    sim_mat=np.zeros([len(tokens),len(vecs)])
    for i,token in enumerate(tokens):
        for j,vector in enumerate(vecs):
            sim_mat[i,j]=model.n_similarity(vector,token)
    return sim_mat

def classify_sentences(model,token_sents,token_words,synonyms,categories):
    token_sents=np.array(token_sents)
    sim_mat=similarity_matrix(model,token_words,synonyms)        
    pred=np.argmax(sim_mat,axis=1)

    classifications={}
    for i,category in enumerate(categories):
        idx=np.where(pred == i)[0].tolist()
        classifications[category]=token_sents[idx].tolist()
    return classifications

#-----------------------------------------------
#               Sentiment Analysis               
#-----------------------------------------------

def create_label(value):
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
    keys=counter_item.keys()
    values=counter_item.values()
    averages=["{}%".format(round(float(value/sum(values))*100)) for value in values]
    zipped_ratios=dict(zip(keys,averages))
    return zipped_ratios

def sentiment_labels_ratios(values):
    list_of_labels=[create_label(TextBlob(value).sentiment.polarity) for value in values]
    counter_item=Counter(list_of_labels)
    ratios=sentiment_ratios(counter_item)
    
    if len(set(counter_item.values())) != 1:
        overall_sentiment=counter_item.most_common(1)[0][0]
        return {"overall_sentiment":overall_sentiment,"ratios":ratios}
    else:
        avg_score=mean([TextBlob(value).sentiment.polarity for value in values])
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
    summary=summarize(text,ratio=ratio)
    if summary:
        summary={"summary": " ".join([sentence.capitalize() for sentence in sent_tokenize(summary)])}
    else:
        summary={"summary": "Input too short to summarize."}
    return summary
