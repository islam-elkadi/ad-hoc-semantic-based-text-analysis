from utils import *

app=Flask(__name__)

#------------------------------------------------------------------------------------------------#
#                                       Error handling and test                                  #
#------------------------------------------------------------------------------------------------#
@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({"error": "Bad request"}),400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({"error": "Not found"}),404)

@app.route("/",methods=["GET"])
def test_app():
    return jsonify({"success": "true"})

#------------------------------------------------------------------------------------------------#
#                                      Non-Functional Endpoints                                  #
#------------------------------------------------------------------------------------------------#
@app.route("/train_FastText",methods=["POST"])
def train_FastText():
   
    pretrained_path=request.json.get("pretrainedPath")
    train_path=request.json.get("trainPath")
    train_params=request.json.get("train_params")
    build_params=request.json.get("build_params")
    model_store_path=request.json.get("savePath")
        
    # instantiate new model isntance with specified parameters or updated existing model
    if pretrained_path:
        model_gensim=FT.load(pretrained_path)
    else:
        model_gensim=FT(**train_params)
    
    # build model vocab
    model_gensim.build_vocab(**build_params)

    # train the model
    model_gensim.train()
    
    # save model
    save_model(model_gensim,model_store_path)
    
    return jsonify({"FastText Training":"Success"})

@app.route("/train_word2vec",methods=["POST"])
def train_word2vec():
   
    pretrained_path=request.json.get("pretrainedPath")
    train_path=request.json.get("trainPath")
    train_params=request.json.get("train_params")
    build_params=request.json.get("build_params")
    model_store_path=request.json.get("savePath")
        
    # instantiate new model isntance with specified parameters or updated existing model
    if pretrained_path:
        model_gensim=WV.load(pretrained_path)
    else:
        model_gensim=WV(**train_params)
    
    # build model vocab
    model_gensim.build_vocab(**build_params)

    # train the model
    model_gensim.train()
    
    # save model
    save_model(model_gensim,model_store_path)
    
    return jsonify({"Word2Vec Training":"Success"})

@app.route("/train_doc2vec",methods=["POST"])
def train_doc2vec():
    
    pretrained_path=request.json.get("pretrained_path")
    train_path=request.json.get("train_path")
    model_store_path=request.json.get("save_path")
    train_params=request.json.get("train_params")
    build_params=request.json.get("build_params")
            
    # instantiate new model instance with specified parameters or updated existing model   
    if pretrained_path:
        model_gensim=D2V.load(pretrained_path)
    else:
        model_gensim=D2V(**train_params)
    
    # build model vocab
    model_gensim.build_vocab(**build_params)

    # train the model
    model_gensim.train()
    
    # remove temporary training data to recuce memorary consumption
    model_gensim.delete_temporary_training_data(keep_doctags_vectors=True,keep_inference=True)
    
    # save model
    save_model(model_gensim,model_store_path)
    
    return jsonify({"Doc2Vec Training":"Success"})

#------------------------------------------------------------------------------------------------#
#                                       Functional Endpoints                                     #
#------------------------------------------------------------------------------------------------#
@app.route("/preprocess_data",methods=["POST"])
def preprocess_data():
    
    target=request.json.get("target")
    save_path=request.json.get("save_path")
    
    df=pd.read_csv("path")
    df[target]=df[target].apply(lambda x: clean_text(x))
    df.dropna(subset=[target],inplace=True)
    with open("{}.txt".format(save_path),"w") as fp:
        fp.write("\n".join(df[target]))
        
    return jsonify({"Preprocessing":"Complete"})

@app.route("/classify_sentence",methods=["POST"])
def classify_sentence():

    data=request.json.get("data")
    categories=request.json.get("categories")

    categories=[x for x in categories if x in model.vocab]
    synonyms=get_syns(model,categories)
    
    clean_data=clean_text(data)
    token_sents=split_sentences(clean_data)
    token_words=text_to_word_tokens(model,clean_data)

    classifications=classify_sentences(model,token_sents,token_words,synonyms,categories)

    return jsonify(classifications)

@app.route("/get_sentiments",methods=["POST"])
def get_sentiments():

    ratio=request.json.get("ratio")
    split=request.json.get("split")
    data=request.json.get("data")
    categories=request.json.get("categories")
    
    categories=[x for x in categories if x in model.vocab]
    synonyms=get_syns(model,categories)

    if split==1:
        results=[]
        for text in data:
            text=clean_text(text)
            token_sents=split_sentences(text)
            token_words=text_to_word_tokens(model,text)
            classifications=classify_sentences(model,token_sents,token_words,synonyms,categories)
            result=analyze_emotions(classifications,ratio)
            results.append(result)
    else:
        data=" ".join(data)
        data=clean_text(data)
        token_sents=split_sentences(data)
        token_words=text_to_word_tokens(model,data)
        classifications=classify_sentences(model,token_sents,token_words,synonyms,categories)
        results=analyze_emotions(classifications,ratio)

    return jsonify(results)

@app.route("/summarize_text",methods=["POST"])
def summarize_text():
    text=request.json.get("text")
    ratio=request.json.get("ratio")
    result=create_summary(text,ratio)
    return jsonify(result)


#-------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    model=api.load("word2vec-google-news-300")
    app.run(host="0.0.0.0",port=5000,debug=True)
