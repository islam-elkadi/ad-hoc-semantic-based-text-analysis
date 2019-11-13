from utils import *
from flask import Flask,jsonify,request,make_response

app=Flask(__name__)

#------------------------------------------------------------------------------------------------#
#                                        Error Handling                                         #
#------------------------------------------------------------------------------------------------#

@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({"error":"Bad request"}),400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({"error":"Not found"}),404)

@app.route("/",methods=["GET"])
def test_app():
    return jsonify({"success":"true"})

#------------------------------------------------------------------------------------------------#
#                                        Endpoint Testing                                        #
#------------------------------------------------------------------------------------------------#

@app.route("/clean_data",methods=["POST"])
def clean_data():
    text=request.json.get("text")
    text=clean_text(text)
    return jsonify({"cleaned_text":text})


#------------------------------------------------------------------------------------------------#
#                                       Main Functionality                                       #
#------------------------------------------------------------------------------------------------#

@app.route("/train_FastText",methods=["POST"])
def train_FastText():
    
    update=request.json.get("update")
    params=request.json.get("params")
    train_path=request.json.get("trainPath")
    model_store_path=request.json.get("savePath")
    
    if not params["min_count"]:
        params["min_count"]=5
    
    if not params["size"]:
        params["size"]=100
        
    if not params["window"]:
        kwargs["window"]=5
    
    if not params["workers"]:
        params["workers"]=3
    
    if not params["alpha"]:
        params["alpha"]=0.025
    
    if not params["min_alpha"]:
        params["min_alpha"]=0.0001
    
    if not params["sg"]:
        params["sg"]=0
    
    if not params["hs"]:
        params["hs"]=0
    
    if not params["seed"]:
        params["seed"]=1
    
    if not params["sample"]:
        params["sample"]=0.001
    
    if not params["negative"]:
        params["negative"]=5
    
    if not params["ns_exponent"]:
        params["ns_exponent"]=0.75
    
    if not params["cbow_mean"]:
        params["cbow_mean"]=1
    
    if not params["iter"]:
        params["iter"]=5

    if not params["sorted_vocab"]:
        params["sorted_vocab"]=1
    
    if not params["batch_words"]:
        params["batch_words"]=10000
    
    if not params["min_n"]:
        params["min_n"]=3
    
    if not params["max_n"]:
        params["max_n"]=6
    
    if not params["word_ngrams"]:
        params["word_ngrams"]=1
    
    if not params["bucket"]:
        params["bucket"]=2000000
    
    
    model_gensim=FT(**params)
    
    if update:
        model_gensim.build_vocab(corpus_file=corpus_file,update=True)
    else:
        model_gensim.build_vocab(corpus_file=corpus_file)
        
    # train the model
    model_gensim.train(
        corpus_file=corpus_file,
        epochs=model_gensim.epochs,
        total_examples=model_gensim.corpus_count,
        total_words=model_gensim.corpus_total_words
    )
    
    save_model(model_gensim,model_store_path)
    
    return jsonify({"FastText Training":"Success"})

@app.route("/train_glove",methods=["POST"])
def train_glove():
    pass

@app.route("/train_doc2vec",methods=["POST"])
def train_doc2vec():
    pass

#-------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
