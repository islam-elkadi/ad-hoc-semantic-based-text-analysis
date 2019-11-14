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
#                                       Main Functionality                                       #
#------------------------------------------------------------------------------------------------#

@app.route("/train_FastText",methods=["POST"])
def train_FastText():
   
    train_path=request.json.get("trainPath")
    train_params=request.json.get("train_params")
    build_params=request.json.get("build_params")
    model_store_path=request.json.get("savePath")
        
    # instantiate model isntance with specified parameters    
    model_gensim=FT(**train_params)
    
    # build model vocab
    model_gensim.build_vocab(**build_params)

    # train the model
    model_gensim.train(corpus_file=corpus_file,epochs=model_gensim.epochs,total_examples=model_gensim.corpus_count,total_words=model_gensim.corpus_total_words)
    
    # save model
    save_model(model_gensim,model_store_path)
    
    return jsonify({"FastText Training":"Success"})


@app.route("/train_doc2vec",methods=["POST"])
def train_doc2vec():
    
    train_path=request.json.get("trainPath")
    model_store_path=request.json.get("savePath")
    train_params=request.json.get("train_params")
    build_params=request.json.get("build_params")
            
    # instantiate model isntance with specified parameters    
    model_gensim=D2V(**train_params)
    
    # build model vocab
    model_gensim.build_vocab(**build_params)

    # train the model
    model_gensim.train(corpus_file=corpus_file,epochs=model_gensim.epochs,total_examples=model_gensim.corpus_count,total_words=model_gensim.corpus_total_words)
    
    # remove temporary training data to recuce memorary consumption
    model_gensim.delete_temporary_training_data(keep_doctags_vectors=True,keep_inference=True)
    
    # save model
    save_model(model_gensim,model_store_path)
    
    return jsonify({"FastText Training":"Success"})


#-------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)
