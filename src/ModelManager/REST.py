@app.route("/train_doc2vec",methods=["POST"])
def train_doc2vec():
    
    update=request.json.get("update")
    params=request.json.get("params")
    train_path=request.json.get("trainPath")
    model_store_path=request.json.get("savePath")
    
    if not params["dm"]:
        params["dm"]=
        
    if not params["vector_size"]:
        kwargs["vector_size"]=
    
    if not params["window"]:
        params["window"]=
    
    if not params["alpha"]:
        params["alpha"]=
    
    if not params["min_alpha"]:
        params["min_alpha"]=
    
    if not params["seed"]:
        params["seed"]=
    
    if not params["min_count"]:
        params["min_count"]=  
        
    if not params["max_vocab_size"]:
        params["max_vocab_size"]=  
 
    if not params["sample"]:
        params["sample"]=  

    if not params["workers"]:
        params["workers"]=  

    if not params["epochs"]:
        params["epochs"]=  

    if not params["hs"]:
        params["hs"]=  

    if not params["negative"]:
        params["negative"]=  

    if not params["ns_exponent"]:
        params["ns_exponent"]=  

    if not params["dm_mean"]:
        params["dm_mean"]=  

    if not params["dm_concat"]:
        params["dm_concat"]=  

    if not params["dm_tag_count"]:
        params["dm_tag_count"]=  

    if not params["dbow_words"]:
        params["dbow_words"]=  

    if not params["trim_rule"]:
        params["trim_rule"]=  


    model_gensim=D2V(**params)
    
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
