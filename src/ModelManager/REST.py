from utils import *
from flask import Flask, jsonify, request, make_response

app = Flask(__name__)

#------------------------------------------------------------------------------------------------#
#                                        Error Handling                                         #
#------------------------------------------------------------------------------------------------#

@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({"error": "Bad request"}), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({"error": "Not found"}), 404)

@app.route("/", methods = ["GET"])
def test_app():
    return jsonify({"success": "true"})

#------------------------------------------------------------------------------------------------#
#                                        Endpoint Testing                                        #
#------------------------------------------------------------------------------------------------#

@app.route("/clean_data", methods = ["POST"])
def clean_data():
    text = request.json.get("text")
    text = clean_text(text)
    return jsonify({"cleaned_text":text})


#------------------------------------------------------------------------------------------------#
#                                       Main Functionality                                       #
#------------------------------------------------------------------------------------------------#

@app.route("/train_gensim", methods = ["POST"])
def train_gensim():

    path = request.json.get("path")

    kwargs = {}
    kwargs["min_count"] = request.json.get("min_count")
    kwargs["size"] = request.json.get("size")
    kwargs["window"] = request.json.get("window")
    kwargs["workers"] = request.json.get("workers")
    kwargs["alpha"] = request.json.get("alpha")
    kwargs["min_alpha"] = request.json.get("min_alpha")
    kwargs["sg"] = request.json.get("sg")
    kwargs["hs"] = request.json.get("hs")
    kwargs["seed"] = request.json.get("seed")
    kwargs["max_vocab_size"] = request.json.get("max_vocab_size")
    kwargs["sample"] = request.json.get("sample")
    kwargs["lr"] = request.json.get("lr")
    kwargs["negative"] = request.json.get("negative")
    kwargs["ns_exponent"] = request.json.get("ns_exponent")
    kwargs["cbow_mean"] = request.json.get("cbow_mean")
    kwargs["iteration"] = request.json.get("iter")
    kwargs["trim_rule"] = request.json.get("trim_rule")
    kwargs["sorted_vocab"] = request.json.get("sorted_vocab" )
    kwargs["batch_words"] = request.json.get("batch_words")
    kwargs["min_n"] = request.json.get("min_n")
    kwargs["max_n"] = request.json.get("max_n")
    kwargs["word_ngrams"] = request.json.get("word_ngrams")
    kwargs["bucket"] = request.json.get("bucket")
    kwargs = {k: v for k, v in kwargs.items() if v}
    keys, values = zip(*dictionary.items())
    
    values = product(*values)
    for combinations in values:
        kwargs = dict(zip(keys, combinations))
        model_gensim = FTG(**kwargs)
        model_gensim.build_vocab(corpus_file=corpus_file)

        # train the model
        model_gensim.train(
            corpus_file=corpus_file, 
            epochs=model_gensim.epochs,
            total_examples=model_gensim.corpus_count, 
            total_words=model_gensim.corpus_total_words
        )


@app.route("/train_word2vec", methods = ["POST"])
def train_word2vec():
    pass

@app.route("/train_glove", methods = ["POST"])
def train_glove():
    pass

@app.route("/train_doc2vec", methods = ["POST"])
def train_doc2vec():
    pass

#-------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 5000, debug = True)
