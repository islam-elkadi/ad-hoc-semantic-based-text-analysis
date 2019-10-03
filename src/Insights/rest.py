
from utils import *
from gensim.models import KeyedVectors
from flask import Flask, jsonify, request, make_response

app = Flask(__name__)

#------------------------------------------------------------------------------------------------#
#                               Global variables to hold the forecast data                       #
#-------------------------------------------------------------------------------------------------

ratio = 0.15

#------------------------------------------------------------------------------------------------#
#                                       Error handling and test                                  #
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
#                                        Endpoint testing                                        #
#------------------------------------------------------------------------------------------------#

@app.route("/clean_data", methods = ["POST"])
def clean_data():
    text = request.json.get("text")
    text = clean_text(text)
    return jsonify({"cleaned_text":text})

@app.route("/word_tokens", methods = ["POST"])
def create_word_tokens():
    text = request.json.get("text")
    tokens = text_to_word_tokens(model, text)
    return jsonify({"word_tokenization":tokens})

@app.route("/synonym_vectors", methods = ["POST"])
def synonym_vectors():
    categories = request.json.get("categories")
    synonyms = get_syns(model, categories)
    return jsonify({"synonyms": synonyms})

#------------------------------------------------------------------------------------------------#
#                           Main functionality and Route definitions                             #
#------------------------------------------------------------------------------------------------#

@app.route("/sentence_classify", methods = ["POST"])
def sentence_classify():

    data = request.json.get("data")
    categories = request.json.get("categories")

    categories = [x for x in categories if x in model.vocab]
    synonyms = get_syns(model, categories)
    clean_data = clean_text(data)

    token_sents = split_sentences(clean_data)
    token_words = text_to_word_tokens(model, clean_data)

    classifications = classify_sentences(model, token_sents, token_words, synonyms, categories)

    return jsonify(classifications)

@app.route("/get_sentiments", methods = ["POST"])
def get_sentiments():

    data = request.json.get("data")
    categories = request.json.get("categories")
    split = request.json.get("split")

    categories = [x for x in categories if x in model.vocab]
    synonyms = get_syns(model, categories)

    if split == 1:
        results = []
        for text in data:
            text = clean_text(text)
            token_sents = split_sentences(text)
            token_words = text_to_word_tokens(model, text)
            classifications = classify_sentences(model, token_sents, token_words, synonyms, categories)
            result = analyze_emotions(classifications, ratio)
            results.append(result)
    else:
        data = " ".join(data)
        data = clean_text(data)
        token_sents = split_sentences(data)
        token_words = text_to_word_tokens(model, data)
        classifications = classify_sentences(model, token_sents, token_words, synonyms, categories)
        results = analyze_emotions(classifications, ratio)

    return jsonify({"analysis":results})

@app.route("/summarize_text", methods = ["POST"])
def summarize_text():
    text = request.json.get("text")
    result = create_summary(text, ratio)
    return jsonify(result)

#-------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    model = KeyedVectors.load_word2vec_format( "../../models/model.bin.gz", binary = True)
    app.run(host = "0.0.0.0", port = 5000, debug = True)
