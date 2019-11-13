from gensim.models.fasttext import FT
from gensim.models.doc2vec import Doc2Vec as D2V, TaggedDocument

def save_model(model,path):
    model.save(path)
    return True

def load_model(path):
    model=model.load(path)
    return model
