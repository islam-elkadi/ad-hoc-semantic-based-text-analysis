from itertools import product

from gensim.models.fasttext import FTG

def save_model(model, path):
    model.save(path)
    return True

def load_model(path):
    model = model.load(path)
    return model

