from Tokenizer.tokenizer import Bailarn_Tokenizer

bailarn_tokenizer = Bailarn_Tokenizer()


def tokenize(sentence):
    return bailarn_tokenizer.predict(sentence=sentence)[0]


__all__ = ['Bailarn_Tokenizer', 'tokenize']
