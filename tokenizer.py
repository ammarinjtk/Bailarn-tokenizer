from Tokenizer.model import Model
from Tokenizer.metric import custom_metric
from Tokenizer import constant
from Tokenizer.utils import Corpus, InputBuilder, index_builder, DottableDict

from keras.models import load_model
import numpy as np


class Bailarn_Tokenizer(object):
    def __init__(self, model_path=None, new_model=False):
        self.new_model = new_model
        self.model_path = model_path
        self.model = None

        if not self.new_model:
            if model_path is not None:
                self.model = load_model(model_path)
            else:
                self.model = load_model(constant.DEFULT_MODEL_PATH)

        # Create index for character and tag
        self.char_index = index_builder(
            constant.CHARACTER_LIST, constant.CHAR_START_INDEX)
        self.tag_index = index_builder(
            constant.TAG_LIST, constant.TAG_START_INDEX)

    def train(self, corpus_directory, word_delimiter="|", tag_delimiter="/",
              model_path=None, num_step=60, valid_split=0.1, initial_epoch=None,
              epochs=100, batch_size=32, learning_rate=0.001, shuffle=False):

        # Load train dataset
        train_dataset = Corpus(corpus_directory, word_delimiter, tag_delimiter)

        # Generate input
        inb = InputBuilder(train_dataset, self.char_index,
                           self.tag_index, num_step)
        x_true = inb.x
        y_true = inb.y

        # Create new model or load model
        hyper_params = DottableDict({
            "num_step": num_step,
            "learning_rate": learning_rate
        })

        if self.new_model:
            initial_epoch = 0
            model = Model(hyper_params).model

        else:
            if not model_path:
                raise Exception("Model path is not defined.")

            if initial_epoch is None:
                raise Exception("Initial epoch is not defined.")

            model = load_model(model_path)

        # Display model summary before train
        model.summary()
        # Train model
        model.fit(x_true, y_true, validation_split=valid_split,
                  initial_epoch=initial_epoch, epochs=epochs,
                  batch_size=batch_size, shuffle=shuffle)

        self.model = model
        self.new_model = False

    def predict(self, sentence=None, corpus_directory=None, word_delimiter="|"):
        texts = " "
        if corpus_directory:
            texts = Corpus(corpus_directory)
            # print("Directory mode:", texts.count, "files.")
        elif sentence:
            texts = Corpus("/")
            texts.add_text(sentence)
            # print("Sentence mode:")
        else:
            texts = " "
            print("Please fill in sentence or corpus_directory!")

        inb = InputBuilder(texts, self.char_index,
                           self.tag_index, num_step=60, y_one_hot=False)

        all_result = list()
        # Run on each text
        for text_idx in range(texts.count):
            # Get character list and their encoded list
            x_true = texts.get_char_list(text_idx)
            encoded_x = inb.get_encoded_char_list(text_idx)

            # Predict
            y_pred = self.model.predict(encoded_x)
            y_pred = np.argmax(y_pred, axis=2)

            # Flatten to 1D
            y_pred = y_pred.flatten()

            # Result list
            result = list()

            # Process on each character
            for idx, char in enumerate(x_true):
                # Character label
                label = y_pred[idx]

                # Pad label
                if label == constant.PAD_TAG_INDEX:
                    continue

                # Append character to result list
                result.append(char)

                # Skip tag for spacebar character
                if char == constant.SPACEBAR:
                    continue

                # Tag at segmented point
                if label != constant.NON_SEGMENT_TAG_INDEX:
                    # Append delimiter to result list
                    result.append(word_delimiter)
            all_result.append(("".join(result)).split(word_delimiter))
        return all_result

    def evaluate(self, sentence=None, corpus_directory=None, model_num_step=60, word_delimiter="|", tag_delimiter="/"):
        # Load test dataset
        if corpus_directory:
            test_dataset = Corpus(
                corpus_directory, word_delimiter, tag_delimiter)
            print("Test for directory mode:", test_dataset.count, "files.")
        elif sentence:
            test_dataset = Corpus("/")
            test_dataset.add_text(sentence)
            print("Test for sentence mode:")
        else:
            print("Error, please fill in sentence or corpus_directory!")

        # Generate input
        inb = InputBuilder(test_dataset, self.char_index, self.tag_index, model_num_step,
                           y_one_hot=False)
        x_true = inb.x
        y_true = inb.y

        # Predict
        y_pred = self.model.predict(x_true)
        y_pred = np.argmax(y_pred, axis=2)

        # Calculate score
        scores, _ = custom_metric(y_true, y_pred)

        # Display score
        for metric, score in scores.items():
            print("{0}: {1:.6f}".format(metric, score))
        return None
