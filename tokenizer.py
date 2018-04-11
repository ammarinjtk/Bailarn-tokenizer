from Tokenizer.model import Model
from Tokenizer.metric import custom_metric
from Tokenizer import constant
from Tokenizer.utils import Corpus, InputBuilder, index_builder, DottableDict

from keras.models import load_model
import numpy as np


class Bailarn_Tokenizer(object):
    def __init__(self, model_path=None, new_model=False, char_index, tag_index):
        self.char_index = char_index
        self.tag_index = tag_index

        self.new_model = new_model
        self.model_path = model_path
        self.model = None

        if not self.new_model:
            if model_path is not None:
                if not os.path.exists(model_path):
                    raise ValueError("File " + model_path + " does not exist")
                self.model = load_model(model_path)
            else:
                self.model = load_model(constant.DEFULT_MODEL_PATH)

    def train(self, X_train, y_train, valid_split=0.1, initial_epoch=None,
              epochs=100, batch_size=32, learning_rate=0.001, shuffle=False):

        # Create new model or load model
        hyper_params = DottableDict({
            "num_step": X_train.shape[1],
            "learning_rate": learning_rate
        })

        if self.new_model:
            initial_epoch = 0
            self.model = Model(hyper_params).model

        # Display model summary before train
        self.model.summary()
        # Train model
        self.model.fit(X_train, y_train, validation_split=valid_split,
                       initial_epoch=initial_epoch, epochs=epochs,
                       batch_size=batch_size, shuffle=shuffle)

        self.new_model = False

    def predict(self, x, word_delimiter="|", char_index):

        # # string mode
        # if isinstance(sentence, str):
        #     mode = "str"
        # else:
        #     try:
        #         # list mode
        #         if all(isinstance(word, str) for word in sentence):
        #             mode = "list"
        #     except TypeError:
        #         pass

        # Predict
        y_pred = self.model.predict(x)
        y_pred = np.argmax(y_pred, axis=2)

        # # Flatten to 1D
        # y_pred = y_pred.flatten()
        # x = x.flatten()

        # Result list
        all_result = list()
        result = list()

        # Process on each character
        for sample_idx, sample in enumerate(x):
            for char_idx, char_tag in enumerate(sample):

                # Character label
                label = y_pred[sample_idx][char_idx]
                char = self.char_index[char_tag]

                # Pad label
                if label == constant.PAD_TAG_INDEX:
                    continue

                # Skip tag for spacebar character
                if char == constant.SPACEBAR:
                    continue

                # Append character to result list
                result.append(char)

                # Tag at segmented point
                if label != constant.NON_SEGMENT_TAG_INDEX:
                    # Append delimiter to result list
                    result.append(word_delimiter)

            all_result.append(("".join(result)).split(word_delimiter))

        return all_result

    def evaluate(self, x_true, y_true):

        # Predict
        y_pred = self.model.predict(x_true)
        y_pred = np.argmax(y_pred, axis=2)

        # Calculate score
        scores = custom_metric(y_true, y_pred)

        # Display score
        for metric, score in scores.items():
            print("{0}: {1:.6f}".format(metric, score))

        return scores

    def save_model(self, filepath):
        """ Save the keras model to a HDF5 file """
        if not self.model:
            raise ValueError("Can't save the model, "
                             "it has not been trained yet")

        if os.path.exists(filepath):
            raise ValueError("File " + filepath + " already exists!")
        self.model.save(filepath)
