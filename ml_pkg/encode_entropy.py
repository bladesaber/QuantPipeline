import numpy as np


class EncodeEntropyUtils:
    @staticmethod
    def message_shannon_entropy(message: str) -> float:
        encoding_dict = {}
        for i in message:
            if i not in encoding_dict:
                encoding_dict[i] = 0
            encoding_dict[i] += 1
        probabilities = [i / len(message) for i in encoding_dict.values()]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    @staticmethod
    def message_lempel_ziv_entropy(message: str) -> float:
        i = 0
        encoding_dict = []
        while i < len(message):
            for j in range(i, len(message)):
                message_ = message[i:j + 1]
                if message_ not in encoding_dict:
                    encoding_dict.append(message_)
                    break
            i = j + 1
        return len(encoding_dict) / len(message)

    @staticmethod
    def message_word_shannon_entropy(message: str, word_length: int) -> float:
        encoding_dict = {}
        for i in range(len(message) - word_length + 1):
            word = message[i:i + word_length]
            if word not in encoding_dict:
                encoding_dict[word] = 0
            encoding_dict[word] += 1
        probabilities = [i / len(message) for i in encoding_dict.values()]
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
