import re
import nltk
from typing import List
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from logic.text_processing import TextProcessing
from logic.lexical_features import lexical_es, lexical_en
from nltk.corpus import stopwords


class FeatureExtraction(BaseEstimator, TransformerMixin):

    def __init__(self, lang='es', text_processing=None):
        try:
            if text_processing is None:
                self.tp = TextProcessing(lang=lang)
            else:
                self.tp = text_processing
            self.lexical = lexical_es if lang == 'es' else lexical_en
        except Exception as e:
            print('Error FeatureExtraction: {0}'.format(e))

    def fit(self, x, y=None):
        return self

    def transform(self, messages: List[str]):
        try:
            result = self.get_features(messages)
            return result
        except Exception as e:
            print('Error transform: {0}'.format(e))

    def get_features(self, messages: List[str]):
        try:
            lexical_features = self.get_features_lexical(messages)

            # features = np.append(lexical_features, bow_features, axis=1)
            features = np.array(lexical_features)
            return features
        except Exception as e:
            print('Error get_features: {0}'.format(e))


    def get_features_lexical(self, messages: List[str]):
        try:
            lexical = self.lexical
            tags = ('mention', 'url', 'hashtag', 'emoji', 'rt')
            result = dict()
            i = 0
            for msg in messages:
                vector = dict()
                tokens_text = TextProcessing.tokenizer(msg)
                if len(tokens_text) > 0:
                    vector['label_mention'] = float(sum(1 for word in tokens_text if word == 'mention'))
                    vector['label_url'] = float(sum(1 for word in tokens_text if word == 'url'))
                    vector['label_hashtag'] = float(sum(1 for word in tokens_text if word == 'hashtag'))
                    vector['label_emoji'] = float(sum(1 for word in tokens_text if word == 'emoji'))
                    vector['label_retweets'] = float(sum(1 for word in tokens_text if word == 'rt'))

                    label_word = vector['label_mention'] + vector['label_url'] + vector['label_hashtag']
                    label_word = label_word + vector['label_emoji'] + vector['label_retweets']
                    vector['label_word'] = float(len(tokens_text) - label_word)

                    vector['first_person_singular'] = float(
                        sum(1 for word in tokens_text if word in lexical['first_person_singular']))
                    vector['second_person_singular'] = float(
                        sum(1 for word in tokens_text if word in lexical['second_person_singular']))
                    vector['third_person_singular'] = float(
                        sum(1 for word in tokens_text if word in lexical['third_person_singular']))
                    vector['first_person_plurar'] = float(
                        sum(1 for word in tokens_text if word in lexical['first_person_plurar']))
                    vector['second_person_plurar'] = float(
                        sum(1 for word in tokens_text if word in lexical['second_person_plurar']))
                    vector['third_person_plurar'] = float(
                        sum(1 for word in tokens_text if word in lexical['third_person_plurar']))

                    vector['avg_word'] = np.nanmean([len(word) for word in tokens_text if word not in tags])
                    vector['avg_word'] = vector['avg_word'] if not np.isnan(vector['avg_word']) else 0.0
                    vector['avg_word'] = round(vector['avg_word'], 4)

                    vector['kur_word'] = kurtosis([len(word) for word in tokens_text if word not in tags])
                    vector['kur_word'] = vector['kur_word'] if not np.isnan(vector['kur_word']) else 0.0
                    vector['kur_word'] = round(vector['kur_word'], 4)

                    vector['skew_word'] = skew(np.array([len(word) for word in tokens_text if word not in tags]))
                    vector['skew_word'] = vector['skew_word'] if not np.isnan(vector['skew_word']) else 0.0
                    vector['skew_word'] = round(vector['skew_word'], 4)

                    # adverbios
                    vector['adverb_neg'] = sum(1 for word in tokens_text if word in lexical['adverb_neg'])
                    vector['adverb_neg'] = float(vector['adverb_neg'])

                    vector['adverb_time'] = sum(1 for word in tokens_text if word in lexical['adverb_time'])
                    vector['adverb_time'] = float(vector['adverb_time'])

                    vector['adverb_place'] = sum(1 for word in tokens_text if word in lexical['adverb_place'])
                    vector['adverb_place'] = float(vector['adverb_place'])

                    vector['adverb_mode'] = sum(1 for word in tokens_text if word in lexical['adverb_mode'])
                    vector['adverb_mode'] = float(vector['adverb_mode'])

                    vector['adverb_cant'] = sum(1 for word in tokens_text if word in lexical['adverb_cant'])
                    vector['adverb_cant'] = float(vector['adverb_cant'])

                    vector['adverb_all'] = float(vector['adverb_neg'] + vector['adverb_time'] + vector['adverb_place'])
                    vector['adverb_all'] = float(vector['adverb_all'] + vector['adverb_mode'] + vector['adverb_cant'])

                    vector['adjetives_neg'] = sum(1 for word in tokens_text if word in lexical['adjetives_neg'])
                    vector['adjetives_neg'] = float(vector['adjetives_neg'])

                    vector['adjetives_pos'] = sum(1 for word in tokens_text if word in lexical['adjetives_pos'])
                    vector['adjetives_pos'] = float(vector['adjetives_pos'])

                    vector['who_general'] = sum(1 for word in tokens_text if word in lexical['who_general'])
                    vector['who_general'] = float(vector['who_general'])

                    vector['who_male'] = sum(1 for word in tokens_text if word in lexical['who_male'])
                    vector['who_male'] = float(vector['who_male'])

                    vector['who_female'] = sum(1 for word in tokens_text if word in lexical['who_female'])
                    vector['who_female'] = float(vector['who_female'])

                    vector['hate'] = sum(1 for word in tokens_text if word in lexical['hate'])
                    vector['hate'] = float(vector['hate'])

                result[i] = list(vector.values())
                i += 1
            features = pd.DataFrame.from_dict(result, orient='index').fillna(0.0)
            # Normilizar
            return features
        except Exception as e:
            print('Error get_lexical_features: {0}'.format(e))

