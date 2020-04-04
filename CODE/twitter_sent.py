# Data Science Libraries
import pandas as pd
import numpy as np
from scipy import stats

# Machine Learning Libraries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_validate, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import RANSACRegressor, HuberRegressor, LinearRegression, ElasticNet, ElasticNetCV, Lars, Lasso, LassoLars, LassoLarsIC, OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV, Ridge, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib

# Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Utilities
import os,random, math, pickle
import re
import sys
from collections import Counter
from datetime import datetime

# NLTK -  Library to Play with Natural Language
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Word2Vec
import gensim
from gensim.models import Word2Vec

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


def find_path():
    # Return DATA Folder Path

    data_path = sys.path[0].split('CODE')[0]

    data_path = f'{data_path}\\DATA\\'

    return data_path


def csv_func(data_path , name):

    csv_reader = open(f'{data_path}{name}.csv', 'rb')
    csv_read = pd.read_csv(csv_reader, encoding='latin1')
    csv_reader.close()

    return csv_read


def see_all():
    # Alongate the view on DataFrames

    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)



def plot_styling():

    # plt.style.use('dark_background')
    plt.style.use('seaborn-bright')

    plt.gca().yaxis.grid(True, color='gray')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12

    # plt.tick_params(top='False', bottom='False', left='False', right='False', labelleft='False', labelbottom='True')

    for spine in plt.gca().spines.values():
        spine.set_visible(False)


def plot_titles(data):

    plt.title('Energy vs Time')

    plt.ylabel('Energy')
    plt.xlabel('Time [mins]')

    max_y = data.loc[:,'T_room'].max() + 0.15*(data.loc[:,'T_room'].max())
    max_x = data.loc[:,'Minutes'].max() + 0.15*(data.loc[:,'Minutes'].max())

    plt.ylim((-5, max_y))
    plt.xlim(0,max_x)

    plt.plot(data.loc[:, 'Minutes'], data.loc[:, 'T_room'],
             ',', markersize=3,
             label=f'Room Temperature',
             zorder=1)

    plt.plot(data.loc[:, 'Minutes'], data.loc[:, 'T_amb'],
             ',', markersize=1, alpha= 0.3,
             label=f'Amb. Temperature',
             zorder=3)

    plt.plot(data.loc[:, 'Minutes'], data.loc[:, 'Power']/6000, ',',
             markersize=10, alpha= 0.5, label= 'Power/6000')

    plt.legend(title='Legend:')
    plt.show()


class Clean_Data(object):

    def __init__(self, df):

        self.df = df

        self.reduce_mem_usage()


    def memory_total_reduction(self):

        print('Mem. usage decreased from {:5.2f}Mb  to {:5.2f}Mb ({:.1f}% reduction)'.format(self.start_mem, self.end_mem, 100 * (self.start_mem - self.end_mem) / self.start_mem))


    def reduce_mem_usage(self, verbose=True):

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

        self.start_mem = self.df.memory_usage().sum() / 1024 ** 2

        for col in self.df.columns:

            col_type = self.df[col].dtypes

            if col_type in numerics:

                c_min = self.df[col].min()
                c_max = self.df[col].max()

                if str(col_type)[:3] == 'int':

                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.df[col] = self.df[col].astype(np.int8)

                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.df[col] = self.df[col].astype(np.int16)

                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.df[col] = self.df[col].astype(np.int32)

                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        self.df[col] = self.df[col].astype(np.int64)

                else:

                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        self.df[col] = self.df[col].astype(np.float16)

                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        self.df[col] = self.df[col].astype(np.float32)

                    else:
                        self.df[col] = self.df[col].astype(np.float64)

        self.end_mem = self.df.memory_usage().sum() / 1024 ** 2

        if verbose:
            print('Mem. usage decreased from {:5.2f}Mb  to {:5.2f}Mb ({:.1f}% reduction)'.format(self.start_mem, self.end_mem, 100 * (self.start_mem - self.end_mem) / self.start_mem))


    def remove_characters(self, verbose=True):

        def operation(text, stem=False):

            stop_words = stopwords.words("english")
            stemmer = SnowballStemmer("english")
            TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

            text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
            tokens = []

            for token in text.split():
                if token not in stop_words:
                    if stem:
                        tokens.append(stemmer.stem(token))
                    else:
                        tokens.append(token)

            return " ".join(tokens)

        operation_start_mem = self.end_mem

        self.df.text = self.df.text.apply(lambda x: operation(x))

        self.end_mem = self.df.memory_usage().sum() / 1024 ** 2

        if verbose:
            print('Mem. usage decreased from {:5.2f}Mb  to {:5.2f}Mb ({:.1f}% reduction)'.format(operation_start_mem, self.end_mem, 100 * (operation_start_mem - self.end_mem) / operation_start_mem))


class Data_Pre_Processing(object):

    def __init__(self, df):

        self.df = df

    def get_missing_data(self):

        total = self.df.isnull().sum().sort_values(ascending=False)
        percent = (self.df.isnull().sum() / self.df.isnull().count() * 100).sort_values(ascending=False)

        self.missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    def plot_missing_data(self):

        self.get_missing_data()

        data_to_plot = self.df.isnull().astype(int)
        columns = data_to_plot.columns

        # convert the list to a 2D NumPy array
        data_to_plot = np.array(data_to_plot).reshape((len(data_to_plot.columns), len(data_to_plot)))
        h, w = data_to_plot.shape

        fig = plt.figure(figsize=(19, 15))
        ax = plt.subplot(111)

        im = ax.matshow(data_to_plot, cmap='binary_r', vmin=0, vmax=1)

        plt.yticks(np.arange(h), columns, fontsize=14)

        ax.set_aspect(w / h)

        plt.colorbar(im, cax = fig.add_axes([0.78, 0.5, 0.03, 0.38]))

        plt.title('Missing Data', fontsize=16)

        plt.show()
        plt.clf()


    def get_correlations(self):

        self.corrs = self.df.corr()

        self.correlation = self.correlations(self.corrs)


    class correlations():

        def __init__(self, df):

            self.corr_df = df


        def print(self):
            print(self.corr_df)

        def plot(self):

            f = plt.figure(figsize=(19, 15))

            plt.matshow(self.corr_df, fignum=f.number)

            plt.xticks(range(self.corr_df.shape[1]), self.corr_df.columns, fontsize=14, rotation=45)
            plt.yticks(range(self.corr_df.shape[1]), self.corr_df.columns, fontsize=14)

            cb = plt.colorbar(cmap='jet')
            cb.ax.tick_params(labelsize=14)

            plt.title('Correlation Matrix', fontsize=16)

            plt.show()
            plt.clf()


    def dist_analysis(self):

        self.data_set_dist = self.data_set_distribution(self.df)

        self.df.hist(bins=50, figsize=(20, 20))
        plt.show()
        plt.clf()


    class data_set_distribution(object):

        def __init__(self, df):
            self.dist_df = df

        def operation(self):
            self.target_count = Counter(self.dist_df.loc[:,'target_qualit'])

        def print(self):

            self.operation()

            print(self.target_count)

        def plot(self):

            self.operation()

            plt.figure(figsize=(16, 8))
            plt.bar(self.target_count.keys(), self.target_count.values())
            plt.title("Dataset labels distribuition")

            plt.show()
            plt.clf()

    def id_outliers(self, parameter):
        # Box plot

        sns.boxplot(x=self.df[parameter])

        plt.show()
        plt.clf()

        # Z-Score

        z = np.abs(stats.zscore(self.df))

        threshold = 3

        high_z = np.where(z > threshold)
        print(high_z)

        # IQR score - InterQuartile Range

        df_o1 = self.df

        Q1 = df_o1.quantile(0.25)
        Q3 = df_o1.quantile(0.75)
        IQR = Q3 - Q1

        print(IQR)

        # The data point where we have False that means these values are valid whereas True indicates presence of an outlier
        print(df_o1 < (Q1 - 1.5 * IQR))
        print(df_o1 > (Q3 + 1.5 * IQR))

        # Removing the outliers
        df_out_0 = self.df[(z < threshold).all(axis=1)]

        df_out_1 = df_o1[~((df_o1 < (Q1 - 1.5 * IQR)).any(axis=1))]
        df_out_1 = df_out_1[~((df_o1 > (Q3 + 1.5 * IQR)).any(axis=1))]

        # boston_df_out = boston_df_o1[~((boston_df_o1 < (Q1 - 1.5 * IQR)) | (boston_df_o1 > (Q3 + 1.5 * IQR))).any(axis=1)]

        print(df_out_1.shape)

    def word2vector(self, df):

        self.W2V_SIZE = 300
        self.W2V_WINDOW = 7
        self.W2V_EPOCH = 32
        self.W2V_MIN_COUNT = 10

        WORD2VEC_MODEL = "model.w2v"

        documents = [_text.split() for _text in df.text]

        w2v_model = gensim.models.word2vec.Word2Vec(size=self.W2V_SIZE,
                                                    window=self.W2V_WINDOW,
                                                    min_count=self.W2V_MIN_COUNT,
                                                    workers=8)

        w2v_model.build_vocab(documents)

        words = w2v_model.wv.vocab.keys()
        self.vocab_size = len(words)
        print("Vocab size", self.vocab_size)

        w2v_model.train(documents, total_examples=len(documents), epochs=self.W2V_EPOCH)

        w2v_model.save(WORD2VEC_MODEL)

        self.w2v_model = w2v_model


    def tokenize_text(self, df_train, df_test):

        SEQUENCE_LENGTH = 300

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(df_train.loc[:,'text'])

        # print(df_train[:,'text'].tolist())
        # quit()

        self.vocab_size = len(self.tokenizer.word_index) + 1
        print("Total words", self.vocab_size)

        # x_train = pad_sequences(self.tokenizer.texts_to_sequences(df_train[:,'text']), maxlen=SEQUENCE_LENGTH)
        # x_test = pad_sequences(self.tokenizer.texts_to_sequences(df_test[:,'text']), maxlen=SEQUENCE_LENGTH)

        train_seq = self.tokenizer.texts_to_sequences(df_train.loc[:,'text'].astype(str).tolist())
        test_seq = self.tokenizer.texts_to_sequences(df_test.loc[:,'text'].astype(str).tolist())

        x_train = pad_sequences(train_seq)
        x_test = pad_sequences(test_seq)

        self.orig_train = df_train
        self.orig_test = df_test

        self.tok_train = x_train
        self.tok_test = x_test

        return x_train, x_test


    def label_encoding(self):

        try:

            labels = self.tok_train.target.unique().tolist()
            labels.append("NEUTRAL")

            encoder = LabelEncoder()
            encoder.fit(self.tok_train.target.tolist())

            y_train = encoder.transform(self.tok_train.target.tolist())
            y_test = encoder.transform(self.tok_test.target.tolist())

        except:
            labels = self.orig_train.target.unique().tolist()
            labels.append("NEUTRAL")

            encoder = LabelEncoder()
            encoder.fit(self.orig_train.target.tolist())

            y_train = encoder.transform(self.orig_train.target.tolist())
            y_test = encoder.transform(self.orig_train.target.tolist())

        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        return y_train, y_test

    def emb_layer(self):

        SEQUENCE_LENGTH = 300

        self.embedding_matrix = np.zeros((self.vocab_size, self.W2V_SIZE))

        for word, i in self.tokenizer.word_index.items():
            if word in self.w2v_model.wv:
                self.embedding_matrix[i] = self.w2v_model.wv[word]

        self.embedding_layer = Embedding(self.vocab_size, self.W2V_SIZE,
                                         weights=[self.embedding_matrix],
                                         input_length=SEQUENCE_LENGTH,
                                         trainable=False)

        return self.embedding_layer


class Create_Sets(object):

    def __init__(self, df):

        self.df = df


    def define_trainig_set(self, df, num):

        strat_train_set, strat_test_set = self.splitting(df, num)

        savings_train = strat_train_set.drop('target', axis=1)
        savings_labels_train = strat_train_set['target'].copy()

        savings_test = strat_test_set.drop('target', axis=1)
        savings_labels_test = strat_test_set['target'].copy()

        X = df.drop('target', axis=1)  # Features
        y = df.loc[:, 'target'].copy()  # Labels

        X_train = savings_train
        y_train = savings_labels_train
        X_test = savings_test
        y_test = savings_labels_test

        return X_train, y_train, X_test, y_test, strat_test_set

    def splitting(self, df, num):

        split = StratifiedShuffleSplit(n_splits=1, test_size=num, random_state=42)

        for train_index, test_index in split.split(df, df["target"]):
            strat_train_set = df.loc[train_index]
            strat_test_set = df.loc[test_index]

        return strat_train_set, strat_test_set

    def get_feats_labels(self):

        # self.X_train, self.y_train, _x, _y, resulted_df = self.define_trainig_set(self.df, 0.3)
        #
        # self.X_valid, self.y_valid, self.X_test, self.y_test, _df = self.define_trainig_set(resulted_df, 0.4)

        self.strat_train_set, _strat_ = self.splitting(self.df, 0.3)
        self.strat_valid_set, self.strat_test_set = self.splitting(_strat_, 0.4)

        # test_set = pd.DataFrame({'T_c': X_test.loc[:, 'T_c'],
        #                          'T_g': X_test.loc[:, 'T_g'],
        #                          'T_e': X_test.loc[:, 'T_e'],
        #                          'spindle_pos': X_test.loc[:, 'spindle_pos']})

    def compare_test_train(self, df_test, df_train, column):

        # Create histograms to compare the test and train set

        fig, ax = plt.subplots(figsize=(10, 10))

        sns.distplot(df_train[column].dropna(), color='green', ax=ax).set_title(column, fontsize=16)
        sns.distplot(df_test[column].dropna(), color='purple', ax=ax).set_title(column, fontsize=16)

        plt.xlabel(column, fontsize=15)
        plt.legend(['train', 'test'])
        plt.show()


def decode_sentiment(label):

    decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}

    return decode_map[int(label)]


def tokenize_and_encode(df_train, df_test):

    # Tokenize

    SEQUENCE_LENGTH = 300

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_train.text)

    vocab_size = len(tokenizer.word_index) + 1
    print("Total words", vocab_size)

    x_train = pad_sequences(tokenizer.texts_to_sequences(df_train[:, 'text']), maxlen=SEQUENCE_LENGTH)
    x_test = pad_sequences(tokenizer.texts_to_sequences(df_test[:, 'text']), maxlen=SEQUENCE_LENGTH)

    feat_train = x_train
    feat_valid = x_test


    # Encode

    encoder = LabelEncoder()
    encoder.fit(df_train.target.tolist())

    y_train = encoder.transform(df_train.target.tolist())
    y_test = encoder.transform(df_test.target.tolist())

    label_train = y_train.reshape(-1, 1)
    label_valid = y_test.reshape(-1, 1)


    return feat_train, feat_valid, label_train, label_valid


def main():

    # data_path = find_path()
    #
    # data_set = csv_func(data_path, 'twitter_data')
    #
    # data_set.columns = ["target", "ids", "date", "flag", "user", "text"]
    #
    # data_set['target_qualit'] = data_set.target.apply(lambda x: decode_sentiment(x))
    #
    # data_cleaner = Clean_Data(data_set)
    # data_cleaner.remove_characters()
    # data_cleaner.memory_total_reduction()
    #
    # data_set = data_cleaner.df
    #
    # data_set = data_set
    #
    # pickle.dump(data_set, open("C:\\Users\\diogo\\Desktop\\perkier tech\\Energy\\CODE\\pickle_rick.p", "wb"))

    data_set = pickle.load( open("C:\\Users\\diogo\\Desktop\\perkier tech\\Energy\\CODE\\pickle_rick.p", "rb" ) )

    pre_proc = Data_Pre_Processing(data_set)

    # pre_proc.dist_analysis()
    # pre_proc.data_set_dist.plot()
    #
    # # pre_proc.missing_data()
    # pre_proc.plot_missing_data()

    # print(data_set)
    # quit()

    # print(len(data_set.loc[:,'ids'].unique()))
    # print(data_set.loc[:, 'target'].unique())


    s = Create_Sets(data_set)
    s.get_feats_labels()

    df_train = s.strat_train_set.reset_index(drop=True)
    df_valid = s.strat_valid_set.reset_index(drop=True)
    df_test = s.strat_test_set.reset_index(drop=True)

    pre_proc.word2vector(df_train)

    x_valid, x_test = pre_proc.tokenize_text(df_train=df_train, df_test=df_valid)
    y_train, y_valid = pre_proc.label_encoding()

    # x_valid, x_test, y_valid, y_test = tokenize_and_encode(df_train, df_valid)

    embedding_layer = pre_proc.emb_layer()

    # pickle.dump(embedding_layer, open("C:\\Users\\diogo\\Desktop\\perkier tech\\Energy\\CODE\\emb.p", "wb"))

    print('Consegui')
    quit()




    quit()






if __name__ == "__main__":

    plot_styling()
    see_all()
    main()
