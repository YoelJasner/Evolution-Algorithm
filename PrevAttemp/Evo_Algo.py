import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import fbeta_score, accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, ReLU
from keras.layers import LSTM
from keras import backend as K
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})


class MV_LSTM(torch.nn.Module):
    '''
    LSTM architechture
    '''

    def __init__(self, n_features, seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 20  # number of hidden states
        self.n_layers = 1  # number of LSTM layers (stacked)

        self.l_lstm = torch.nn.LSTM(input_size=n_features,
                                    hidden_size=self.n_hidden,
                                    num_layers=self.n_layers,
                                    batch_first=True)
        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Linear(self.n_hidden * self.seq_len, 2)

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x, self.hidden)
        # lstm_out(with batch_first = True) is
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size, -1)
        return self.l_linear(x)


def load_process_data():
    '''
    read train & validation file pre process the data
    :return: X_train, y_train, X_val, y_val,
    '''
    df_train = pd.read_csv('train.csv', header=None)
    df_validation = pd.read_csv('validate.csv', header=None)
    df_test = pd.read_csv('test.csv', header=None)

    # split to X, y
    X_train = df_train.loc[:, df_train.columns != 0]
    y_train = df_train.loc[:, df_train.columns == 0]
    X_validation = df_validation.loc[:, df_validation.columns != 0]
    y_validation = df_validation.loc[:, df_validation.columns == 0]
    X_test = df_test.loc[:, df_validation.columns != 0]

    # log scaling
    X_train = np.log(X_train)
    X_validation = np.log(X_validation)
    X_test = np.log(X_test)

    # Robust scaler
    scaler = preprocessing.RobustScaler().fit(X_train)

    # move to tensor 700000, 30, 4 with scaling
    # X_train_scale = np.stack(np.split(scaler.transform(X_train), 30, 1), 1)
    # X_validation_scale = np.stack(np.split(scaler.transform(X_validation), 30, 1), 1)

    # move to tensor 700000, 30, 4 without scaling
    # X_train_scale = np.stack(np.split(X_train, 30, 1), 1)
    # X_validation_scale = np.stack(np.split(X_validation, 30, 1), 1)

    # robust scaling
    X_train_scale = scaler.transform(X_train)
    X_validation_scale = scaler.transform(X_validation)
    X_test_scale = scaler.transform(X_test)

    # z-scoring by axis 1 means by rows
    # X_train_scale = preprocessing.scale(X_train, axis=0)
    # X_validation_scale = preprocessing.scale(X_validation, axis=0)

    # plot cluster map
    # sns.clustermap(X_train_scale[:10000, :], cmap='RdBu', vmin=-1, vmax=1)
    # plt.savefig("clustermap")

    return X_train_scale[:10000, :], y_train[:10000], X_validation_scale[:10000, :], y_validation[:10000], X_test_scale


def fbeta_keras(y_true, y_pred, threshold_shift=0):
    '''
    f_beta score implematation using keras
    :param y_true: true label
    :param y_pred: predicted label
    :param threshold_shift:
    :return: calculate f0.25 score
    '''
    beta = 0.25

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())


def plot_histogram(X):
    for i in range(4):
        plt.clf()
        plt.close()
        plt.hist(X[:, i], bins=100)
        plt.gca().set(title='feature number ' + str(i) + ' Histogram', ylabel='Frequency')
        plt.savefig("histogram of feature num" + str(i) + ".png")


def MLPclassification(X_train, y_train, X_validation, y_validation):
    # clf = MLPClassifier(max_iter=250, verbose=2, batch_size=64).fit(X_train, y_train)
    # clf = RandomForestClassifier(n_estimators=1000, max_depth=7, verbose=1).fit(X_train, y_train)
    clf = svm.SVC(C=0.01, kernel='linear', probability=True).fit(X_train, y_train)

    # you can change the metric if you want
    # y_train_pred = clf.predict(X_train)
    # y_val_pred = clf.predict(X_validation)

    y_train_pred = np.where(clf.predict_proba(X_train)[:, 1] > 0.58, 1, 0)
    y_val_pred = np.where(clf.predict_proba(X_validation)[:, 1] > 0.58, 1, 0)

    print('Train accuracy', accuracy_score(y_train, y_train_pred))
    print('Validation accuracy', accuracy_score(y_validation, y_val_pred))

    print('Train precision', precision_score(y_train, y_train_pred))
    print('Validation precision', precision_score(y_validation, y_val_pred))

    print('Train recall', recall_score(y_train, y_train_pred))
    print('Validation recall', recall_score(y_validation, y_val_pred))

    print('Train f-beta score', fbeta_score(y_train, y_train_pred, beta=0.25))
    print('Validation f-beta score', fbeta_score(y_validation, y_val_pred, beta=0.25))

    return clf


X_train, y_train, X_validation, y_validation, X_test = load_process_data()

X_train_mean = np.mean(np.stack(np.split(X_train, 30, 1), 1), axis=1)
X_validation_mean = np.mean(np.stack(np.split(X_validation, 30, 1), 1), axis=1)
X_test_mean = np.mean(np.stack(np.split(X_test, 30, 1), 1), axis=1)

X_train_std = np.std(np.stack(np.split(X_train, 30, 1), 1), axis=1)
X_validation_std = np.std(np.stack(np.split(X_validation, 30, 1), 1), axis=1)
X_test_std = np.std(np.stack(np.split(X_test, 30, 1), 1), axis=1)

X_train_max = np.max(np.stack(np.split(X_train, 30, 1), 1), axis=1)
X_validation_max = np.max(np.stack(np.split(X_validation, 30, 1), 1), axis=1)
X_test_max = np.max(np.stack(np.split(X_test, 30, 1), 1), axis=1)

X_train_min = np.min(np.stack(np.split(X_train, 30, 1), 1), axis=1)
X_validation_min = np.min(np.stack(np.split(X_validation, 30, 1), 1), axis=1)
X_test_min = np.min(np.stack(np.split(X_test, 30, 1), 1), axis=1)

X_train = np.concatenate((X_train_mean, X_train_std, X_train_min, X_train_max), axis=1)
X_validation = np.concatenate((X_validation_mean, X_validation_std, X_validation_min, X_validation_max), axis=1)
X_test = np.concatenate((X_test_mean, X_test_std, X_test_min, X_test_max), axis=1)

'''
df=pd.DataFrame(X_train[:,-4:])
df['target'] = y_train.astype(str)
sns.pairplot(df,hue = 'target', diag_kind= 'hist',
             vars=df.columns[:-1],
             plot_kws=dict(alpha=0.5),
             diag_kws=dict(alpha=0.5))
plt.savefig("pariplot_map")
'''
# plot_histogram(X_train)


'''
#drop all odd/even feature
features_idx = [i for i in range(121) if i%2==1]
X_train = np.delete(X_train, features_idx, axis=1)
X_validation = np.delete(X_validation, features_idx, axis=1)
'''

clf = MLPclassification(X_train, y_train, X_validation, y_validation)
y_test_pred = clf.predict(X_test)
# np.savetxt('203768460_204380992_4.txt', y_test_pred.astype(int), fmt='%i', delimiter='\n')


# move to tensor #samples, #time_stamp, #fetaures per time stamp
# X_train = np.stack(np.split(X_train, 30, 1), 1)
# X_validation = np.stack(np.split(X_validation, 30, 1), 1)

'''
#keras LSTM

# design network
model = Sequential()
model.add(LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True,  dropout=0.2, recurrent_dropout=0.2))
#model.add(LSTM(units=30, return_sequences=True))
model.add(ReLU())
model.add(Dropout(0.3))
model.add(LSTM(units=50))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy', fbeta_keras])
# fit network
history = model.fit(X_train, y_train, epochs=100, batch_size=72, validation_data=(X_validation, y_validation), verbose=2, shuffle=True)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

y_val_pred = model.predict(X_validation)
scores = model.evaluate(X_validation, y_validation, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print('Validation beta score', fbeta_score(y_validation, y_val_pred, beta=0.25))
'''

# pytorch LSTM
'''
X = X_train
y = np.array(y_train)

n_features = X_train.shape[2] # this is number of parallel inputs
n_timesteps = X_train.shape[1] # this is number of timesteps

# convert dataset into input/output
print(X.shape, y.shape)

# create NN
mv_net = MV_LSTM(n_features,n_timesteps)
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-1)

train_episodes = 500
batch_size = 16

mv_net.train()
for t in range(train_episodes):
    y_pred = []
    for b in range(0,len(X),batch_size):
        inpt = X[b:b+batch_size,:,:]
        target = y[b:b+batch_size]

        x_batch = torch.tensor(inpt,dtype=torch.float32)
        y_batch = torch.tensor(target,dtype=torch.float32)

        mv_net.init_hidden(x_batch.size(0))
    #    lstm_out, _ = mv_net.l_lstm(x_batch,nnet.hidden)
    #    lstm_out.contiguous().view(x_batch.size(0),-1)
        output = mv_net(x_batch)
        y_pred = y_pred + [np.argmax(i) for i in output.data.numpy()]
        loss = criterion(output.view(-1), y_batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    print('step : ' , t , 'loss : ' , loss.item(), 'Validation beta score', fbeta_score(y_train, np.array(y_pred), beta=0.25))
'''