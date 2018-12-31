import tensorflow as tf
import random
import tensorflow_hub as hub
from sklearn.base import BaseEstimator, ClassifierMixin
import os
import numpy as np
import pandas as pd



class TFHUBCLASSIFIER(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            hidden_units=None,
            text_feature_key=None,
            num_classes=None,
            num_epochs=None,
            learning_rate=None,
            batch_size=None,
            training_steps=None,
            trainable=None,
            batch_norm=None,
            dropout=None,
            activation_fn=None,
            l1_regularization_strength=None,
            l2_regularization_strength=None,
            shuffle=None,
            tf_hub_module=None
    ):
        # embedding params
        self.trainable = trainable
        self.text_feature_key = text_feature_key

        # DNN classifier params
        self.hidden_units = hidden_units
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.training_steps = training_steps
        self.l1_regularization_strength = l1_regularization_strength
        self.l2_regularization_strength = l2_regularization_strength
        self.tf_hub_module = tf_hub_module

    def fit(self, X, y=None):

        embedded_text_feature_column = hub.text_embedding_column(key=self.text_feature_key, module_spec=self.tf_hub_module, trainable=self.trainable)

        # train_input_fn = tf.estimator.inputs.pandas_input_fn(dfs['description'].to_frame(), dfs["label"], batch_size=1200, num_epochs=None, shuffle=False)
        train_input_fn = tf.estimator.inputs.pandas_input_fn(
            X, y, batch_size=self.batch_size, num_epochs=None, shuffle=True)

        optimizer = tf.train.ProximalAdagradOptimizer(
            learning_rate=self.learning_rate,
            l1_regularization_strength=self.l1_regularization_strength,
            l2_regularization_strength=self.l2_regularization_strength
        )
	activation = {
                  'relu':     tf.nn.relu,
                  'tanh':     tf.nn.tanh,
                  'sigmoid':  tf.nn.sigmoid,
                  'elu':      tf.nn.elu,
                  'softplus': tf.nn.softplus,
                  'softsign': tf.nn.softsign,
                  'relu6':    tf.nn.relu6
        }
	path = ''.join(random.choice('0123456789ABCDEF') for i in range(16))
	path = '../' + path
	self.model_dir = path
	os.mkdir(self.model_dir)
	print self.model_dir 
	#run_config = tf.estimator.RunConfig(save_summary_steps=None, save_checkpoints_secs=None)
        self.estimator = tf.estimator.DNNClassifier(
            hidden_units=self.hidden_units,
            feature_columns=[embedded_text_feature_column],
            n_classes=self.num_classes,
            optimizer=optimizer,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            activation_fn=activation[self.activation_fn],
	    #config=run_config,
	    model_dir=self.model_dir
        )
	path = ''.join(random.choice('0123456789ABCDEF') for i in range(16))
	self.eval_path = './' + path
	os.mkdir(self.eval_path)
	print self.eval_path
	early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
	    self.estimator,
	    metric_name='loss',
	    max_steps_without_decrease=400,
	    min_steps=100,
	    eval_dir=self.eval_path)

        self.estimator.train(input_fn=train_input_fn, steps=self.training_steps)
	tf.reset_default_graph()
	return self

    def predict(self, X):
        print 'calling predict'
	l = self.estimator.predict(tf.estimator.inputs.pandas_input_fn(X, y=None, num_epochs=1, shuffle=False))
	res = []
	for i in l:
	    res.append(int(i['classes'][0]))
	print pd.DataFrame({'res': res}).res.value_counts()
	npres = np.array(res)
	return npres


def trn():
        import tensorflow as tf
        import tensorflow_hub as hub
        from sklearn.base import BaseEstimator, ClassifierMixin
        import numpy as np
        import pandas as pd
        from sklearn import preprocessing
        import os
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        #from classifier_desc.tfhclf import TFHUBCLASSIFIER
        from sklearn.model_selection import train_test_split, StratifiedKFold
        from sklearn.metrics import f1_score, make_scorer

        df = pd.read_csv('/data/kdata/desc/random_timespan_fsn_descriptions_training_raw_set.csv')
        le = preprocessing.LabelEncoder()
        le.fit(df.vertical.tolist())
        df['label'] = df.vertical.apply(lambda x: le.transform([x])[0])
        dfs = df.sample(frac=1)
        X_train, X_test, y_train, y_test = train_test_split(dfs['description'], dfs['label'], test_size=0.2, random_state=7656, stratify=dfs['label'].values)
        os.environ['TFHUB_CACHE_DIR']="/data/kdata/tfh"

        param_grid = {
            'activation_fn': ['relu', 'tanh', 'sigmoid', 'softsign'],
            'batch_norm': [True],
            'batch_size': [2000, 2800, 3600, 4400],
            'dropout': [0.2, 0.3, 0.4],
            'hidden_units': [[500, 100], [500]],
            'l1_regularization_strength': [0.0, 0.01, 0.06],
            'l2_regularization_strength': [0.0, 0.01, 0.06],
            'learning_rate': [0.1, 0.3, 0.5],
            'shuffle': [True],
            'tf_hub_module': ["https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1"],
            'trainable': [True],
            'training_steps': [4000, 6000]
        }
        thf_clf = TFHUBCLASSIFIER(
            num_classes = 348,
            text_feature_key = "description"
        )
        scorer = make_scorer(f1_score, average='micro')
        n_iter_search = 5
        random_search = RandomizedSearchCV(thf_clf, param_distributions=param_grid,random_state=108,
                                           n_iter=n_iter_search, cv=5, verbose=10, scoring=scorer)

        random_search.fit(X_train.to_frame(), y_train.to_frame()['label'])


'''
    thf_clf = TFHUBCLASSIFIER(
            num_classes = 348,
            text_feature_key = "description",
            num_epochs=1,
            learning_rate=0.1,
            batch_size=200,
            training_steps=200,
            trainable=True,
            batch_norm=True,
            dropout=0.3,
            activation_fn='relu',
            l1_regularization_strength=0.01,
            l2_regularization_strength=0.1,
            hidden_units=[100],
            shuffle=True,
            tf_hub_module='https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1'
         
        )
    thf_clf.fit(X_train.to_frame(), y_train.to_frame()['label'])
    return thf_clf
'''
