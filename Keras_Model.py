import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings("ignore")


import pandas as pd
import numpy as np
from sklearn  import preprocessing, decomposition, base
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso, SGDClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, validation_curve, KFold
from sklearn.model_selection import RandomizedSearchCV, cross_validate, ParameterGrid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,VotingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, make_scorer
from sklearn.metrics import roc_curve, f1_score, precision_score, recall_score, auc, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.formula.api as sm
from sklearn.svm import SVC

import xgboost as xgb
from xgboost import XGBClassifier, XGBRFClassifier
from xgboost import plot_importance
import time

import copy 
import pickle
from IPython.display import display
from matplotlib import interactive 
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
pd.options.display.max_columns = 40

from datetime import*
import time



from sklearn import datasets, linear_model
from keras import regularizers
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

import copy
from keras.layers import *   #è incluso Flatten
from keras.models import *
from keras.layers import Concatenate, concatenate

#from Target_enc_class import *


def plot_model_history(history, measures = ["acc"]):

  plt.style.use("ggplot")
  (fig, ax) = plt.subplots(1, 3, figsize=(20, 5))

  # loop over the accuracy names
  for (i, l) in enumerate(measures):
    # plot the loss for both the training and validation data
    ax[i].set_xlabel("Epoch #")
    if l!= 'loss':
      ax[i].set_title("{}".format(l ))
      ax[i].set_ylabel("{}".format(l))
    else:
      ax[i].set_title("Loss")
      ax[i].set_ylabel("Loss")
    #ax[i].plot(np.arange(0, epochs), history.history[l], label=l)
    #ax[i].plot(np.arange(0, epochs), history.history["val_" + l], label="val_" + l)
    ax[i].plot(history.history[l], label=l)
    ax[i].plot( history.history["val_" + l], label="val_" + l)
    ax[i].legend()

  plt.tight_layout() 
  plt.show()
  plt.close()
  
  
  
def run_sequential_model(estimator, final_inputs_train, final_inputs_test, param_grid= None, random= False, verbose=2, cv=3, n_jobs= None,
               return_single_mod= True, return_results=True, n_to_show=5, early= None, class_weight=  class_weights, plot_metric= ['acc'],
              epochs= epochs, batch_size= batch_size):
    """estimator è il modello, caso puo essere: ['statico', 'rolling75', 'rolling25']. \n
    Return Grid Search estimator. """
    
    temp=time.time()

    ytrain= ytrain_binary
    ytest= ytest_binary
    X_train= final_inputs_train
    X_test= final_inputs_test

    if param_grid is None:
      history= estimator.fit(final_inputs_train, ytrain_binary, validation_split= 0.2, callbacks= [early], class_weight=  class_weights, epochs= epochs, batch_size= batch_size)
      
      plot_model_history(history, measures= plot_metric)
      preds= estimator.predict_classes(final_inputs_test)
      diz_summary= summary_classifier(estimator, final_inputs_train, ytrain, ytest, preds)
      print('CON EVALUATE')
      print('Performance on Train: ', history.model.evaluate(X_train, ytrain))
      print('Performance on Test: ', history.model.evaluate(X_test, ytest))
      print('execution time (min)=', round((time.time()-temp)/60, 1), ' finished at ', datetime.today())
      if return_single_mod==True:
          return [history, diz_summary]
    
    else:
      ################## Divido Tra Randomized e non ##################
      if random== False:
        print('\n It is not a single model but a Grid Search \n')
        Grid_RF = GridSearchCV(estimator, param_grid, scoring='f1_weighted', cv=cv, return_train_score=True, 
                                n_jobs= n_jobs, verbose= verbose)
        

      else:
        print('\n It is not a single model but a (Randomized) Grid Search \n')
        Grid_RF = RandomizedSearchCV(estimator, param_distributions=param_grid, 
                                      scoring='f1_weighted', cv=cv, return_train_score=True, verbose= verbose, n_jobs= n_jobs)

      ################ Faccio il FIT ##################
      if early is None:
        Grid_RF.fit(X_train, ytrain, class_weight=  class_weights)
      else:
        Grid_RF.fit(X_train, ytrain, callbacks= [early], class_weight=  class_weights)
      
      ##################### statistiche ##################
        
      summary_grid(Grid_RF, X_train, ytrain, X_test, ytest)  
      print('execution time (min)=', round((time.time()-temp)/60, 1), ' finished at ', datetime.today())
      if return_results==True:
        print('best_results')
        print(df_cv_results(Grid_RF, n_to_show))
        
      return  Grid_RF 

    
    
    
def NN(nodi_hiddens= [64,32,16,8], with_dropout=True , dropout= 0.5, lr= 0.00001, n_layers= 1,
                loss_func= 'binary_crossentropy', metrics= ['accuracy', f1]):   #loss_func= 'mean_square_error', metrics= ['mse', 'mae', 'mape']

  #diz_pre_model= create_embedded_inputs(X_train, X_test, max_emb_size=50, method= 'half')
  model = Sequential()
  print('\n\033[1m Params modello: n_layers: {}, nodi_hiddens: {}, lr: {} \033[0m'.format(n_layers, nodi_hiddens, lr))
  for i in range (n_layers):
    if i==0:
      model.add(Dense(nodi_hiddens[0], input_dim=len(container_class['X_train'].columns), activation='relu'))
      if with_dropout== True:
        model.add(Dropout(rate= dropout))
    else:
      model.add(Dense(nodi_hiddens[i], activation='relu'))
      if with_dropout==True:
        model.add(Dropout(rate= dropout))

  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss= loss_func, optimizer= optimizers.Adam(lr= lr), metrics= metrics)  #fbeta_score(y_true, y_pred, beta=1)

  return model




#################################### EXAMPLE #####################################################################
early = EarlyStopping(monitor='loss', mode='min', verbose=5, patience=5, min_delta= 0.5/100)  # Voglio 0.1% di miglioramento in 7 iterazioni
nn_base_one = KerasClassifier(build_fn=NN, epochs=10, batch_size=256, verbose=2)

param_grid = dict(
        n_layers= [1, 2],
        epochs= [40],
        batch_size= [32, 256],
        dropout= [0.5],
        lr= [1e-03] )  #default

history= run_model(nn_base_one, container_class['X_train'], container_class['X_test'],  param_grid= param_grid, early= early, class_weight=  class_weights, cv=2, verbose=5)

#################################### END EXAMPLE #####################################################################


