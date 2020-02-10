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
import matplotlib.pyplot as plt
from matplotlib.pyplot import pie, axis, show
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, validation_curve, KFold
from sklearn.model_selection import RandomizedSearchCV, cross_validate, ParameterGrid
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, make_scorer, roc_curve, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from datetime import datetime

from IPython.display import display
pd.options.display.max_columns = 25

import xgboost as xgb
from xgboost import XGBClassifier, XGBRFClassifier
from xgboost import plot_importance
import time

#os.getcwd()


def encoding_cat_noise(xtrain_orig, xtest_orig, target= 'target', verbose= False, minimum_std= 1, n_folds= 10, 
                              prop_std_per_noise= 0.05, threshold= 10, showPrint= False, cols_to_encode= None):
    
    """return diz= with 'X_train', 'X_test', 'ytrain', 'ytest' """
    
    label= target  #naming convention
    
    xtrain= xtrain_orig.copy()
    xtest= xtest_orig.copy()
    ytrain= xtrain[label]  
    ytest=  xtest[label]   
    
    print('Train and Test initial shapes (including target): ', xtrain.shape, xtest.shape)
    print('Threshold is: ', threshold)
    #purch_doc_train= xtrain['Purchase document']
    #purch_doc_test= xtest['Purchase document']
    #xtrain= xtrain.drop('Purchase document', axis=1)
    #xtest= xtest.drop('Purchase document', axis=1)
    
    
    #Faccio Encoding di tutte quelle di tipo Oggetto, cioè stringe
    cardinalita_Xtrain={}
    for col in xtrain.select_dtypes(include=[object]).columns:
        cardinalita_Xtrain[col]= xtrain[col].nunique()
    cardinalita_Xtrain= pd.Series(cardinalita_Xtrain)
    cardinalita_Xtrain= cardinalita_Xtrain.sort_values(ascending=False)
    cardinalita_Xtrain= pd.DataFrame(cardinalita_Xtrain, columns= ['cardinalità_classe'])
    cardinalita_Xtrain


    columns_to_encode= list(cardinalita_Xtrain.index)
    if cols_to_encode is not None:  #le cambio solo se glielo specifico, sennò lo faccio per tutti i dtypes= object
        columns_to_encode= cols_to_encode
    
    print('...Performing Encoding...')
    time_init= time.time()
    for col in columns_to_encode:
        if showPrint== True:
            print('\n COLUMN NAME: ', col)  
        targetc = KFoldTargetEncoderTrain_std(col, label ,n_fold= n_folds, verbosity= verbose,threshold = threshold, showPrint= False)
        xtrain = targetc.fit_transform(xtrain)
        #print(xtrain.iloc[:2, 17:], '\n')

        test_targetc = KFoldTargetEncoderTest_std(xtrain,col, col+ '_enc_mean', col+ '_enc_std')
        xtest= test_targetc.fit_transform(xtest)
        
        xtrain= xtrain.drop(col, axis=1)
        xtest= xtest.drop(col, axis=1)
    print('...Encoding Terminated...')
    print('execution time (minutes) for encoding=', round((time.time()-time_init)/60, 1))
        
    xtrain= xtrain.drop( [label], axis=1)
    xtest= xtest.drop([label], axis=1)
             
    #### Metto minimum_std dev a chi ha 0, così poi ci sarà un po' di Noise pure in corrispondenza di quei samples
    cols_std= [col for col in xtrain.columns if col.split('_')[-1]== 'std']
    
    for col in cols_std:
        xtrain[col]= xtrain[col].apply(lambda x: minimum_std if x==0 else x)
        xtest[col]= xtest[col].apply(lambda x: minimum_std if x==0 else x)
        
    #for col in cols_std:
        #print('Ci sono {} std = 0 per colonna {}'.format(xtrain[col][xtrain[col]== 0].sum(), col))
    
    ###### Final Encoding con Noise #########
    cols_mean= [col for col in xtrain.columns if col.split('_')[-1]== 'mean']
    diz_mean_std= dict(zip(cols_mean, [col[: len(col)- len('mean')] +'std' for col in cols_mean] ))
    
    for key in diz_mean_std.keys():  #key è la colonna con la media
        #if showPrint== True:
            #print(' key and value are: ', key, diz_mean_std[key] ) 
        for X in [xtrain, xtest]:   # Creo colonna con encoding finale e droppo le due di mean e std encoding
            ### Metto Noise sono nel Training. Nel test solo medie.
            if X is xtrain:   
                X[key[: - len('_enc_mean')] + '_FINAL_ENC' ] = np.random.normal(X[key], prop_std_per_noise* X[diz_mean_std[key]], len(X))
            elif X is xtest:
                X[key[: - len('_enc_mean')] + '_FINAL_ENC' ]= X[key]  #key è la colonna con la media
            X.drop([key, diz_mean_std[key]], axis=1, inplace= True)
                
    ##### Faccio dei Check #####
    if len(xtrain.select_dtypes(include=object).columns) ==0 and len(xtest.select_dtypes(include=object).columns) ==0:
        print('\n \033[1mCORRETTO: Tutte colonne sono numeriche\033[0m')
    else:
        print('\n \033[1mSBAGLIATO: n° colonne non numeriche non è 0, ma ', max(len(xtrain.select_dtypes(include=object).columns),  
                                                                                   len(xtest.select_dtypes(include=object).columns)))
        
        
    ### Ultimo check: colonne Train e Test devono essere uguali
    if list(xtrain.columns) != list(xtest.columns): 
        print('ERRORE: COLONNE IN TRAIN E TEST SONO DIVERSE')
        
    #Gli rimetto Purchase Document, solo per andare in Join dopo
    #xtrain.insert (0, "Purchase document", purch_doc_train)
    #xtest.insert(0, "Purchase document", purch_doc_test)
    
    #numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    for x in [xtrain, xtest]:
        for col in x.select_dtypes(include= 'float64'):  #include='numerics'
            x[col]= x[col].apply(lambda x: round(x, 3))
        
    print('\n Train and Test final shapes: ', xtrain.shape, xtest.shape)
        
    return {'X_train': xtrain, 'X_test': xtest, 'ytrain': ytrain, 'ytest': ytest}





encoded_train_test = encoding_cat_noise(pre_sel_train, pre_sel_test, target= 'target', verbose= False, minimum_std= 0.05, n_folds= 10, 
                              prop_std_per_noise= 0.001, threshold= 5, showPrint= True, cols_to_encode= None)





def plot_feat_imp_adj(best_est, X_train= scaled_train, X_test= scaled_test, 
                  n_feat_to_plot= 10, color='r', figsize= True, show= True):
    """RETURN DF CON FEATURE IMPORTANCES ORDINATE"""
    
    feats_imp= pd.DataFrame(pd.Series(dict(zip(X_train.columns, best_est.feature_importances_))), columns= ['importance'])
    feats_imp= feats_imp.sort_values(by= 'importance', ascending=False)
    
    if show== False:
        return feats_imp
    
    if figsize==True:
        feats_imp.iloc[: n_feat_to_plot]['importance'].plot.barh(color= color, edgecolor='k',figsize= (12,7),
                                      linewidth=2)
    else:
        feats_imp.iloc[: n_feat_to_plot]['importance'].plot.barh(color= color, edgecolor='k', linewidth=2)

    #plt.figure(figsize=(26,18))
    ax = plt.gca()
    ax.invert_yaxis()
    plt.xticks(size=20) 
    plt.yticks(size=18)

    plt.title('Most Important Features for Random Forest', size=28);
    #plt.savefig('Feature Importance RF2.png', format= 'png')
    
    return feats_imp





def run_model_no_val(estimator, param_grid= None, X_train= preliminary_train, X_test= preliminary_test, 
                     only_imp= False, print_model= True, catb= False):
    
        
    if only_imp== True:
        try:
            X_train= X_train_imp
            X_test= X_test_imp
        except:
            pass
        
    """if catb== True:
        X_train= train_top
        X_test= test_top"""
    
    print('Shape di X_train e X_test sono: ', X_train.shape, X_test.shape)
    temp=time.time()
    
    "The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))"
    
    
    
    if param_grid is None:
        
        if catb== True:
            estimator.fit(X_train, ytrain, cat_features= X_train.select_dtypes(include= object).columns)
        else:
            estimator.fit(X_train, ytrain, )
            
        preds= estimator.predict(X_test)
        print('execution time (min)=', round((time.time()-temp)/60, 1), ' finished at ', datetime.today())
        summary_classifier(estimator, X_train, ytrain, ytest, preds)
        return estimator
    
    else:
        #print(X_train.columns)
        models= {}
        iteration= 1

        for diz_params in list(ParameterGrid(param_grid)):
            print('\n\t \033[1mITERATION {}/{} \033[0m'.format(iteration, len(list(ParameterGrid(param_grid)))))
            print('\t Current Grid of Parameters is {}'.format(diz_params))     
            
            if catb== True:
                est= cb.CatBoostClassifier()
                est.set_params(**diz_params)
                est.fit(X_train, ytrain, cat_features= X_train.select_dtypes(include= object).columns)
            else:
                est= estimator.set_params(**diz_params)
                est.fit(X_train, ytrain )

            pred_train= est.predict_proba(X_train)
            pred_test= est.predict_proba(X_test)

            roc_train= float(round(roc_auc_score(ytrain, pred_train[:, 1]),4))
            roc_test=  float(round(roc_auc_score(ytest, pred_test[:, 1]),4))
            print('\n\033[1mROC Train is {a}, ROC Test is {b}\033[0m'.format(a= roc_train, 
                                                                                b= roc_test))

            degree_overfitting = float((roc_train-roc_test)*100)

            models[iteration] = {}
            
            if print_model== True:
                print(est)
            
            models[iteration]['diz_params']= diz_params
            models[iteration]['degree_overfitting(%)']= degree_overfitting
            models[iteration]['roc_train']= roc_train
            models[iteration]['roc_test']= roc_test
            #models[iteration]['est']= est
            
            iteration += 1
            del est
    
    models= pd.DataFrame(models).T
    
    for col in models.columns:
        try:
            models[col]=  models[col].astype(float)
        except:
            pass
    
    print('execution time (min)=', round((time.time()-temp)/60, 1), ' finished at ', datetime.today())
    return models.sort_values(by= 'roc_test', ascending= False)


def run_model(estimator, param_grid= None, X_train= preliminary_train, X_test= preliminary_test, only_imp= False, random= False, verbose=5, cv=3, n_jobs= None, comparison= False): #param_grid= None
    """estimator è il modello, task puo essere: ['class', 'reg']. \n
    Return Grid Search estimator. """
        
    if only_imp== True:
        try:
            X_train= X_train_imp
            X_test= X_test_imp
        except:
            pass
    
    print('Shape di X_train e X_test sono: ', X_train.shape, X_test.shape)
    temp=time.time()
    
    "The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))"
    
    if param_grid is None:
        estimator.fit(X_train, ytrain, )
        preds= estimator.predict(X_test)
        print('execution time (min)=', round((time.time()-temp)/60, 1), ' finished at ', datetime.today())
        summary_classifier(estimator, X_train, ytrain, ytest, preds)
        return estimator
    
    else:
        if random== False:
            Grid_RF = GridSearchCV(estimator, param_grid, scoring='roc_auc', cv=cv, return_train_score=True, 
                                   n_jobs= n_jobs, verbose= verbose)
            Grid_RF.fit(X_train, ytrain)
            print('\n It is not a single model but a Grid Search \n')
            summary_grid(Grid_RF, X_train, ytrain, X_test, ytest)  
            print('execution time (min)=', round((time.time()-temp)/60, 1), ' finished at ', datetime.today())
        
        elif random== True:
            Grid_RF = RandomizedSearchCV(estimator, param_distributions=param_grid, 
                                         scoring='roc_auc', cv=cv, return_train_score=True, verbose= verbose, n_jobs= n_jobs)
            Grid_RF.fit(X_train, ytrain)
            print('\n It is not a single model but a (Randomized) Grid Search \n')
            summary_grid(Grid_RF, X_train, ytrain, X_test, ytest)
            print('execution time (min)=', round((time.time()-temp)/60, 1), ' finished at ', datetime.today())
        return  Grid_RF


def df_cv_results (grid_fit, n_to_show= 5, rename= True):
    """ MAIN CV RESULTS """
    df= pd.DataFrame(grid_fit.cv_results_)
    df= df[[col for col in df.columns if 'param_' in col]+['mean_test_score', 'mean_train_score']]
    if rename:
        preliminary_train.rename({'mean_test_score': 'Test_AUC', 'mean_train_score': 'Train_AUC'}, axis=1)
        
    return df.sort_values(by= 'mean_test_score', ascending= False).head(n_to_show)

   

def summary_classifier(classifier, x_train, y_train, y_test, predictions):
    
    #posso predire scaled_test, quello con tutte le 369 feauters: lui da solo ne toglierà alcune con lasso, è gia incorporato
    print('\n\033[1mtrain accuracy TP+TN/tot is {} \033[0m'.format(round(accuracy_score(y_train, classifier.predict(x_train)),3)))
    print('\033[1mtest accuracy TP+TN/tot is {} \033[0m'.format(round(accuracy_score(y_test, predictions),3)))
    #print("cross validation accuracy sul train è: ", cross_val_score(estimator = grid_fit.best_estimator_, X = scaled_train, y = Y_train, cv = 10).mean())
    print('\n\033[1mCLASSIFICATION REPORT\033[0m: \n', classification_report(y_test, predictions))
    print('\033[1mCONFUSION MATRIX\033[0m \n', confusion_matrix(y_test, predictions))
    """ROC CURVE SCORES (non esiste per multiclass classification)"""
    print('\n\033[1mROC (Area under curve) Train is {a}, for ROC test is {b}\033[0m'.format(a= round(roc_auc_score(y_train, classifier.predict_proba(x_train)[:, 1]),3), 
                                                                                            b= round(roc_auc_score(y_test, classifier.predict_proba(x_test)[:, 1]),3)))
    print('\n\033[1mF1 train is {a}, F1 test is {b} \033[0m'.format(a= round(f1_score(y_train, classifier.predict(x_train), 
                                                                average='weighted'),3), b= round(f1_score(y_test, predictions, average='weighted'),3)))
    
def summary_grid(grid_fit, scaled_train, Y_train, scaled_test, Y_test):
    """ AFTER A GRIDSEARCH, I CONSIDER RELEVANT ANALYZING THE FOLLOWING SCORES
    NB: Per accuracy il Train è tutto l'80%, senza split con validation, e test sul 20%. Per F1-weighted invece validation score, l ultimo,
    è valutato come media dello score sulle K folds (5) del train, quindi su 80%/5..."""
    
    print('best param combination: ', grid_fit.best_params_)   #'C': 0.357
    #print('best estimator: ', grid_fit.best_estimator_)
    #print('predictions', grid_fit.predict(scaled_test))
    
    #posso predire scaled_test, quello con tutte le 369 feauters: lui da solo ne toglierà alcune con lasso, è gia incorporato
    print('\n\033[1m Train Accuracy is {}\033[0m'.format(round(accuracy_score(Y_train, grid_fit.predict(scaled_train)),3)))
    print('\033[1m Test Accuracy is {}\033[0m'.format(round(accuracy_score(Y_test, grid_fit.predict(scaled_test)),3)), '\033[0m')
    #print("\033[1mcross validation accuracy sul train è: ", round(cross_val_score(estimator = grid_fit.best_estimator_, X = scaled_train, y = Y_train, cv = 5).mean(),3), '\033[0m')
    print('\n\033[1mCLASSIFICATION REPORT\033[0m: \n', classification_report(Y_test, grid_fit.best_estimator_.predict(scaled_test)))
    print('\033[1mCONFUSION MATRIX\033[0m \n', confusion_matrix(Y_test, grid_fit.best_estimator_.predict(scaled_test)))
    """ROC CURVE SCORES (non esiste per multiclass classification)"""
    print('\n\033[1mROC train is {a}, \n\033[1mROC test is {c} \033[0m'.format(a= round(roc_auc_score(Y_train, grid_fit.predict_proba(scaled_train)[:, 1]),3), 
                                                                            c= round(roc_auc_score(Y_test, grid_fit.predict_proba(scaled_test)[:, 1]),3)))
    print('\n\033[1mF1 TRAIN: {a}, \n\033[1mF1 TEST: {b}\033[0m'.format(
        a= round(f1_score(Y_train, grid_fit.predict(scaled_train), average='weighted' ),3), 
        b= round(f1_score(Y_test, grid_fit.predict(scaled_test), average='weighted'),3)))  
    
    
    
    
    
    

