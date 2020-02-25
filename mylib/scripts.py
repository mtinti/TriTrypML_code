#import libraries
from itertools import islice

#system
import warnings
import os
import sys
import re

#data handling
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm_notebook

#visualisation
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import missingno as msno
import matplotlib
from IPython.display import Image
from adjustText import adjust_text

#machine learning
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import eli5
from eli5.sklearn import PermutationImportance
import lightgbm as lgb
from boruta import BorutaPy
import lime
from lime.lime_tabular import LimeTabularExplainer

#biological sequence analysis
from Bio import SeqIO

#garbage collector
import gc
#sys.path.append(os.path.realpath(__file__))
#import make_roc_cv
warnings.filterwarnings("ignore")

#split iterable in even chunk size
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

#helper function to visualize the correlation between experiments
def plot_correlation(df, figname='corr_prot'):
    #function to annotate the axes with
    #the pearson correlation coefficent
    def corrfunc(x, y, **kws):
        corr = np.corrcoef(x, y)
        r = corr[0][1]
        ax = plt.gca()
        ax.annotate("p = {:.2f}".format(r),
                    xy=(.1, .9), xycoords=ax.transAxes)
    
    #prepare the seaborn grid and plot
    g = sns.PairGrid(df.dropna(), palette=["red"], height=1.8, aspect=1.5)
    g.map_upper(plt.scatter, s=5)
    g.map_diag(sns.distplot, kde=False)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_lower(corrfunc)
    sns.set(font_scale=1.1)


#helper function to revert a protein ids to gene ids
#used with TryTripDB identifiers Treu927
def clean_id(temp_id):
    temp_id = temp_id.split(':')[0]
    if temp_id.count('.')>=3:
        temp_id = '.'.join(temp_id.split('.')[0:-1])
    return temp_id


#convinent function to decrese the size of the dataframe
#it finds the smallest data typa that can accomodate 
#the values of each columns
#cretits:
#https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

def reduce_mem_usage(df, skip_cols_pattern='cat_'):
    """ 
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in tqdm_notebook(df.columns):

        if skip_cols_pattern in col:
            print(f"don't optimize index {col}")

        else:
            col_type = df[col].dtype

            if col_type != object:

                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        #df[col] = df[col].astype(np.float16)
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            else:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

#count how many comment lines
#are present in a file
#used to select the starting point to assemble
#a pandas dataframe
def count_comment_line(infile='', tag='##'):
    a=0
    for l in open(infile):
        if l.startswith(tag):
            a+=1
        else:
            break
    return a

#transform the attributes of a gff file
#in a dictionary
def get_attributes(att=''):
    attributes = {}
    for n in att.split(';'):
        attributes[n.split('=')[0]]=n.split('=')[1]
    return attributes
 
#extract the chromosome size
def get_source_size(infile=''):
    a=0
    size_dictionary = {}
    for l in open(infile):
        if l.startswith('##sequence-region'):
            source_id = l.split(' ')[1]
            length = int(l.split(' ')[-1].strip())
            size_dictionary[source_id]=length
        else:
            a+=1
        if a>=3:
            break
    return size_dictionary



def make_scatter_matrix(in_df):
    sns.set(font_scale = 1)
    #sns.set(style="white")
    def corrfunc(x, y, **kws):
        r, _ = stats.pearsonr(x, y)
        ax = plt.gca()
        ax.annotate("r = {:.3f}".format(r),
                    xy=(.1, .9), xycoords=ax.transAxes)
    
    g = sns.PairGrid(in_df, palette=["red"]) 
    g.map_upper(plt.scatter, s=10)
    g.map_diag(sns.distplot, kde=False) 
    g.map_lower(sns.kdeplot, cmap="Blues_d") 
    g.map_lower(corrfunc)
    plt.show()


def make_pca(in_df, palette, top=500):
    cols = in_df.columns
    pca = PCA(n_components=2)
    
    sorted_mean = in_df.mean(axis=1).sort_values()
    select = sorted_mean.tail(top)
    #print(top)
    in_df = in_df.loc[select.index.values]
    pca.fit(in_df)
    temp_df = pd.DataFrame()
    temp_df['pc_1']=pca.components_[0]
    temp_df['pc_2']=pca.components_[1]
    temp_df.index = cols
    print(pca.explained_variance_ratio_)
    temp_df['color']=palette
    fig,ax=plt.subplots(figsize=(12,6))
    temp_df.plot(kind='scatter',x='pc_1',y='pc_2',s=30, c=temp_df['color'], ax=ax)
    #print(temp_df.index.values)
       
    texts = [plt.text(temp_df.iloc[i]['pc_1'], 
                       temp_df.iloc[i]['pc_2'],
                       cols[i])
                       for i in range(temp_df.shape[0])]
    
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    ax.set_title('PCA',size=14)
    ax.set_xlabel('PC1_{:.3f}'.format(pca.explained_variance_ratio_[0]),size=12)
    ax.set_ylabel('PC2_{:.3f}'.format(pca.explained_variance_ratio_[1]),size=12)
    
    ax.yaxis.label.set_size(12)
    ax.xaxis.label.set_size(12)
    plt.show()
    
def make_mds(in_df, palette, top=500):
    cols = in_df.columns
    pca = MDS(n_components=2,metric=True)
    
    sorted_mean = in_df.mean(axis=1).sort_values()
    select = sorted_mean.tail(top)
    #print(top)
    in_df = in_df.loc[select.index.values]
    temp_df = pd.DataFrame(pca.fit_transform(in_df.T),
                                 index=cols,columns =['pc_1','pc_2'] )
    
    temp_df['color']=palette
    fig,ax=plt.subplots(figsize=(12,6))
    temp_df.plot(kind='scatter',x='pc_1',y='pc_2',s=50, c=temp_df['color'], ax=ax)
    #print(temp_df.index.values)
       
    texts = [plt.text(temp_df.iloc[i]['pc_1'], 
                       temp_df.iloc[i]['pc_2'],
                       cols[i])
                       for i in range(temp_df.shape[0])]
    
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    ax.set_title('MDS',size=14)
    ax.set_xlabel('DIM_1',size=12)
    ax.set_ylabel('DIM_2',size=12)
    
    ax.yaxis.label.set_size(12)
    ax.xaxis.label.set_size(12)

    plt.show()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar, ax


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def add_subplot_index(ax, text='B)' ):
    ax.text(-0.1, 1.1, text, horizontalalignment='center', 
               verticalalignment='center',
               transform=ax.transAxes,fontsize=16)



def make_dict_from_cols(df, key='Protein IDs',
                        value='abundance'):
    res = {}
    for n,a in zip(df[key], df[value]):
        prot_list = n.split(';')
        prot_list = [clean_id(n) for n in prot_list]
        for prot in prot_list:
            res[prot]=a
    return res


def troncate_name(in_list):
    #remove puntuaction and space
    in_list = [s.replace('  ',' ') for s in in_list]
    in_list = [re.sub(r'[^\w]', ' ', s) for s in in_list]
    in_list = [re.sub(' +', ' ', s) for s in in_list]
    in_list = [s.replace(' ','_') for s in in_list]
    in_list = [s[0:25]+'...' if len(s) > 25 else s for s in in_list]
    return in_list


def impute(X):
    "Impute missing values from pandas dataframe"
    temp = X.copy()
    for col in temp.columns:  
        temp[col]=temp[col].fillna(temp[col].mean())
        temp[col] = temp[col].replace(-np.inf, -999)
        temp[col] = temp[col].replace(np.inf, 999)
    return temp

def run_lgb(X, y, params, test_size=0.35, random_state =1976, plot_roc=True):
    nfold=3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state =random_state, stratify=y)
    train_set = lgb.Dataset(X_train, y_train)
    print('DataSet size')
    print('X_train:',X_train.shape, 
    'X_test:',X_test.shape, 
    'y_train:',y_train.shape, 
    'y_test',y_test.shape,'\n',
    y_test.value_counts())

    best_scores = []
    optimal_rounds = []
    
    
    
    fig,ax = plt.subplots()
    
    for  seed in tqdm_notebook([10, 50, 1976, 2015, 34]):
        cv_results = lgb.cv(params, 
                        train_set, 
                        nfold = nfold, 
                        num_boost_round = 10000, 
                        early_stopping_rounds = 1, 
                        metrics = 'auc', seed = seed)
        #print(cv_results)
        best_score = max(cv_results['auc-mean'])
        optimal_round =len(cv_results['auc-mean'])
        #print(optimal_round)
        best_scores.append(best_score) 
        optimal_rounds.append(optimal_round)
        
        

    best_scores=pd.Series(best_scores)
    best_scores.plot(kind='box')
    plt.show()
    optimal_rounds=pd.Series(optimal_rounds)

    params['n_estimators']=max(optimal_rounds)
    model = lgb.LGBMClassifier(**params,
                )

    fitted_model = model.fit(X_train, y_train)
    
    blind_preds = fitted_model.predict(X_test)
    blind_prb = fitted_model.predict_proba(X_test)
    print(classification_report(blind_preds,y_test))
    blind_score = roc_auc_score(blind_preds,y_test)
    
    
    
    
    
    dummy = DummyClassifier()
    dummy.fit(X_train, y_train)
    dummy_preds = dummy.predict(X_test)
    dummy_prob = dummy.predict_proba(X_test)
    
    dummy_score = roc_auc_score(dummy_preds,y_test)
    print('1 dummy_roc_auc_score:', dummy_score)    
    print('1 blind_roc_auc_score:', blind_score)
    cm = confusion_matrix(blind_preds, y_test)

    print('\nConfusion Matrix')
    print(cm)
    
    
    if plot_roc:
        #Print Area Under Curve
        
        
        
        #print(y_test)
        #print(blind_preds)
        fig,ax = plt.subplots()
        false_positive_rate, recall, thresholds = roc_curve(y_test, blind_prb[:,-1])
        
        ax.set_title('Receiver Operating Characteristic (ROC)')
        ax.plot(false_positive_rate, recall, 'b', label = 'Blind AUC = %0.3f' %blind_score)
        
        false_positive_rate, recall, thresholds = roc_curve(y_test, dummy_prob[:,-1])
        ax.plot(false_positive_rate, recall, 'r', label = 'Dummy AUC = %0.3f' %dummy_score)
        
        plt.legend(loc='lower right')
        #plt.plot([0,1], [0,1], 'r--')
        plt.xlim([-0.1,1.1])
        plt.ylim([-0.1,1.1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    #ax= plt.subplot()
    #sns.heatmap(cm, annot=True, ax = ax)
    #ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    #ax.set_title('Confusion Matrix'); 
    #ax.xaxis.set_ticklabels(['1', '0']); 
    #ax.yaxis.set_ticklabels(['0', '1']);
    #plt.show()
    
    
    fitted_model = model.fit(X, y)

    return fitted_model, params, blind_score

def boruta_select(X, y, params):
    rf = RandomForestClassifier(**params)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=1)
    feat_selector.fit(X.values, y)
    print(feat_selector.support_)

    score_df = pd.DataFrame()
    #eliDf['fimp']=perm.feature_importances_
    score_df['support']=feat_selector.support_
    score_df['f']=X.columns
    return score_df

def eli_select(X, y, params):
    model = lgb.LGBMClassifier(
                **params,
                )
    fitted_model = model.fit(X, y)
    
    #the algorithms will runs n_iter times 
    #using a 3 fold cross validation
    perm = PermutationImportance(fitted_model, 
                                 random_state=1976,
                                 n_iter=10)

    #we need to replace misssing / infinite values
    #from the dataset as eli5 dosen't cope well with them
    temp = impute(X)

    temp.head()        
    perm.fit(temp, 
             y, cv=3, 
             scoring='auc')           
    #visualize the output
    n = eli5.show_weights(perm, 
                  feature_names = troncate_name(list(temp.columns)), 
                  top=30,show_feature_values=True)
    
    #add the output to a dataframe to do some selection
    eliDf = pd.DataFrame()
    eliDf['fimp']=perm.feature_importances_
    eliDf['f']=X.columns
    eliDf.sort_values(by='fimp',ascending=False).head()
    return eliDf


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def input_missing_values(in_df, columns):
    for n in tqdm_notebook(columns):
        #capture all the categorical features
        #if n == 'EF_CellCycle_max':
            #in_df[n]=in_df[n].replace(np.nan,in_df[n].min()+1)
        if not (n.endswith('_min_max')) and ( n.endswith('_max') or n.endswith('_min')):
            in_df[n]=in_df[n].replace(np.nan,in_df[n].min()-1)
        else:
            data = in_df[n].replace([np.inf, -np.inf], np.nan)
            data = data.dropna()
            min_data = data.min()
            max_data = data.max()
            ave_data = data.mean()
            in_df[n]=in_df[n].replace(np.nan,ave_data)
            in_df[n]=in_df[n].replace(-np.inf, ave_data)
            in_df[n]=in_df[n].replace(np.inf, ave_data )
            

    return in_df
 