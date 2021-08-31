#连续元素离散化
import pandas as pd
import csv
from sklearn.cluster import KMeans
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from plsa import Corpus, Pipeline
from plsa.pipeline import DEFAULT_PIPELINE
from plsa.algorithms import PLSA
from Tools.FEC_Elastic_Net import FEC_Classification
from Tools.externalmetrics import ex_metrics

#Parameters
K = 8
Alpha = 0.3
L1 = 0.2

#Sampling Rate
Sample_Ratio = 0.5 #选取的采样比0.5。

#Initialize path
DataSetPath = ' ' # Your path of experiment.
DataSetName = 'UNSW-NB15'
ExperimentName = 'EX_2'

#Settthe number of topics
n_topics = 8 
filter_topic = 1

# Delete unwanted features from the dataset
del_list = ['ID','SAVETIME','USERIP','label'] #UNSW-NB15

def Kmeans(k,data):
    kmodel = KMeans(n_clusters = k, max_iter = 500, n_init = k) 
    kmodel.fit(data.values.reshape((len(data),1))) 
    c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)
    w = c.rolling(2).mean().iloc[1:] 
    w_list = list(w[0])
    if min(w_list) < 0 :
        min_index = w_list.index(min(w_list))
        w_list[min_index] = 0.0
    if max(w_list) > data.max() :
        max_index = w_list.index(max(w_list))
        w_list[max_index] = data.max()
    w = [-1] + w_list + [data.max() + 1] 
    w = [round(i,3) for i in w]
    w = excute_duplicate(w)
    labels_1 = range(len(w) - 1) 
    return pd.cut(data,w,labels = labels_1,duplicates="drop"),w

def float_to_int_in_list(list_f):
    list_i = []
    for i in list_f:
        i =int(i)
        list_i.append(i)
    return list_i

def excute_duplicate(old_list):
    new_list = []
    for i in old_list:
        if i not in new_list:
            new_list.append(i)
    return new_list

def dupe(items):
    seen = set()
    for item in items:
        if item not in seen:
            yield item
            seen.add(item)


def deal_zero_in_df(element_from_df):
    if element_from_df == 0 :
        return element_from_df + 1
    else:
        return element_from_df


def topn_dict(d, n):   
    L = sorted(d.items(),key = lambda x:x[1],reverse = True) #对value排序
    max_L = L[:n]
    min_L = L[(-n):]
    return max_L,min_L


def arg_of_value_in_dict(d): 
    sum_arg = 0
    for k,v in d.items():
        sum_arg = sum_arg + v
    return sum_arg / len(d.keys())

def returnplus(mydict):
    plus_all = 1
    for i in mydict: 
        plus_all = plus_all * mydict[i] 
      
    return plus_all

# Begin
trainfile = DataSetPath + DataSetName + '/'+ ExperimentName + '/' + DataSetName + '_EX_2_' + 'TrainSet' + str(Sample_Ratio) + '.csv'
testfile = DataSetPath + DataSetName + '/'+ ExperimentName + '/' + DataSetName + '_EX_2_' + 'TestSet' + str(Sample_Ratio) + '.csv'

dealed_file =  DataSetPath + DataSetName + '/'+ ExperimentName + '/' + DataSetName + '_EX_2_' + 'TrainSet'  + str(Sample_Ratio) + '_doc.csv'
separated_doc_file =  DataSetPath + DataSetName + '/'+ ExperimentName + '/' + DataSetName + '_EX_2_' + 'TrainSet'  + str(Sample_Ratio) + '_separated.csv'
result_file =  DataSetPath + DataSetName + '/Results' + '/'+ ExperimentName + '/' + DataSetName + '_EX_2_' + '_result_ratio.csv'

csvFile = open(dealed_file,'w',newline='')
writer = csv.writer(csvFile)
string_write = []
row_write =[]
data_write_in = pd.DataFrame()
value_word_all = dict()
TestSet = pd.read_csv(testfile)
real_label_test = TestSet.iloc[:,-1]
TestSet =  TestSet.drop(columns=del_list)

data = pd.read_csv(trainfile) 
for indicator in range(0,len(data.keys())):
    row_write = []
    attribute_name = data.keys()[indicator] 
    data_attribute = data[attribute_name].copy() 
    if attribute_name in del_list:
        continue

    #Kmeans
    d3,list_cut = Kmeans(K,data_attribute)
    list_cut[0] = 0
    list_cut[-1] = list_cut[-1] - 1

    new_words = dict()
    for i in range(0,len(list_cut) - 1):
        new_words_name = '%d_%s'%(i,attribute_name)
        new_words[i] = str.lower(new_words_name)

    value_word = dict()
    for i in range(0,len(list_cut) - 1):
        value_word[new_words[i]] = [list_cut[i],list_cut[i+1]]
    value_word_all[str.lower(attribute_name)] = value_word

    for token in range(len(d3)):
        num_def = d3[token]
        if new_words.get(num_def):
            new_words_for_csv = new_words.get(num_def)
            row_write.append(new_words_for_csv)
    data_write_in[attribute_name] = row_write

Document_file = pd.DataFrame(columns=['documents'])
for indicator_assemble in range(0,data_write_in.shape[0]):
    #以行操作
    string_for_features = ""
    string_write = []
    for feature_real in data_write_in.iloc[indicator_assemble]:
        string_for_features = string_for_features + feature_real + " "
    string_for_features = string_for_features[:-1] 
    Document_file = Document_file.append([{'documents':string_for_features}], ignore_index=True)
Document_file.to_csv(dealed_file,index = False,line_terminator='\n')
delete_blank_row(dealed_file)
data_write_in.to_csv(separated_doc_file,sep=',',index = False,line_terminator='\n')
delete_blank_row(separated_doc_file)

train_set = pd.read_csv(trainfile) 
real_label = train_set.iloc[:,-1]
for del_element in del_list:
    train_set = train_set.drop([del_element],axis=1) 
features_trainset = train_set.columns.values.tolist()
train_set = train_set.applymap(deal_zero_in_df)


pipeline = Pipeline(*DEFAULT_PIPELINE)
corpus = Corpus.from_csv(dealed_file, pipeline, col = 0)

topic_for_abnormal = dict() 
topic_for_all = dict() 
rate_for_abnormal = dict() 
numerator = [] 
denominator = [] 
max_n_topic = []
min_n_topic = []
ratio_results = pd.DataFrame()
key_to_word_nu = []
key_to_word_de = []


for index_topic in range(0,n_topics):
    topic_for_all[index_topic] = 0
    topic_for_abnormal[index_topic] = 0
    rate_for_abnormal[index_topic] = 0.0

break2 = False
while(1):
    for index_topic in range(0,n_topics):
        topic_for_all[index_topic] = 0
        rate_for_abnormal[index_topic] = 0.0
        topic_for_abnormal[index_topic] = 0
    plsa = PLSA(corpus, n_topics, True)
    result = plsa.fit()
    for doc_index in range(0,len(result.topic_given_doc)):
        topic_index = result.topic_given_doc[doc_index].tolist().index(max(result.topic_given_doc[doc_index].tolist()))
        topic_for_all[topic_index] = topic_for_all[topic_index] + 1
        if (real_label[doc_index] == 1):
            topic_for_abnormal[topic_index] = topic_for_abnormal[topic_index] + 1
    for index_to_all in range(0,n_topics):
        if topic_for_all[index_to_all] == 0:
            topic_for_all[index_to_all] = topic_for_all[index_to_all] + 1
        rate_for_abnormal[index_to_all] = topic_for_abnormal[index_to_all] / topic_for_all[index_to_all] 
    for rate_i in rate_for_abnormal.keys():
        if rate_for_abnormal[rate_i] > 0.5:
            break2 = True
            break
    print(rate_for_abnormal)
    if break2:
        break

max_n_topic,min_n_topic = topn_dict(rate_for_abnormal,1) 

for max_n in max_n_topic:
    word_given_topic = list(result.word_given_topic[max_n[0]])
    topic_given_word = list(result._PlsaResult__topic_given_word[max_n[0]])
    for index_word in range(len(topic_given_word)):
        if topic_given_word[index_word] == 1:
            numerator.append(list(corpus.vocabulary.values())[index_word])
        if topic_given_word[index_word] == 0:
            denominator.append(list(corpus.vocabulary.values())[index_word])

columns_name = numerator + denominator
feature_numerator = pd.DataFrame(columns = numerator)
feature_denominator = pd.DataFrame(columns = denominator )
for feature_selected_PLSA in numerator:
    key_to_word = list(corpus.vocabulary.keys())[list(corpus.vocabulary.values()).index(feature_selected_PLSA)] 
    key_to_word_nu.append(key_to_word)

for feature_selected_PLSA in denominator:
    key_to_word = list(corpus.vocabulary.keys())[list(corpus.vocabulary.values()).index(feature_selected_PLSA)] 
    key_to_word_de.append(key_to_word)

#Construct the input doc-word matrix for Elastic Net.
index_doc = 0
for array_doc_word in corpus._Corpus__doc_word:
    doc_word_de = []
    doc_word_nu = []
    list_f = array_doc_word.tolist()
    list_i = float_to_int_in_list(list_f)

    for word_index in key_to_word_de:
        doc_word_de.append(list_i[word_index])

    for word_index in key_to_word_nu:
        doc_word_nu.append(list_i[word_index])
    
    feature_denominator.loc[index_doc] = doc_word_de
    feature_numerator.loc[index_doc] = doc_word_nu

    index_doc += 1
feature_denominator['label'] = real_label
feature_numerator['label'] = real_label

#FEC
ratio_results = FEC_Classification(feature_numerator,feature_denominator,Alpha,L1,numerator,denominator,value_word_all,TestSet,result_file,'EX_2')

#External Metrics

external_metrics = ex_metrics(ratio_results,real_label_test,ExperimentName)

print('##########################Experiment is done########################')