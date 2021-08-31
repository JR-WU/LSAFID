import pandas as pd
import math
from FeaSlect import Feature_Selector

def returnplus(mydict):
    plus_all = 1
    for i in mydict: 
        plus_all = plus_all * mydict[i] 
      
    return plus_all


def FEC_Classification(feature_numerator,feature_denominator,Alpha,L1,numerator,denominator,value_word_all,TestSet,result_file,exp):
    ratio_for_flows_list = []
    ratio_results = pd.DataFrame()
    nu_feature = math.ceil(len(numerator)/10)
    de_feature = math.ceil(len(denominator)/10)
    label_name = 'label'
    selector = Feature_Selector()
    results_ElasticNet_nu = selector.ElasticNet_selector(feature_numerator,Alpha,L1,numerator,label_name,nu_feature,scale=True)
    results_ElasticNet_de = selector.ElasticNet_selector(feature_denominator,Alpha,L1,denominator,label_name,de_feature,scale=True)
    Selected_words = results_ElasticNet_nu + results_ElasticNet_de
    
    value_word_selected = dict()
    for word in Selected_words:
        if word[2] == '_' and word[1] != '_':
            word_dealed = word[3:]
        elif word[1] == '_' and word[2] != '_':
            word_dealed = word[2:]
        else:
            print('wrong word selected!')
            exit(1)
        if word_dealed in value_word_all :
            value_word_selected[word] = value_word_all[word_dealed][word]
    for index_sep in range(len(TestSet)):
        numerator_add = dict() 
        denominator_add = dict()
        for index,value in TestSet.iloc[index_sep].items():
            for value_word_single in value_word_selected:
                if value_word_single[2] == '_' and value_word_single[1] != '_':
                    value_word_single_dealed = value_word_single[3:]
                elif value_word_single[1] == '_' and value_word_single[2] != '_':
                    value_word_single_dealed = value_word_single[2:]
                else:
                    print('There is no right word in value_word_selected!')
                    exit(1)
                if str.lower(index) == value_word_single_dealed:
                    if value >= value_word_selected[value_word_single][0] and value <= value_word_selected[value_word_single][1] and value_word_single in numerator:
                        if str.upper(value_word_single_dealed) == 'SYN_TO_PORT_PEERS':
                            numerator_add[value_word_single] = value + 0.1
                        else:
                            numerator_add[value_word_single] = value + 1.0
                    if value >= value_word_selected[value_word_single][0] and value <= value_word_selected[value_word_single][1] and value_word_single in denominator:
                        if str.upper(value_word_single_dealed) == 'SYN_TO_PORT_PEERS':
                            denominator_add[value_word_single] = value + 0.1
                        else:
                            denominator_add[value_word_single] = value + 1.0
        if not bool(numerator_add):
            numerator_add['void'] = 0
        if not bool(denominator_add):
            denominator_add['void'] = 0
        numerator_plus = returnplus(numerator_add)
        denominator_plus = returnplus(denominator_add)
        if numerator_plus == 0 and denominator_plus == 0: 
            ratio_for_flows = 0.0
        elif numerator_plus == 0 and denominator_plus != 0:
            ratio_for_flows = 0.0
        elif numerator_plus != 0 and denominator_plus == 0:
            ratio_for_flows = 100
        else:
            ratio_for_flows = numerator_plus / denominator_plus
        ratio_for_flows_list.append(ratio_for_flows)

    ratio_results['ratio'] = ratio_for_flows_list
    if exp == 'EX_2':
        ratio_results.index = ratio_results.index + 1
        ratio_results.to_csv(result_file)
        return ratio_results
    else:
        return ratio_results
