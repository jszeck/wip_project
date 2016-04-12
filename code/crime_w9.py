
import pandas as pd
import numpy as np
import graphlab as gl
import re



# ''' File 5: all of the above,
# tldr: the above table is all previous tables joined on year and quarter, household_id, and person_id
# 
# The incident-level extract file is not the same as the incident record-type file. The incident-level extract file is a file created by prepending
# household and person variables to incident records using the YEARQ (YEAR AND QUARTER OF INTERVIEW), IDHH (NCVS ID FOR
# HOUSEHOLDS), and IDPER (NCVS ID FOR PERSONS) variables as match keys. For data-year formats, this file has been "bounded"
# to contain incidents occurring within the specific calendar year, regardless of when the interview was conducted. Under the collection-year
# format, the incident-level extract file is not "bounded" based on when the incidents occurred; rather, it contains incidents reported in 2012,
# regardless of when they occurred
# '''


def main():



    code_dict = codetxtIntoDict()
    print (code_dict['V4115'])


def codetxtIntoDict():

    with open('data/data_meta/BIG-Codebook.txt') as f:
        codebook = f.readlines()

    codebook = codebook[10000:]

    objs = []
    v4_keys = []
    v4_values = []
    lastline = False
    for line in codebook:

        if lastline and line != "\n":
            v4_values.append(line)
        lastLine = False    

        matchObj = re.match('V4...', line, re.I)

        if matchObj:
            objs.append(matchObj)
            v4_keys.append(line)
            lastline = True

    v4_keys = v4_keys[0:604]
    kk = []
    for line in v4_keys:
        ll = line.replace(" - ", "-").replace("\n", "")
        #print ll
        sp = ll.split('-')
        kk.append(sp)

    code_dict = {}
    for line in kk:
        code_dict[line[0]] = line[1]
    return code_dict
    




df_data = pd.read_table('data/5-Data.tsv')



def delFeatures(features):
    for cur_feat in features:
        del df_data[cur_feat]
        
        

low_var_features = ['V2001', 'V2009', 'V2027', 'V2028', 'V2029', 'V2030', 'V2031', 'V2109', 
                    'V2110', 'V2112', 'V2114', 'V2115', 'V2123', 'V2131', 'V2142', 'V3001', 
                    'V3027', 'V3051', 'V3057', 'V3060', 'V3069', 'V3082', 'V4001', 'V4319', 
                    'V4320']

                        #V4528
overpowered_features = ['V4529', 'V4112', 'V4060']

'''
V4528    TYPE OF CRIME CODE (OLD, NCS)
V4529 - TYPE OF CRIME CODE (NEW, NCVS)
V4112 -    INJURIES: RAPE INJURIES  
V4060 - OFFENDER HIT OR ATTACK (ALLOCATED)
'''
delFeatures(low_var_features)
delFeatures(overpowered_features)





def doOneModel(predict, df_to_predict_on):
    
    sf_data = gl.SFrame(df_to_predict_on)
    
    if predict in code_dict.keys():
        print (predict + " is in the dictionary; " + predict + " = " + code_dict[predict] + '\n')
    
    # graphlab.boosted_trees_classifier.create(dataset, target, features=None, 
    # max_iterations=10, validation_set='auto', class_weights=None, max_depth=6, 
    # step_size=0.3, min_loss_reduction=0.0, min_child_weight=0.1, row_subsample=1.0, 
    # column_subsample=1.0, verbose=True, random_seed=None, metric='auto', **kwargs)
    
    model = gl.boosted_trees_classifier.create(sf_data, predict, features=None, 
            max_iterations=3, validation_set='auto', class_weights=None, max_depth=10, 
            step_size=0.3, min_loss_reduction=0.0, min_child_weight=0.1, row_subsample=1.0, 
            column_subsample=1.0, verbose=True, random_seed=42, metric='auto')
    
    predictions = model.predict(sf_data)
    results = model.evaluate(sf_data)
    print (" -> Finished computing model.")
    print ("\n<------------------------------------------------------------->")
    print ("Results for " + predict + " " + code_dict[predict] + " :\n")
    print ("<------------------------------------------------------------->")
    return model
    
    



def printResults(model):
    
    print ("Feature importance: \n", model.get_feature_importance())
    
    print ("Model summary: \n", model.summary())



def printNFeatures(N, model):

    top_n_features = N

    features_small = list(model.get_feature_importance()['name'][0:top_n_features])
    print ("top " + str(top_n_features) + " features of model;")

    for f in features_small:
        print (f, " ", code_dict[f])
        

predict = 'V4528'
# 4528, type of crime
features_small.append(predict)

df_small = df_data[features_small]
#df_small.head()
    
    return features_small




predict = 'V4528'

m_boostedT = doOneModel(predict, df_data)

printResults(m_boostedT)
topNFeatures = printNFeatures(10, m_boostedT)



'''
Multiple columns can be selected by passing a list of column names:
>>> sf = SFrame({'id':[1,2,3],'val':['A','B','C'],'val2':[5,6,7]})

'''

# make a slice of df_data with given feature_importance and then make a model with just those features
# 4528, type of crime

m_boosted_small = doOneModel(predict, df_small)
printResults(m_boosted_small)


#
# 
# # Things to try later:
# https://dato.com/products/create/docs/graphlab.toolkits.feature_engineering.html#categorical-features
# 
# RandomProjection 	Project high-dimensional numeric features into a low-dimensional subspace.
# 
# OneHotEncoder 	Encode a collection of categorical features using a 1-of-K encoding scheme.
# CountThresholder 	Map infrequent categorical variables to a new/separate category.
# CategoricalImputer 	The purpose of this imputer is to fill missing values (None) in data sets
#  that have categorical data.
# 
# 
