
import pandas as pd
import numpy as np
import graphlab as gl
import re

class Analizer(object):


    def __init__(self, new_file='data/5-Data.tsv'):

        self.df_data = pd.read_table(new_file)

        self.code_dict = self.codetxtIntoDict()

        self.df_small = None
        self.model = None
        self.predictions = None
        self.results = None

        self.predict = 'V4528'

        self._main()



    def _main(self):

        '''
        What:
        inputs:
        outputs:
        preproccessing:
        '''

        print ("\n main(), for " + self.code_dict[self.predict])

        low_var_features = ['V2001', 'V2009', 'V2027', 'V2028', 'V2029', 'V2030', 'V2031', 'V2109',
                            'V2110', 'V2112', 'V2114', 'V2115', 'V2123', 'V2131', 'V2142', 'V3001',
                            'V3027', 'V3051', 'V3057', 'V3060', 'V3069', 'V3082', 'V4001', 'V4319']

        overpowered_features = ['V4529', 'V4112', 'V4060']

        self.delFeatures(low_var_features)
        self.delFeatures(overpowered_features)

        self.model = self.doOneModel(self.predict, self.df_data)
        self.printResults()

        # make a slice of df_data with given feature_importance and then make a model with just those features
        self.df_small = self.computeWithN_Features(10)
        self.model = self.doOneModel(self.predict, self.df_small)
        self.printResults()


    def codetxtIntoDict(self, my_file='data/data_meta/BIG-Codebook.txt'):

        with open(my_file) as f:
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
            sp = ll.split('-')
            kk.append(sp)

        code_dict = {}

        for line in kk:
            code_dict[line[0]] = line[1]
        return code_dict








    def delFeatures(self, features):


        for cur_feat in features:
            del self.df_data[cur_feat]








    def doOneModel(self, predict, df_to_predict_on):

        sf_data = gl.SFrame(df_to_predict_on)

        if predict in self.code_dict.keys():
            print (predict + " is in the dictionary; " + predict + " = " + self.code_dict[predict] + '\n')

        # graphlab.boosted_trees_classifier.create(dataset, target, features=None,
        # max_iterations=10, validation_set='auto', class_weights=None, max_depth=6,
        # step_size=0.3, min_loss_reduction=0.0, min_child_weight=0.1, row_subsample=1.0,
        # column_subsample=1.0, verbose=True, random_seed=None, metric='auto', **kwargs)

        model = gl.boosted_trees_classifier.create(sf_data, predict, features=None,
                max_iterations=3, validation_set='auto', class_weights=None, max_depth=10,
                step_size=0.3, min_loss_reduction=0.0, min_child_weight=0.1, row_subsample=1.0,
                column_subsample=1.0, verbose=True, random_seed=42, metric='auto')

        self.predictions = model.predict(sf_data)
        self.results = model.evaluate(sf_data)
        print (" -> Finished computing model.")
        print ("\n<------------------------------------------------------------->")
        print ("Results for " + predict + " " + self.code_dict[predict] + " :\n")
        print ("<------------------------------------------------------------->")
        return model





    def printResults(self):

        print("Feature importance: \n", self.model.get_feature_importance())

        print("Model summary: \n", self.model.summary())



    def computeWithN_Features(self, N):

        top_n_features = N

        features_small = list(self.model.get_feature_importance()['name'][0:top_n_features])
        print ("top " + str(top_n_features) + " features of model;")

        for f in features_small:
            print (f, " ", self.code_dict[f])


        predict = 'V4528'
        # 4528, type of crime
        features_small.append(predict)

        df_small = self.df_data[features_small]

        return df_small






analizer()

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
