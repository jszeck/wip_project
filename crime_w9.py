
import pandas as pd
import numpy as np
import graphlab as gl
import re

class Analizer(object):


    def __init__(self, new_file='data/5-Data.tsv', new_predict = 'V4529'):
        '''
        What: initialize various objects for making a graphlab boosted trees classifier. Calls _main()

        Inputs: file location for data.tsv, which is dataset with 10,000 entries and 1100+ features
        string name of feature I want to make a new model for. See self.code_dict for looking up feature codes
        Outputs: a pandas dataframe based on input file, and a dictionary for looking up feature codes.
        '''


        self.df_data = pd.read_table(new_file)

        self.code_dict = self.codebookIntoDict()

        self.df_small = None
        self.model = None
        self.predictions = None
        self.results = None

        self.predict = new_predict

        self._main()


    def printCodes(self, codes):

        for i, code in enumerate(codes):
            if code in self.code_dict.keys():
                print code + ", " + self.code_dict[code]
            else:
                print code
        print "\n"


    def _main(self):

        '''
        What: This function calls the data hygiene, simple feature engineering, and model constuction methods;
        and prints the results

        Inputs: receives initial data from __init__
        Outputs: print statements
        '''

        ll = len(self.code_dict)
        testkey = self.predict in self.code_dict.values()
        print ("\n Main(), for " + self.predict + '\n')

        low_var_features = ['V2001', 'V2009', 'V2027', 'V2028', 'V2029', 'V2030', 'V2031', 'V2109',
                            'V2110', 'V2112', 'V2114', 'V2115', 'V2123', 'V2131', 'V2142', 'V3001',
                            'V3027', 'V3051', 'V3057', 'V3060', 'V3069', 'V3082', 'V4001', 'V4319',
                            'V4320']

        overpowered_features = ['V4528',
                                'V4112', 'V4060', 'V4094', 'V4364', 'V4321', 'V4287', 'V4289',
                                'V4373', 'V4288', 'V4290',
                                'V3080', 'V3026',
                                'V2008', 'V2002', 'V2116', 'V2117', 'V2118',
                                'INCREPWGT148', 'INCREPWGT40', 'INCREPWGT2', 'INCREPWGT3', 'INCREPWGT51',
                                'INCREPWGT14', 'INCREPWGT43',
                                'FRCODE', 'WGTPERCY', 'YEARQ', 'WGTHHCY']

        print "Over powered features: \n"
        self.printCodes(overpowered_features)

        self.delFeatures(low_var_features)
        self.delFeatures(overpowered_features)

        self.model = self.doOneModel(self.predict, self.df_data, 4)
        self.printResults()

        # make a slice of df_data with given feature_importance and then make a model with just those features
        self.df_small = self.getTopNfeatures(20)
        self.model = self.doOneModel(self.predict, self.df_small, 15)
        self.printResults()
        self.getTopNfeatures(15)


    def codebookIntoDict(self, my_file='data/data_meta/BIG-Codebook.txt'):

        '''
        What: Turns codebook.txt into a dictionary, when can be accessed with self.code_dict

        Inputs: file location of codebook.txt
        Outputs: dictionary of all 1100+ features stored into self.code_dict
        '''

        with open(my_file) as f:
            c_temp = f.readlines()

        codebook = c_temp[10880:40820]
        keys_values = []

        for line in codebook:

            matchObj = re.match('V[1-4]...', line, re.I)
            matchObj2 = re.match('INC', line, re.I)

            if matchObj and "-" in line:
                keys_values.append(line)
            if matchObj2:
                keys_values.append(line)

        # take off the last N values
        offset = 0
        keys_shorter = keys_values[0:len(keys_values)- offset]

        key_value_pairs = []

        for line in keys_shorter:
            # V4004 - D Assault: attempted
            if "-" in line:
                ll = line.replace("\n", "")
                sp = ll.split(' - ')
                key_value_pairs.append(sp)

        code_dict = {}
        keys_temp = code_dict.keys()

        for line in key_value_pairs:
            if len(line) > 1 and line[0] not in keys_temp:
                code_dict[line[0]] = line[1]

        return code_dict




    def delFeatures(self, features):


        for cur_feat in features:
            del self.df_data[cur_feat]






    def doOneModel(self, predict, df_to_predict_on, iters=5):
        '''
        What: Make a predictive model using a graphlab boosted trees classifier.

        Inputs: string predict, which is the feature we want to make a model of.
        pandas dataframe df_to_predict_on
        Outputs: model object
        various print statements
        '''
        sf_data = gl.SFrame(df_to_predict_on)

        if predict in self.code_dict.keys():
            print ("\n" + predict + " is in the dictionary; " + predict + " = " + self.code_dict[predict] + '\n')

        # graphlab.boosted_trees_classifier.create(dataset, target, features=None,
        # max_iterations=10, validation_set='auto', class_weights=None, max_depth=6,
        # step_size=0.3, min_loss_reduction=0.0, min_child_weight=0.1, row_subsample=1.0,
        # column_subsample=1.0, verbose=True, random_seed=None, metric='auto', **kwargs)

        model = gl.boosted_trees_classifier.create(sf_data, predict, features=None,
                max_iterations=iters, validation_set='auto', class_weights=None, max_depth=10,
                step_size=0.3, min_loss_reduction=0.0, min_child_weight=0.1, row_subsample=1.0,
                column_subsample=1.0, verbose=True, random_seed=42, metric='auto')

        self.predictions = model.predict(sf_data)
        self.results = model.evaluate(sf_data)
        print ("\n -> Finished computing model.")
        print ("\n<------------------------------------------------------------->")
        print ("Results for " + predict + " " + self.code_dict[predict] + " :\n")
        print ("<------------------------------------------------------------->")
        return model





    def printResults(self):

        print "\nFeature importance: \n"
        print self.model.get_feature_importance()

        #print("Model summary: \n", self.model.summary())



    def getTopNfeatures(self, N=10):
        '''
        What: After I compute a model for all features, then go and create a new pandas dataframe, df_small,
         to be used to re-make the model with only the top N most important features.

        Inputs: self.model, N number of most important features to recompute with
        Outputs: a pandas dataframe, df_small, which is a subset of self.df_data, including only the top N features
        various print statements
        '''
        top_n_features = N

        features_small = list(self.model.get_feature_importance()['name'][0:top_n_features])
        print ("\n Top " + str(top_n_features) + " features of model: \n")

        self.printCodes(features_small)


        predict = self.predict
        # 4528, type of crime
        features_small.append(predict)

        df_small = self.df_data[features_small]

        return df_small






Analizer()

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
