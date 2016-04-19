
import pandas as pd
import numpy as np
import graphlab as gl
import re
from collections import Counter
from statsmodels.discrete.discrete_model import Logit
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from statsmodels.tools import tools

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
        self.crime_dict = {1: "Sexual assault or rape", 5: "Robbery with assault", 7: "Robbery without assault",
                           11: "Assault", 12: "Assault with a weapon", 16: "Unwanted sexual contact without force",
                           21: "Pickpocketing or purse snatching", 31: "Home invasion theft", 40: "Car theft",
                           54: "Theft under $250", 57: "Theft over $250", -1: "Not defined / error"}

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

    def compressV4529(self):
        '''
        What: squish down many similiar types of crime.
                IE, (assault,completed + assault,attempted = assault)

        In: self.df_data
        Out: modifies self.df_data['V4529'] to have less variance
        '''
        # make an empty list
        values = [None] * len(self.df_data['V4529'])

        for i, oldCode in enumerate(self.df_data['V4529']):
            crimeCode='unassigned'

            '''
            if int(oldCode) in [1, 2, 3, 4, 15, 18, 19]:
                crimeCode = 'Sexual assault'  # turn rape and sexual assault into one cat.

            elif int(oldCode) in [5, 6, 8, 9]:
                crimeCode = 'Theft with assault'  # theft with assault'

            elif int(oldCode) in [7, 10]:
                crimeCode = 'Theft without assault'  # 'robbery withOUT assault'

            elif int(oldCode) in [11, 14, 17, 20]:
                crimeCode = 'Assault'  # assault

            elif int(oldCode) in [12, 13]:
                crimeCode = 'Assault with weapon'  # assault with a Weapon

            elif int(oldCode) in [16]:
                crimeCode = 'Unwanted sexual contact'  # unwanted sexual contact without force

            elif int(oldCode) in [21, 22, 23]:
                crimeCode = 'P.pocketing/purse-snatching'  # pickpocketing / purse snatching

            elif int(oldCode) in [31, 32, 33]:
                crimeCode = 'Home invasion/theft'  # home invasion burglary

            elif int(oldCode) in [40, 41]:
                crimeCode = 'Car theft'  # car theft

            elif int(oldCode) in [54, 55, 56, 58, 59]:
                crimeCode = 'Theft under $250'  # theft under $250

            elif int(oldCode) in [57]:
                crimeCode = 'Theft over $250'  # theft over $250
            '''


            if int(oldCode) in [1,2,3,4,15,18,19]:
                crimeCode = 1 # turn rape and sexual assault into one catagory

            elif int(oldCode) in [5,6,8,9]:
                crimeCode = 5 # robbery with assault'

            elif int(oldCode) in [7, 10]:
                crimeCode = 7  # 'robbery withOUT assault'

            elif int(oldCode) in [11,14,17,20]:
                crimeCode = 11  # assault

            elif int(oldCode) in [12,13]:
                crimeCode = 12  # assault with a Weapon

            elif int(oldCode) in [16]:
                crimeCode = 16  # unwanted sexual contact without force

            elif int(oldCode) in [21,22,23]:
                crimeCode = 21  # pickpocketing / purse snatching

            elif int(oldCode) in [31,32,33]:
                crimeCode = 31  # home invasion burglary

            elif int(oldCode) in [40,41]:
                crimeCode = 40  # car theft

            elif int(oldCode) in [54,55,56,58,59]:
                crimeCode = 54  # theft under $250

            elif int(oldCode) in [57]:
                crimeCode = 57  # theft over $250


            values[i] = crimeCode

        del self.df_data['V4529']
        self.df_data['V4529'] = values


    def trimFeatures(self, verbose=False):
        low_var_features = ['V2001', 'V2009', 'V2027', 'V2028', 'V2029', 'V2030', 'V2031', 'V2109',
                            'V2110', 'V2112', 'V2114', 'V2115', 'V2123', 'V2131', 'V2142', 'V3001',
                            'V3027', 'V3051', 'V3057', 'V3060', 'V3069', 'V3082', 'V4001', 'V4319',
                            'V4320']

        overpowered_features = ['V4528','V4529', 'V4526', # 4526/8/9, type of crime
                            'V4140B1', 'V4140B2', 'V4140B3',
                            'V4002', 'V4049', 'V4097', 'V4098', 'V4099', 'V4100', 'V4101', # 4097, attacked: shot
                            'V4102', 'V4103', 'V4104', 'V4105', 'V4106', 'V4107', 'V4108', 'V4109', 'V4111', 'V4123',
                            'V4059', 'V4093', 'V4062', 'V4005', 'V4077', 'V4060', 'V4061', 'V4096', 'V4127', 'V4040',
                            'V4112', 'V4094', 'V4364', 'V4321', 'V4287', 'V4288', 'V4289', 'V4028', 'V4027',
                            'V4373', 'V4290', 'V4012', 'V4026', 'V4095', 'V4078', 'V4079', 'V4080', 'V4081', 'V4082',
                            'V4050', # 4050, what was weapon
                            'V4291', 'V4292', 'V4293', 'V4294', 'V4295', 'V4296', 'V4297', 'V4298',
                            'V4322', 'V4323', 'V4324', 'V4326', 'V4357', 'V4385', 'V4397',
                            'V4063', 'V4064', 'V4065', 'V4066', 'V4067', 'V4068', 'V4069', 'V4070', #these are all targets
                            'V4422', 'V4011', # 4422, reason not reported
                            'V3080', 'V3026', 'V3002', 'V3008', 'V3013', 'V3005',
                            'V2008', 'V2002', 'V2116', 'V2117', 'V2118', 'V2033', 'V2005', 'V2006',
                            # 'INCREPWGT148', 'INCREPWGT40', 'INCREPWGT2', 'INCREPWGT3', 'INCREPWGT51',
                            # 'INCREPWGT14', 'INCREPWGT43', 'INCREPWGT57', 'INCREPWGT83',
                            'FRCODE', 'WGTPERCY', 'WGTHHCY', 'IDPER', 'IDHH', 'V4008', 'YEARQ']  #

        if self.predict in overpowered_features:
            overpowered_features.remove(self.predict)

        if verbose is False:
            print "Over powered features: \n"
            self.printCodes(overpowered_features)

        self.delFeatures(low_var_features)
        self.delFeatures(overpowered_features)


    def examineFeature(self, feature='V4529'):

        print "Examine: " + feature

        df_examine = self.df_data[feature]

        print str(df_examine.describe()) + "\n"

        freq = Counter(list(df_examine.values))

        for item in freq.items():
            print "item ", item

    def testForDataLeakage(self, typeOfCrime=1, top_n=10):

        feature_aucs = {}

        for feature in list(self.df_data.columns.values):
            model_cur = LogisticRegression()

            cur_X = self.df_data[feature].values
            cur_X = cur_X.reshape(cur_X.shape[0], 1)

            cur_y = self.df_data[self.predict].apply(lambda x: 1 if x==typeOfCrime else 0).values

            auc_cur = cross_val_score(model_cur, cur_X, cur_y, n_jobs=-1)

            feature_aucs[feature] = np.mean(auc_cur)

        auc_counter = Counter(feature_aucs).most_common(top_n)

        return list(auc_counter)

    def _main(self):

        '''
        What: This function calls the data hygiene, simple feature engineering, and model constuction methods;
        and prints the results

        Inputs: receives initial data from __init__
        Outputs: print statements
        '''
        # self.printSurveyAbstract()
        print ("\n Main(), for " + self.predict + '\n')

        self.trimFeatures(verbose=True)

        #self.regressOnFeatureSM()

        self.compressV4529()
        #self.examineFeature(self.predict)

        crime_to_features_causation = []

        for crime in [1,11]:
            top_features = self.testForDataLeakage(crime, 10)
            crime_to_features_causation.append((self.crime_dict[crime], top_features))

        for correlation in crime_to_features_causation:
            print correlation
            print ""

        #self.predictFeatureGL(8)



    def printCrimeDict(self):
        print "\n"
        for item in self.crime_dict.items():
            print item


    def regressOnFeatureSM(self):
        '''
        What: create a regression model with Statsmodels and try and find the Betas,
        ie, try to find feature causation.

        In: trimmed-down data
        Out: print statements and a linear model regression
        '''
        df_dummies = pd.get_dummies(self.df_data)

        df_dummies = pd.concat([df_dummies, self.df_data], axis=0)

        X_train, X_test, y_train, y_test = train_test_split(df_dummies, self.df_data[self.predict].values,
                                                            test_size=0.3, random_state=42)
        X_train = tools.add_constant(X_train)
        modelSM = Logit(y_train, X_train).fit()



        print modelSM.summary()


    def predictFeatureGL(self, GLiters=4):
        '''
        What: Does model constuction methods; and prints the results

        Inputs: receives initial data from __init__
        Outputs: print statements, model -> self.model
        '''

        self.model = self.doOneGLModel(self.predict, self.df_data, iters=GLiters)
        self.printResults()

        #if self.predict == 'V4529':
         #   self.printCrimeDict()

        # make a slice of df_data with given feature_importance and then make a model with just those features
        self.df_small = self.getTopNfeatures(20)
        self.model = self.doOneGLModel(self.predict, self.df_small, iters=GLiters*3)
        self.printResults()

        self.getTopNfeatures(len(self.df_small))



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



    def doOneGLModel(self, predict, df_to_predict_on, iters=5):
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
                max_iterations=iters, validation_set='auto', class_weights=None, max_depth=15,
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

        print("\nModel results: \n")

        print self.results



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

        if " HC" in features_small:
            print 'Note: HC = Hate Crime \n'

        self.printCodes(features_small)


        predict = self.predict
        # 4528, type of crime
        features_small.append(predict)

        df_small = self.df_data[features_small]

        return df_small


    def printSurveyAbstract(self):
        abstract = '''The National Crime Victimization Survey (NCVS), ... has been collecting data on personal and
        household victimization through an ongoing survey of a nationally-representative sample of residential addresses since 1972. The survey
        is administered by the U.S. Census Bureau (under the U.S. Department of Commerce) on behalf of the Bureau of Justice Statistics (under
        the U.S. Department of Justice).
        The NCS and NCVS were both designed with four primary objectives:

        To develop detailed information about the victims and consequences of crime,
            To estimate the numbers and types of crimes not reported to the police,
            To provide uniform measures of selected types of crimes, and
            To permit comparisons over time and types of areas.

        Beginning in 1992 the NCVS categorizes crimes as "personal" or "property" covering the personal crimes of rape and sexual attack,
        robbery, aggravated and simple assault and purse-snatching/ pocket-picking; and the property crimes of burglary, theft, and motor vehicle
        theft. Beyond a simple count of victimizations, the NCVS gathers details on the incident itself insofar as the victim can report them. Such
        -4-- Study 34650 -
        details include the month, time, and location of the crime, the relationship between victim and offender, characteristics of the offender,
        self-protective actions taken by the victim during the incident and results of those actions, consequences of the victimization, type of
        property lost, whether the crime was reported to police and reasons for reporting or not reporting, and offender use of weapons, drugs,
        and alcohol. Basic demographic information, such as age, race, gender, and income, is also collected to enable analysis of crime by
        various sub-populations for select years. Information is also obtained on vandalism and identity theft experienced by the household. The
        glossary (see p. 493) describes terms relevant to the NCVS.
        Analysts who are interested in the history of the redesign of the NCVS or are interested in conducting analyses using both NCS and NCVS
        data should refer to the NCVS Resource Guide (http://www.icpsr.umich.edu/NACJD/NCVS) on the NACJD Web site.'''
        print abstract

Analizer(new_predict='V4529')
# V4529 type of crime
# V4060 OFFENDER HIT OR ATTACK (ALLOCATED)


'''
More things to try

make sub populations and make models for them

iterate over 100 feature and models for each.
see which features are easiet to predict

'''


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
