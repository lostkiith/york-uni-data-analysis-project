import graphviz
import pandas as pd
import sklearn as sklearn
from numpy import mean
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class DataController(object):

    @staticmethod
    def convert_csv_to_json(filename) -> object:
        """" opens the csv file converts it into a dictionary then returns it as a json file."""
        try:
            with open(filename, encoding="utf-8-sig", newline='') as inFile:
                data_reader = pd.read_csv(inFile, low_memory=False).to_json(orient="records")
                return data_reader
        except FileNotFoundError:
            raise FileNotFoundError
        except TypeError:
            raise TypeError
        except ValueError:
            raise ValueError

    @staticmethod
    def prep_data(file_name):
        """" reads the file as a json object into a dataframe then drops duplicate and incomplete rows."""
        try:
            data_frame = pd.read_json(file_name, convert_dates=True)

            # break down to items of interest for general health and make column names readable.
            health = data_frame[
                ['GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'POORHLTH', '_HCVU651', 'PERSDOC2', 'MEDCOST', 'CHECKUP1',
                 '_RFHYPE5',
                 'BPMEDS', '_CHOLCHK', '_RFCHOL', '_MICHD', 'CVDSTRK3', 'ASTHMA3',
                 'ASTHNOW', 'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD1', 'HAVARTH3', 'ADDEPEV2', 'CHCKIDNY',
                 'DIABETE3', 'DIABAGE2', 'BLIND', 'SEX', 'AGE', '_RFBMI5', 'MARITAL', '_EDUCAG', 'RENTHOM1', 'EMPLOY1',
                 'CHILDREN', '_INCOMG', 'QLACTLM2', 'USEEQUIP', 'DECIDE', 'DIFFWALK',
                 '_SMOKER3', '_RFDRHV5', 'DIFFDRES', 'DIFFALON', 'EXERANY2', 'EXRACT11', '_PAINDX1', '_PASTRNG']]

            health.rename(columns={'GENHLTH': 'general_health', 'PHYSHLTH': 'thinking_physical_health-30d',
                                   'MENTHLTH': 'thinking_mental_health-30d',
                                   'POORHLTH': 'mental/physical_health_impact-30d',
                                   '_HCVU651': 'Health_Care_Access_18-64', 'PERSDOC2': 'named_doctor',
                                   'MEDCOST': 'needed_medical_but_cost', 'CHECKUP1': 'last_check-up',
                                   '_RFHYPE5': 'high_blood_pressure', 'BPMEDS': 'high_blood_pressure_meds',
                                   '_CHOLCHK': 'blood_cholesterol_checked_within_5y',
                                   '_RFCHOL': 'blood_cholesterol_checked_is_high',
                                   '_MICHD': 'have_had_heart_attack/heart_disease', 'CVDSTRK3': 'had_a_stroke',
                                   'ASTHMA3': 'had_asthma', 'ASTHNOW': 'still_have_asthma',
                                   'CHCSCNCR': 'had_skin_cancer', 'CHCOCNCR': 'other_cancer_types',
                                   'CHCCOPD1': 'COPD, emphysema_or_chronic_bronchitis',
                                   'HAVARTH3': 'arthritis_rheumatoid_arthritis_gout_lupus_fibromyalgia',
                                   'ADDEPEV2': 'depressive_disorder', 'CHCKIDNY': 'kidney_disease',
                                   'DIABETE3': 'have_diabetes', 'DIABAGE2': 'age_of_diabetes_onset',
                                   'BLIND': 'are you blind', 'SEX': 'sex', 'AGE': 'age', '_RFBMI5': 'Body Mass',
                                   '_EDUCAG': 'education_level', 'RENTHOM1': 'own/rent_your_home',
                                   'EMPLOY1': 'currently_employed', 'CHILDREN': 'children', '_INCOMG': 'income',
                                   '_SMOKER3': 'has_ever_smoked', '_RFDRHV5': "is_a_heavy_drinker",
                                   'QLACTLM2': 'limited_activities_because_of_physical/mental_problems',
                                   'USEEQUIP': 'health_problems_use_special_equipment',
                                   'DECIDE': 'physical/mental_condition_difficulty_concentrating',
                                   'DIFFWALK': 'serious_difficulty_walking_or_climbing_stairs',
                                   'DIFFDRES': 'difficulty_dressing_or_bathing',
                                   'DIFFALON': 'physical/mental_difficulty_doing_errands_alone',
                                   'EXERANY2': 'past_month_participate_in_physical_activities',
                                   'EXRACT11': 'type_of_physical_activity',
                                   '_PAINDX1': 'do_they_Meet_Aerobic_Recommendations',
                                   '_PASTRNG': 'do_they_meet_muscle_strengthening_recommendations'
                                   }, inplace=True)

            # convert from continual
            health['is_a_heavy_drinker'].replace({1: 'No', 2: 'Yes', 9: 'Refused'}, inplace=True)

            for column in health.columns:
                health.drop(index=health[health[column] == "Refused"].index, inplace=True)

            return health
        except TypeError as te:
            raise TypeError(f"Must be a JSON file. {te}")
        except ValueError as ve:
            raise ValueError(f"File not in correct format. {ve}")

    @staticmethod
    def prep_dataset_for_depressive_predictor(dataset):
        """" cleans the dataset for a depressive predictor."""

        # One Hot Encoding of categorical data
        categorical = [var for var in dataset.columns if dataset[var].dtype == 'object']

        # creates new columns for each of the categorical options.
        health = pd.get_dummies(dataset, columns=categorical)

        health.drop('depressive_disorder_No', axis=1, inplace=True)

        return health

    @staticmethod
    def prep_dataset_for_type_2_diabetes_predictor(dataset):
        """" cleans the dataset for a type 2 diabetes predictor."""

        # drop responses that are not useful from depressive_disorder
        dataset.drop('age_of_diabetes_onset', axis=1, inplace=True)
        dataset.drop(index=dataset[dataset['have_diabetes'] == "Don't know/Not sure"].index, inplace=True)

        # One Hot Encoding of categorical data
        categorical = [var for var in dataset.columns if dataset[var].dtype == 'object']

        # creates new columns for each of the categorical options.
        health = pd.get_dummies(dataset, columns=categorical)

        health.drop('have_diabetes_Yes', axis=1, inplace=True)
        health.drop('have_diabetes_No', axis=1, inplace=True)
        health.drop('have_diabetes_Yes, but female told only during pregnancy', axis=1, inplace=True)

        return health

    @staticmethod
    def Create_Decision_Tree_Model(health):
        # Decision Tree classification on depressive_predictor

        # set x to all features
        x = health.loc[:, health.columns != 'depressive_disorder_Yes']

        # set y to target depressive_disorder_Yes
        y = health.depressive_disorder_Yes

        # setup the test and training data.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)
        decision_tree = tree.DecisionTreeClassifier(random_state=1, max_depth=5, min_samples_leaf=1)

        decision_tree.fit(x_train, y_train)

        # test the model using cross validation
        cross_val = sklearn.model_selection.KFold(n_splits=10, random_state=1, shuffle=True)
        scores = sklearn.model_selection.cross_val_score(decision_tree, x, y, scoring='accuracy', cv=cross_val)
        average_score = mean(scores)
        print('Overall Accuracy:', average_score)

        classes = ["Yes", "no"]
        health.drop('depressive_disorder_Yes', axis=1, inplace=True)

        dot_data = tree.export_graphviz(decision_tree, out_file=None,
                                        feature_names=health.columns,
                                        class_names=classes,
                                        filled=True, rounded=True,
                                        special_characters=True)

        graph = graphviz.Source(dot_data)
        graph.render('depressive disorder decision tree')

        return decision_tree

    @staticmethod
    def Create_SVM_Model(health):
        # SVM on type_2_diabetes_predictor

        # set x to all features
        x = health.loc[:, health.columns != 'have_diabetes_No, pre-diabetes or borderline diabetes']

        # set y to target have_diabetes_No, pre-diabetes or borderline diabetes
        y = health['have_diabetes_No, pre-diabetes or borderline diabetes']

        # setup the test and training data.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=42)
        svm = SVC(kernel='poly', cache_size=500)

        svm.fit(x_train, y_train)

        # test the model using cross validation
        cross_val = sklearn.model_selection.KFold(n_splits=5, random_state=1, shuffle=True)
        scores = sklearn.model_selection.cross_val_score(svm, x, y, scoring='accuracy', cv=cross_val)
        average_score = mean(scores)
        print('Overall Accuracy:', average_score)

        return svm
