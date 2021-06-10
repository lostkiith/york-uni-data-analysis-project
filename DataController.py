import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import sklearn as sklearn
from sklearn.linear_model import LogisticRegression
from numpy import mean
import pandas as pd
import pymongo

from pymongo.errors import ServerSelectionTimeoutError


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
    def replace_database_collection(file):
        """" replaces the database collection if it exists with the new collection."""
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=1000)
        try:
            db = client["HealthData"]
            db.drop()  # drops current collection
            db.insert_many(file.reset_index().to_dict('records'))
        except ServerSelectionTimeoutError as exc:
            raise RuntimeError('Failed to open database') from exc

    @staticmethod
    def read_from_database(choice):
        """" returns the data stored in the database by the file name."""
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=1000)
        try:
            db = client["HealthData"]
            return pd.DataFrame(db.find({}, {'index': 0}))
        except ServerSelectionTimeoutError as exc:
            raise RuntimeError('Failed to open database') from exc

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
                 'DIABETE3', 'DIABAGE2', 'SEX', 'AGE', '_RFBMI5', 'MARITAL', '_EDUCAG', 'RENTHOM1', 'EMPLOY1',
                 'CHILDREN', '_INCOMG', 'QLACTLM2', 'USEEQUIP', 'DECIDE', 'DIFFWALK',
                 'DIFFDRES', 'DIFFALON', 'EXERANY2']]

            health.rename(columns={'GENHLTH': 'general_health', 'PHYSHLTH': 'thinking_physical_health-30d',
                                   'MENTHLTH': 'thinking_mental_health-30d',
                                   'POORHLTH': 'mental/physical_health_impact-30d',
                                   '_HCVU651': 'Health_Care_Access_18-64', 'PERSDOC2': 'personal_doctor',
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
                                   'SEX': 'sex', 'AGE': 'age', '_RFBMI5': 'BMI', '_EDUCAG': 'education_level',
                                   'RENTHOM1': 'own/rent_your_home',
                                   'EMPLOY1': 'currently_employed', 'CHILDREN': 'children', '_INCOMG': 'income',
                                   'QLACTLM2': 'limited_activities_because_of_physical/mental_problems',
                                   'USEEQUIP': 'health_problems_use_special_equipment',
                                   'DECIDE': 'physical/mental_condition_difficulty_concentrating',
                                   'DIFFWALK': 'serious_difficulty_walking_or_climbing_stairs',
                                   'DIFFDRES': 'difficulty_dressing_or_bathing',
                                   'DIFFALON': 'physical/mental_difficulty_doing_errands_alone',
                                   'EXERANY2': 'past_month_participate_in_physical_activities',
                                   'EXRACT11': 'type_of_physical_activity'
                                   }, inplace=True)

            # data_frame = data_frame.dropna()  # drops incomplete rows
            # data_frame = data_frame.drop_duplicates()  # drops duplicate rows from dataframe.

            return health
        except TypeError as te:
            raise TypeError(f"Must be a JSON file. {te}")
        except ValueError as ve:
            raise ValueError(f"File not in correct format. {ve}")

    @staticmethod
    def prep_dataset_for_depressive_predictor(dataset):
        """" cleans the dataset for a depressive predictor."""

        # drop responses that are not useful from depressive_disorder
        dataset.drop(index=dataset[dataset['depressive_disorder'] == "Don't know/Not sure"].index, inplace=True)
        dataset.drop(index=dataset[dataset['depressive_disorder'] == "Refused"].index, inplace=True)

        # One Hot Encoding of categorical data
        categorical = [var for var in dataset.columns if dataset[var].dtype == 'object']

        # print("Number of categorical variables: ", len(categorical))
        # print(categorical)
        # for col in categorical:
        # print(np.unique(health[col]))

        # creates new columns for each of the categorical options.
        health = pd.get_dummies(dataset, columns=categorical)

        health.drop('depressive_disorder_No', axis=1, inplace=True)

        return health

    @staticmethod
    def clean_dataset_for_k_mean_clustering(dataset):
        """" cleans the dataset for k-mean clustering."""

        return dataset

    @staticmethod
    def Create_Logistic_Regression_Model(health):
        # logistic regression on general_health_Poor

        # set x to all features
        x = health.loc[:, health.columns != 'depressive_disorder_Yes']

        # set y to target depressive_disorder_Yes
        y = health.depressive_disorder_Yes

        # setup the test and training data.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
        logreg = LogisticRegression(solver='liblinear')
        logreg.fit(x_train, y_train)
        logreg.max_iter = 10000

        # test the model using cross validation
        cross_val = sklearn.model_selection.KFold(n_splits=10, random_state=1, shuffle=True)
        scores = sklearn.model_selection.cross_val_score(logreg, x, y, scoring='accuracy', cv=cross_val)
        average_score = mean(scores)
        print('Overall Accuracy:', average_score)

        return logreg
