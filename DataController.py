import graphviz
import imblearn
import numpy as np
import pandas as pd
import sklearn as sklearn
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_roc_curve
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.naive_bayes import ComplementNB


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
                 '_RFHYPE5', 'BPMEDS', '_CHOLCHK', '_RFCHOL', '_MICHD', 'CVDSTRK3', 'ASTHMA3',
                 'ASTHNOW', 'CHCSCNCR', 'CHCOCNCR', 'CHCCOPD1', 'HAVARTH3', 'ADDEPEV2', 'CHCKIDNY',
                 'DIABETE3', 'DIABAGE2', 'BLIND', 'SEX', 'AGE', '_RFBMI5', 'MARITAL', '_EDUCAG', 'RENTHOM1', 'EMPLOY1',
                 'CHILDREN', '_INCOMG', 'QLACTLM2', 'USEEQUIP', 'DECIDE', 'DIFFWALK',
                 '_SMOKER3', '_RFDRHV5', 'DIFFDRES', 'DIFFALON', 'EXERANY2', 'EXRACT11', '_PAINDX1', '_PASTRNG',
                 '_FRTLT1', '_VEGLT1']]

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
                                   'BLIND': 'are you blind', 'SEX': 'sex', 'AGE': 'age', '_RFBMI5': 'Body_Mass',
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
                                   '_PASTRNG': 'do_they_meet_muscle_strengthening_recommendations',
                                   '_FRTLT1': 'Fruit_intake_per_day',
                                   '_VEGLT1': 'Vegetable_intake_per_day'
                                   }, inplace=True)

            # convert from continual
            health['is_a_heavy_drinker'].replace({1: 'No', 2: 'Yes', 9: 'Refused'}, inplace=True)

            # remove any rows with refused from the data
            for column in health.columns:
                health.drop(index=health[health[column] == "Refused"].index, inplace=True)

            # numeric outlier detection with z zscore
            z = np.abs(stats.zscore(health._get_numeric_data()))
            print(z)
            health = health[(z < 3).all(axis=1)]

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
    def Create_Decision_Tree_Model(health):
        # Decision Tree classification on depressive_predictor

        # set x to all features
        x = health.loc[:, health.columns != 'depressive_disorder_Yes']

        # set y to target depressive_disorder_Yes
        y = health.depressive_disorder_Yes

        # setup the test and training data.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
        decision_tree = tree.DecisionTreeClassifier(random_state=1, max_depth=15, min_samples_leaf=1,
                                                    class_weight={0: 100, 1: 0.5})
        decision_tree.fit(x_train, y_train)

        # DataController.tree_weight_testing(decision_tree, x, y)

        # test the model using cross validation
        print('decision_tree on depressive_predictor')
        DataController.model_testing(decision_tree, x, x_test, y, y_test)

        health.drop('depressive_disorder_Yes', axis=1, inplace=True)
        classes = ["depressive disorder", "no depressive disorder"]
        DataController.Create_Decision_Tree(decision_tree, health, classes, 'depressive disorder decision tree')

        return decision_tree

    @staticmethod
    def prep_dataset_for_type_2_diabetes_predictor(dataset):
        """" cleans the dataset for a type 2 diabetes predictor."""

        # drop responses that are not useful from depressive_disorder
        dataset.drop('age_of_diabetes_onset', axis=1, inplace=True)
        dataset.drop('education_level', axis=1, inplace=True)
        dataset.drop('own/rent_your_home', axis=1, inplace=True)
        dataset.drop('currently_employed', axis=1, inplace=True)
        dataset.drop('MARITAL', axis=1, inplace=True)
        dataset.drop('children', axis=1, inplace=True)
        dataset.drop('income', axis=1, inplace=True)

        dataset.drop(index=dataset[dataset['have_diabetes'] == "Don't know/Not Sure"].index, inplace=True)
        dataset.drop(index=dataset[dataset['have_diabetes'] == "Yes, but female told only during pregnancy"]
                     .index, inplace=True)

        # dataset.drop(index=dataset[dataset['have_diabetes'] == "Yes"].index, inplace=True)
        # dataset.drop(index=dataset[dataset['have_diabetes'] == "No"].index, inplace=True)

        # One Hot Encoding of categorical data
        categorical = [var for var in dataset.columns if dataset[var].dtype == 'object']

        # creates new columns for each of the categorical options.
        health = pd.get_dummies(dataset, columns=categorical)

        health.drop('have_diabetes_No', axis=1, inplace=True)
        # health.drop('have_diabetes_Yes', axis=1, inplace=True)

        return health

    @staticmethod
    def Create_ComplementNB_Model(health):
        # ComplementNB on type_2_diabetes_predictor

        # set x to all features
        x = health.loc[:, health.columns != 'have_diabetes_No, pre-diabetes or borderline diabetes']

        # set y to target have_diabetes_No, pre-diabetes or borderline diabetes
        y = health['have_diabetes_No, pre-diabetes or borderline diabetes']

        # setup the test and training data.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)

        naive_bayes = ComplementNB()

        naive_bayes.fit(x_train, y_train)
        # classes = ["pre-diabetes or borderline diabetes", "not pre-diabetic"]

        # test the model using cross validation
        print('naive_bayes on type_2_diabetes_predictor')
        DataController.model_testing(naive_bayes, x, x_test, y, y_test)

        return naive_bayes

    @staticmethod
    def prep_dataset_for_heart_disease_predictor(dataset):
        """" cleans the dataset for a heart disease."""

        # drop responses that are not useful from heart_disease
        dataset.drop(index=dataset[dataset['have_had_heart_attack/heart_disease']
                                   == "Not asked or Missing"].index, inplace=True)
        dataset.drop('education_level', axis=1, inplace=True)
        dataset.drop('own/rent_your_home', axis=1, inplace=True)
        dataset.drop('currently_employed', axis=1, inplace=True)
        dataset.drop('MARITAL', axis=1, inplace=True)
        dataset.drop('children', axis=1, inplace=True)
        dataset.drop('income', axis=1, inplace=True)

        # One Hot Encoding of categorical data
        categorical = [var for var in dataset.columns if dataset[var].dtype == 'object']

        # creates new columns for each of the categorical options.
        health = pd.get_dummies(dataset, columns=categorical)
        health.drop('have_had_heart_attack/heart_disease_Did not report having MI or CHD', axis=1, inplace=True)

        return health

    @staticmethod
    def Create_Forests_Model(health):
        # Forests on heart-disease

        # set x to all features
        x = health.loc[:, health.columns != 'have_had_heart_attack/heart_disease_Reported having MI or CHD']

        # set y to target have_had_heart_attack/heart_disease_Reported having MI or CHD
        y = health['have_had_heart_attack/heart_disease_Reported having MI or CHD']

        # setup the test and training data.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
        forest = imblearn.ensemble.EasyEnsembleClassifier(n_estimators=100, n_jobs=-1)
        forest.fit(x_train, y_train)
        classes = ["Reported MI/CHD", "no Reported MI/CHD"]

        # test the model using cross validation
        print('forest on heart-disease')
        DataController.model_testing(forest, x, x_test, y, y_test)

        return forest

    @staticmethod
    def model_testing(model, x, x_test, y, y_test):
        cross_val = sklearn.model_selection.StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        scores = sklearn.model_selection.cross_val_score(model, x, y, scoring='accuracy', cv=cross_val)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        scores = sklearn.model_selection.cross_val_score(model, x, y, scoring='roc_auc', cv=cross_val)
        print("%0.2f roc auc score " % scores.mean())
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(cm)
        # print vis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        # confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        rfc_disp = plot_roc_curve(model, x_test, y_test, ax=ax2)
        # plot two figs
        disp.plot(ax=ax1)
        rfc_disp.plot(ax=ax2)
        # show
        plt.show()

    @staticmethod
    def Create_Decision_Tree(model, health, classes, title):

        dot_data = tree.export_graphviz(model, out_file=None,
                                        feature_names=health.columns,
                                        class_names=classes,
                                        filled=True, rounded=True,
                                        special_characters=True,
                                        max_depth=6)
        graph = graphviz.Source(dot_data)
        graph.render(title)

    @staticmethod
    def confusion_matrix_scorer(clf, X, y):
        y_pred = clf.predict(X)
        cm = confusion_matrix(y, y_pred)
        return {'tn': cm[0, 0], 'fp': cm[0, 1],
                'fn': cm[1, 0], 'tp': cm[1, 1]}

    @staticmethod
    def tree_weight_testing(decision_tree, x, y):
        balance = ['balanced', {0: 8600, 1: .5}, {0: 8600, 1: .5}, {0: 8500, 1: .6}, {0: 5000, 1: .2}, {0: 100, 1: .6},
                   {0: 100, 1: .5},
                   {0: 100, 1: .3}]
        param_grid = dict(class_weight=balance)
        # define evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # define grid search
        grid = GridSearchCV(estimator=decision_tree, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
        # execute the grid search
        grid_result = grid.fit(x, y)
        # report the best configuration
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        # report all configurations
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
