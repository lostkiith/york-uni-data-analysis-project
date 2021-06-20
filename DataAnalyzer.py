from tkinter.filedialog import askopenfilename

from DataController import DataController


def DataAnalyzer():
    df = DataController.prep_data(DataController.convert_csv_to_json(open_file()))

    # prep the data for the logistic regression model to predict a depressive disorder
    depressive_predictor_data = df.copy()
    depressive_predictor_prep = DataController.prep_dataset_for_depressive_predictor(depressive_predictor_data)
    depressive_disorder_decision_tree = DataController.Create_Decision_Tree_Model(depressive_predictor_prep)

    # prep the data for the svm model to predict type 2 diabetes
    type_2_diabetes_predictor_data = df.copy()
    type_2_diabetes_predictor_prep = \
        DataController.prep_dataset_for_type_2_diabetes_predictor(type_2_diabetes_predictor_data)
    type_2_diabetes_predictor_svm = DataController.Create_SVM_Model(type_2_diabetes_predictor_prep)

    # prep the data for the forest model to predict heart disease
    heart_disease_predictor_data = df.copy()
    heart_disease_predictor_prep = DataController.prep_dataset_for_heart_disease(heart_disease_predictor_data)
    heart_disease_predictor_forest = DataController.Create_Forests_Model(heart_disease_predictor_prep)


def open_file():
    """Open a file for editing."""
    filepath = askopenfilename(
        filetypes=[("Text Files", "*.csv"), ("All Files", "*.*")]
    )
    if not filepath:
        return
    else:
        return filepath


DataAnalyzer()
