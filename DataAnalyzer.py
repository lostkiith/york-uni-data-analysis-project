from tkinter.filedialog import askopenfilename

from DataController import DataController


def DataAnalyzer():
    df = DataController.prep_data(DataController.convert_csv_to_json(open_file()))

    # prep the data for the logistic regression model to predict a depressive disorder
    depressive_predictor = DataController.prep_dataset_for_depressive_predictor(df)
    depressive_disorder_decision_tree = DataController.Create_Decision_Tree_Model(depressive_predictor)

    # prep the data for the k-mean cluster model
    type_2_diabetes_predictor = DataController.prep_dataset_for_type_2_diabetes_predictor(df)
    type_2_diabetes_predictor_svm = DataController.Create_SVM_Model(type_2_diabetes_predictor)


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
