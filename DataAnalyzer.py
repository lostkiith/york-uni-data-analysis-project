import matplotlib.pyplot as plt
import numpy as np
from DataController import DataController
from tkinter.filedialog import askopenfilename


def DataAnalyzer():
    df = DataController.prep_data(DataController.convert_csv_to_json(open_file()))

    # prep the data for the logistic regression model to predict a depressive disorder
    depressive_predictor = DataController.prep_dataset_for_depressive_predictor(df)
    depressive_disorder_logreg = DataController.Create_Logistic_Regression_Model(depressive_predictor)

    # prep the data for the k-mean cluster model
    k_mean_clustering = DataController.clean_dataset_for_k_mean_clustering(df)


def open_file():
    """Open a file for editing."""
    filepath = askopenfilename(
        filetypes=[("Text Files", "*.csv"), ("All Files", "*.*")]
    )
    if not filepath:
        return
    else:
        return filepath


def testing():
    df = DataController.prep_data(DataController.convert_csv_to_json(open_file()))
    health = DataController.prep_dataset_for_depressive_predictor(df)

    # Split data in to 80% train and 20% test using mask to select random rows
    msk = np.random.rand(len(health)) < 0.8
    train = health[msk]
    test = health[~msk]

    # Train data distribution
    plt.scatter(train.ENGINE_SIZE, train.CO2EMISSIONS, color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.show()

    # Use sklearn to model data
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    train_x = np.asanyarray(train[['ENGINE_SIZE']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])
    model.fit(train_x, train_y)
    # The coefficients
    print('Coefficients: ', model.coef_)
    print('Intercept: ', model.intercept_)

    # Plot fit line over data
    plt.scatter(train.ENGINE_SIZE, train.CO2EMISSIONS, color='blue')
    plt.plot(train_x, model.coef_[0][0] * train_x + model.intercept_[0], '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")

    # Evaluation with Mean Abs Error, Mean Sq Error, RMSE and R^2

    from sklearn.metrics import r2_score

    test_x = np.asanyarray(test[['ENGINE_SIZE']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])
    test_y_ = model.predict(test_x)

    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y, test_y_))

    # K Fold Cross-Validation
    print("K-Fold")

    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score

    model = LinearRegression()

    X, y = health['ENGINE_SIZE'], health['CO2EMISSIONS']

    X = X.values.reshape(-1, 1)
    y = y.values.reshape(-1, 1)

    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)
    print(scores)


DataAnalyzer()
