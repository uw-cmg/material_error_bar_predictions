"""
This module contains methods for easily making new predictions on test data once a suitable model has been trained. Also
available is output of calibrated uncertainties for each prediction.
make_prediction:
    Method used to take a saved preprocessor, model and calibration file and output predictions and calibrated uncertainties
    on new test data.
"""

import pandas as pd
import joblib
import numpy as np
import os
from mastml import feature_generators

def make_prediction(X_test, model, X_test_extra=None, preprocessor=None, calibration_file=None, featurize=False,
                    featurizer=None, features_to_keep=None, featurize_on=None, **kwargs):
    '''
    Method used to take a saved preprocessor, model and calibration file and output predictions and calibrated uncertainties
    on new test data
    Args:
        X_test: (pd.DataFrame or str), dataframe of featurized test data to be used to make prediction, or string of path
            containing featurized test data in .xlsx or .csv format ready for import with pandas. Only the features used
            to fit the original model should be included, and they should be in the same order as the training data used
            to fit the original model.
        model: (str), path of saved model in .pkl format (e.g., RandomForestRegressor.pkl)
        X_test_extra: (list or str), list of strings denoting extra columns present in X_test not to be used in prediction.
            If a string is provided, it is interpreted as a path to a .xlsx or .csv file containing the extra column data
        preprocessor: (str), path of saved preprocessor in .pkl format (e.g., StandardScaler.pkl)
        calibration_file: path of file containing the recalibration parameters (typically recalibration_parameters_average_test.xlsx)
        featurize: (bool), whether or not featurization of the provided X_test data needs to be performed
        featurizer: (str), string denoting a mastml.feature_generators class, e.g., "ElementalFeatureGenerator"
        features_to_keep: (list), list of strings denoting column names of features to keep for running model prediction
        featurize_on: (str), string of column name in X_test to perform featurization on
        **kwargs: additional key-value pairs of parameters for feature generator, e.g., composition_df=composition_df['Compositions'] if
            running ElementalFeatureGenerator
    Returns:
        pred_df: (pd.DataFrame), dataframe containing column of model predictions (y_pred) and, if applicable, calibrated uncertainties (y_err).
            Will also include any extra columns denoted in extra_columns parameter.
    '''

    # Load model:
    model = joblib.load(model)

    # Check if recalibration params exist:
    if calibration_file is not None:
        if '.xlsx' in calibration_file:
            recal_params = pd.read_excel(calibration_file, engine='openpyxl')
        elif '.csv' in calibration_file:
            recal_params = pd.read_csv(calibration_file)
        else:
            raise ValueError('calibration_file should be either a .csv or .xlsx file to be loaded using pandas')
    else:
         recal_params = None

    if isinstance(X_test, str):
        if '.xlsx' in X_test:
            X_test = pd.read_excel(X_test, engine='openpyxl')
        elif '.csv' in X_test:
            X_test = pd.read_csv(X_test)
        else:
            raise ValueError('You must provide X_test as .xlsx or .csv file, or loaded pandas DataFrame')

    if X_test_extra is not None:
        if isinstance(X_test_extra, str):
            if '.xlsx' in X_test_extra:
                df_extra = pd.read_excel(X_test_extra, engine='openpyxl')
            elif '.csv' in X_test_extra:
                df_extra = pd.read_csv(X_test_extra)
        elif isinstance(X_test_extra, list):
            df_extra = X_test[X_test_extra]
            X_test = X_test.drop(X_test_extra, axis=1)

    if featurize == False:
        df_test = X_test
    else:
        featurizer = getattr(feature_generators, featurizer)(**kwargs)
        df_test, _ = featurizer.fit_transform(X_test[featurize_on])
        df_test = df_test[features_to_keep]

    # Load preprocessor
    if preprocessor is not None:
        preprocessor = joblib.load(preprocessor)
        df_test = preprocessor.transform(df_test)

    # Check the model is an ensemble and get an error bar:
    ensemble_models = ['RandomForestRegressor','GradientBoostingRegressor','BaggingRegressor','ExtraTreesRegressor','AdaBoostRegressor']
    try:
        model_name = model.model.__class__.__name__
    except:
        model_name = model.__class__
    yerr = list()    
    if model_name in ensemble_models:
        X_aslist = df_test.values.tolist()
        for x in range(len(X_aslist)):
            preds = list()
            if model_name == 'RandomForestRegressor':
                for pred in model.model.estimators_:
                    preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
            elif model_name == 'BaggingRegressor':
                for pred in model.model.estimators_:
                    preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
            elif model_name == 'ExtraTreesRegressor':
                for pred in model.model.estimators_:
                    preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
            elif model_name == 'GradientBoostingRegressor':
                for pred in model.model.estimators_.tolist():
                    preds.append(pred[0].predict(np.array(X_aslist[x]).reshape(1, -1))[0])
            elif model_name == 'AdaBoostRegressor':
                for pred in model.model.estimators_:
                    preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
            if recal_params is not None:
                yerr.append(recal_params['a'][0]*np.std(preds)+recal_params['b'][0])
            else:            
                yerr.append(np.std(preds))

    if model_name == 'GaussianProcessRegressor':
        y_pred_new, yerr = model.model.predict(df_test, return_std=True)
    else:
        y_pred_new = model.predict(df_test)

    if len(yerr) > 0:
        pred_df = pd.DataFrame(y_pred_new, columns=['y_pred'])
        pred_df['y_err'] = yerr
    else:
        pred_df = pd.DataFrame(y_pred_new, columns=['y_pred'])

    if X_test_extra is not None:
        for col in df_extra.columns.tolist():
            pred_df[col] = df_extra[col]

    return pred_df

def keras_model():
    model = Sequential()
    model.add(Dense(2048, input_dim=len(Original_Data.keys()), kernel_initializer='normal', activation='relu'))
    model.add(Dense(2048, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


import mastml
from mastml.mastml import Mastml
from mastml.datasets import LocalDatasets
from mastml.preprocessing import SklearnPreprocessor
from mastml.models import SklearnModel
from mastml.data_splitters import SklearnDataSplitter
import os
import random
import tensorflow as tf
from mastml.models import EnsembleModel
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np

predict_folder = "/home/mse10/vidit-work/FirstModelRuns"

Models = ["replace_model"]
Datasets = ["replace_dataset"]
Points = [0, 2000, 5000, 10000, 20000, 40000, 50000, 80000, 100000]
Randomization = ["replace_random"]
distance = 0

Diffusion = "/home/mse10/vidit-work/orig_datasets/diffusion_data_selectfeatures.xlsx"
Perovskite = "/home/mse10/vidit-work/orig_datasets/Perovskite_70_Selected_Features.xlsx"
Supercond = "/home/mse10/vidit-work/orig_datasets/Supercon_data_features_selected.xlsx"
orig_dataset_dict = {'Diffusion': Diffusion, 'Perovskite': Perovskite, 'Superconductivity': Supercond}

save_folder = "/home/mse10/vidit-work/AugmentedDatasets"


for model in Models:
    for data in Datasets:
        for point in Points:
            for rand in Randomization:

                # Calibration file path
                calibration_predict_path = predict_folder + "/data_" + data + "_model_" + model + "_CV"
                dir_list = os.listdir(calibration_predict_path)
                dir_list.remove('mastml_metadata.json')
                if len(dir_list) > 1:
                    dir_list.remove('.DS_Store')
                calibration_predict_path = calibration_predict_path + "/" + dir_list[0]
                CalibrationFilePath = calibration_predict_path + "/recalibration_parameters_average_test.csv"

                # Model file path and preprocessor path
                predict_path = predict_folder + "/data_" + data + "_model_" + model + "_NoSplit"
                dir_list = os.listdir(predict_path)
                dir_list.remove('mastml_metadata.json')
                if len(dir_list) > 1:
                    dir_list.remove('.DS_Store')
                predict_path = predict_path + "/" + dir_list[0]

                ModelFilePath = predict_path
                PreprocessorFilePath = os.path.join(predict_path, "MinMaxScaler.pkl")


                if(data == 'Diffusion'):
                    extra_columns = ['Material compositions 1', 'Material compositions 2', 'E_regression']

                elif(data == 'Perovskite'):
                    extra_columns = ['Unnamed: 0', 'EnergyAboveHull']
                elif(data == 'Superconductivity'):
                    extra_columns = ['name', 'group', 'ln(Tc)', 'Tc']

                Original_Data = pd.read_excel(orig_dataset_dict[data])
                Original_Data = Original_Data.drop(extra_columns, axis=1)


                datapath = save_folder + "/data_{}_points_{}_random_{}.xlsx".format(data, point, rand)
                if(model == "RandomForestRegressor"):
                    ModelFilePath = ModelFilePath + "/RandomForestRegressor.pkl"
                elif(model == "GaussianProcessRegressor"):
                    ModelFilePath = ModelFilePath + "/BaggingRegressor.pkl"
                elif(model == "NeuralNetwork"):
                    ModelFilePath = ModelFilePath + "/BaggingRegressor.pkl"
                elif(model == "NearestNeighbor"):
                    ModelFilePath = ModelFilePath + "/split_0/BaggingRegressor.pkl"
                elif(model == "KerasNetwork"):

                    model_keras = KerasRegressor(build_fn=keras_model, epochs=100, batch_size=100, verbose=0)
                    model_bagged_keras_rebuild = EnsembleModel(model=model_keras, n_estimators=20)
                    num_models = 20
                    models = list()
                    for i in range(num_models):
                        models.append(tf.keras.models.load_model(os.path.join(ModelFilePath,'keras_model_'+str(i))))
                    model_bagged_keras_rebuild.model.estimators_ = models
                    model_bagged_keras_rebuild.model.estimators_features_ = [np.arange(0,Original_Data.shape[1]) for i in models]


                Augmented_Data = [[]]

                if rand == "0.1":
                    distance = 0.1
                elif rand == "0.2":
                    distance = 0.2
                elif rand == "0.3":
                    distance = 0.3
                elif rand == "0.4":
                    distance = 0.4
                elif rand == "0.5":
                    distance = 0.5
                elif rand == "0.01":
                    distance = 0.01
                elif rand == "0.001":
                    distance = 0.001
                Augmented_Data = []
                cols_count = 0
                if distance != 0:
                    for i in range(point):
                        randRow = Original_Data.iloc[random.randint(0,len(Original_Data)-1)]
                        new_row =[]
                        for key in randRow.keys():
                            col_range = max(Original_Data[key])-min(Original_Data[key])
                            col_value = distance*col_range
                            val = randRow[key]
                            random_val = random.uniform(val-col_value, val+col_value)
                            if(random_val < min(Original_Data[key])):
                                random_val = min(Original_Data[key])
                            if(random_val > max(Original_Data[key])):
                                random_val = max(Original_Data[key])
                            new_row.append(random_val)


                        Augmented_Data.append(new_row)
                    Augmented_Data_df = pd.DataFrame(Augmented_Data,columns=Original_Data.keys())
                    distance = 0
                elif rand == "ColumnRange":
                    Augmented_Data = pd.DataFrame()
                    for key in Original_Data.keys():
                        key_min = min(Original_Data[key])
                        key_max = max(Original_Data[key])
                        col = []
                        for i in range(point):
                            if Original_Data[key].dtypes == 'int64':
                                col.append(random.randint(key_min, key_max))
                            else:
                                rnd = random.uniform(0,1)
                                col.append(key_min+(key_max-key_min)*rnd)
                        Augmented_Data[key] = col
                    Augmented_Data_df = pd.DataFrame(Augmented_Data, columns=Original_Data.columns)

                if model == 'KerasNetwork':
                    print("right loop!")
                    # preprocess Data
                    # preprocessor = SklearnPreprocessor(preprocessor='MinMaxScaler', as_frame=True)
                    # X = preprocessor.evaluate(X=Original_Data)

                    preprocessor = joblib.load(PreprocessorFilePath)
                    X = preprocessor.transform(Original_Data)
                    
                    # recalibrate errors
                    slope = pd.read_csv(CalibrationFilePath)['a'].values[0]
                    intercept = pd.read_csv(CalibrationFilePath)['b'].values[0]

                    # reloaded_model = joblib.load(ModelFilePath)
                    Original_Data_pred = {'y_pred': [], 'y_err':[]}
                    preds_each = list()
                    ebars_each = list()
                    for i, x in X.iterrows():
                        preds_per_data = list()
                        for m in model_bagged_keras_rebuild.model.estimators_:
                            preds_per_data.append(m.predict(pd.DataFrame(x).T))
                            tf.keras.backend.clear_session()
                        Original_Data_pred['y_err'].append(np.std(preds_per_data)*slope+intercept)
                        Original_Data_pred['y_pred'].append(np.mean(preds_per_data))
                    Original_Data_pred = pd.DataFrame(Original_Data_pred)
                    Original_Data = pd.concat([Original_Data, Original_Data_pred], axis=1)

                    if point == 0:
                        SAVEPATH = save_folder + "/data_{}_predmodel_{}_points_{}_randomization_{}".format(data, model, str(point), rand) + ".xlsx"
                        Original_Data.to_excel(SAVEPATH)
                    else:
                        preprocessor = preprocessor = joblib.load(PreprocessorFilePath)
                        AugmentedX = preprocessor.evaluate(X=Augmented_Data_df)
                        Augmented_Data_df_predict = {'y_pred':[], 'y_err':[]}

                        for i, x in AugmentedX.iterrows():
                            preds_per_data = list()
                            for m in model_bagged_keras_rebuild.model.estimators_:
                                preds_per_data.append(m.predict(pd.DataFrame(x).T))
                                tf.keras.backend.clear_session()
                            Augmented_Data_df_predict['y_err'].append(np.std(preds_per_data)*slope+intercept)
                            Augmented_Data_df_predict['y_pred'].append(np.mean(preds_per_data))
                        Augmented_Data_df_predict = pd.DataFrame(Augmented_Data_df_predict)
                        frames = [Augmented_Data_df, Augmented_Data_df_predict]
                        Augmented_Data_df = pd.concat(frames, axis=1)
                        Final_Data = pd.concat([Original_Data, Augmented_Data_df], axis=0)
                        SAVEPATH = save_folder + "/data_{}_predmodel_{}_points_{}_randomization_{}".format(data, model, str(point), rand) + ".xlsx"
                        Final_Data.to_excel(SAVEPATH)

                else:
                    if(point == 0):
                        Original_Data_pred = make_prediction(Original_Data, ModelFilePath, calibration_file = CalibrationFilePath, preprocessor = PreprocessorFilePath)
                        Original_Data = pd.concat([Original_Data, Original_Data_pred], axis=1)
                        SAVEPATH = save_folder + "/data_{}_predmodel_{}_points_{}_randomization_{}".format(data, model, str(point), rand) + ".xlsx"
                        Original_Data.to_excel(SAVEPATH)
                    else:
                        Original_Data_pred = make_prediction(Original_Data, ModelFilePath, calibration_file = CalibrationFilePath, preprocessor = PreprocessorFilePath)
                        Original_Data = pd.concat([Original_Data, Original_Data_pred], axis=1)
                        Augmented_Data_df_predict = make_prediction(Augmented_Data_df, ModelFilePath, calibration_file = CalibrationFilePath, preprocessor = PreprocessorFilePath)
                        frames = [Augmented_Data_df, Augmented_Data_df_predict]
                        Augmented_Data_df = pd.concat(frames, axis=1)
                        Final_Data = pd.concat([Original_Data, Augmented_Data_df], axis=0)
                        SAVEPATH = save_folder + "/data_{}_predmodel_{}_points_{}_randomization_{}".format(data, model, str(point), rand) + ".xlsx"
                        Final_Data.to_excel(SAVEPATH)








