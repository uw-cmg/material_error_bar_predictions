import mastml
from mastml.mastml import Mastml
from mastml.datasets import LocalDatasets
from mastml.preprocessing import SklearnPreprocessor
from mastml.models import SklearnModel
from mastml.data_splitters import SklearnDataSplitter
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping


data_folder = "/home/mse10/vidit-work/AugmentedDatasets"

model_rf = SklearnModel(model='RandomForestRegressor')

def keras_model():
    model = Sequential()
    model.add(Dense(2048, input_dim=len(X.keys()), kernel_initializer='normal', activation='relu'))
    model.add(Dense(2048, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

keras_regressor = KerasRegressor(build_fn=keras_model, epochs=300, batch_size=100, verbose=0)

model_dict = {'model_neighbor':'NearestNeighbor','model_keras':'KerasNetwork','model_rf':'RandomForestRegressor'}
Datasets = ["replace_dataset"]
points = [0, 2000, 5000, 10000, 20000, 40000, 50000, 80000, 100000]
Randomization = ["replace_random"]

save_folder = "/home/mse10/vidit-work/SecondModelRuns"

for model in ['replace_model']:
    for data in Datasets:
        for point in points:
            for rand in Randomization:
                data_path = data_folder + "/data_{}_model_{}_points_{}_randomization_{}".format(data, model, point, rand) + ".xlsx"
                if not os.path.exists(data_path):
                    print(data_path+"--does not exist;")
                    continue
                target = 'y_err'
                extra_columns = ["Unnamed: 0", "y_pred"]


                SAVEPATH = save_folder + "/data_{}_model_{}_points_{}_randomization_{}".format(data, model, point, rand)
                mastml = Mastml(savepath=SAVEPATH)
                savepath = mastml.get_savepath

                d = LocalDatasets(file_path=data_path, 
                target=target,
                extra_columns=extra_columns, 
                testdata_columns=None,
                as_frame=True)

                data_dict = d.load_data()

                X = data_dict['X']
                y = data_dict['y']
                X_extra = data_dict['X_extra']
                X_testdata = data_dict['X_testdata']

                preprocessor = SklearnPreprocessor(preprocessor='MinMaxScaler', as_frame=True)
                metrics = ['r2_score', 'mean_absolute_error', 'root_mean_squared_error', 'rmse_over_stdev']

                splitter = SklearnDataSplitter(splitter='RepeatedKFold', n_repeats=1, n_splits=5)
                splitter.evaluate(X=X, y=y, remove_split_dirs = True, models = [keras_regressor], parallel_run = True, metrics = metrics, preprocessor= preprocessor, plots=['Histogram', 'Scatter'], savepath= savepath, verbosity = 1)
                print("-------- data_{}_model_{}_points_{}_randomization_{}".format(data, model, point, rand),"Completed ------")


