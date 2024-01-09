import mastml
from mastml.mastml import Mastml
from mastml.datasets import LocalDatasets
from mastml.preprocessing import SklearnPreprocessor
from mastml.models import SklearnModel, EnsembleModel
from mastml.data_splitters import SklearnDataSplitter,NoSplit
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


model_rf = SklearnModel(model='RandomForestRegressor')
model_neighbor = EnsembleModel(model="KNeighborsRegressor", n_estimators=20, metric = "euclidean", 
        n_neighbors = 5,  weights = "distance")

def keras_model():
    model = Sequential()
    model.add(Dense(2048, input_dim=len(X.keys()), kernel_initializer='normal', activation='relu'))
    model.add(Dense(2048, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


Diffusion = "datasets/diffusion_data_selectfeatures.xlsx"
Perovskite = "datasets/Perovskite_70_Selected_Features.xlsx"
Supercond = "datasets/Supercon_data_features_selected.xlsx"
save_folder = "/home/mse10/vidit-work/FirstModelRuns"

model_dict = {model_rf: 'RandomForestRegressor', model_neighbor: 'NearestNeighbor', model_keras: 'KerasNetwork'}
dataset_dict = {Diffusion: 'Diffusion', Perovskite: 'Perovskite', Supercond: 'Superconductivity'}


for datapath in [Diffusion, Perovskite, Supercond]:

    if(datapath == Diffusion):
        target = 'E_regression'
        extra_columns = ['Material compositions 1', 'Material compositions 2']

    elif(datapath == Perovskite):
        target = 'EnergyAboveHull'
        extra_columns = ['Unnamed: 0']
    elif(datapath == Supercond):
        target = 'Tc'
        extra_columns = ['name', 'group', 'ln(Tc)']


    d = LocalDatasets(file_path=datapath,
                    target=target,
                    extra_columns=extra_columns,
                    testdata_columns=None,
                    as_frame=True)

    # Load the data with the load_data() method
    data_dict = d.load_data()



    # Let's assign each data object to its respective name
    X = data_dict['X']
    y = data_dict['y']
    X_extra = data_dict['X_extra']
    X_testdata = data_dict['X_testdata']

    preprocessor = SklearnPreprocessor(preprocessor='MinMaxScaler', as_frame=True)
    metrics = ['r2_score', 'mean_absolute_error', 'root_mean_squared_error', 'rmse_over_stdev']

    keras_regressor = KerasRegressor(build_fn=keras_model, epochs=100, batch_size=100, verbose=0)
    model_keras=EnsembleModel(model=keras_regressor, n_estimators = 20)

    for modelType in [model_rf, model_neighbor, model_keras]:
        for splitter_type in ['CV', 'NoSplit']:

            SAVEPATH = save_folder + "/data_{}_model_{}_{}".format(dataset_dict[datapath], model_dict[modelType],splitter_type)
            mastml = Mastml(savepath=SAVEPATH)
            savepath = mastml.get_savepath

            if splitter_type == 'CV':
                splitter = SklearnDataSplitter(splitter='RepeatedKFold', n_repeats=1, n_splits=5)
            elif splitter_type == 'NoSplit':
                splitter = NoSplit()

            #Running this model according to Palmer specifications
            splitter.evaluate(X=X, y=y, models=[modelType], mastml=mastml, preprocessor=preprocessor, metrics=metrics,
                            plots=['Error', 'Scatter', 'Histogram'],
                            parallel_run = True,
                            savepath=savepath,
                            X_extra=X_extra,
                            Nested_CV = True,
                            error_method='stdev_weak_learners',
                            recalibrate_errors=True,
                            remove_outlier_learners=True,
                            verbosity=3)
