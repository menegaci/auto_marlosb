import datetime
import os

from azure.storage.blob import ContainerClient
import feature_engine.missing_data_imputers as mdi
from feature_engine import categorical_encoders as ce
from feature_engine import variable_transformers as vt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import linear_model
from sklearn import model_selection

key = os.environ.get('STORAGE_ACCOUNT_KEY')
container_url = 'https://asnrocks.blob.core.windows.net/auto'

start_time = datetime.datetime.now()

data_path = 'data/'
data_file = 'auto.xlsx'

model_path = 'models/'
model_name = (
       f'auto_model_{start_time.strftime("%Y-%m-%d--%H-%M-%S")}.pickle')

def read_input_data(data_path, data_file, container_url, key):
    '''Downloads specified data file from specified container,
       import this data to a pandas DataFrame an returns it.
       
       Data file must be Excel file,
       container URL and key must be from Azure Blob Storage.
    '''
    # Instanciated a container client
    container_client = ContainerClient.from_container_url(
                                        container_url=container_url,
                                        credential=key)
    # Create a handle to the blob (the file)
    blob_client = container_client.get_blob_client(blob=data_path 
                                                        + data_file)
    # Download blob to local disk
    with open(data_path + data_file, 'wb') as my_file:
        blob_data = blob_client.download_blob()
        blob_data.readinto(my_file)
    # close used handles
    blob_client.close()
    container_client.close()
    # Read file to pandas DataFrame
    dataframe = pd.read_excel(data_path + data_file)
    return dataframe 

def save_model(model, model_path, model_name, container_url, key):
    '''First, serialize model object to pickle and save to disk,
       then uploads it to cloud.
    '''
    # Save model to disk
    model.to_pickle(model_path + model_name)
    # Instanciated a container client
    container_client = ContainerClient.from_container_url(
                                        container_url=container_url,
                                        credential=key)
    # Create a handle to the blob (the file)
    blob_client = container_client.get_blob_client(blob=model_path 
                                                        + model_name)
    # First checks if a file with same names already exists
    if not blob_client.exists():
        with open(model_path + model_name, 'rb') as my_file:
            blob_client.upload_blob(my_file)
    # close used handles
    container_client.close()
    blob_client.close()
    # print confirmation message
    print('Upload completed!')

def run_train(data_path, data_file, model_path, model_name,
              container_url, key):
    '''Run the complete model training flow
       Downloads and imports data from cloud, 
       run all transformation pipeline,
       fit model and save it to cloud.
    '''
    # download and import input data
    auto_df = read_input_data(data_path, data_file, container_url, key)
    # create calculated features
    auto_df['cylinder_displacement'] = (auto_df['cylinders'] 
                                              * auto_df['displacement'])
    auto_df['specific_torque'] = (auto_df['horsepower'] 
                                     / auto_df['cylinder_displacement'])
    auto_df['fake_torque'] = auto_df['weight'] / auto_df['acceleration']
    # define feature types
    target = 'mpg' # Milhas por galão
    num_vars = ['cylinders', 'displacement', 'horsepower', 'weight', 
                'acceleration', 'year', 'cylinder_displacement', 
                'specific_torque', 'fake_torque']
    cat_vars = ['origin']
    auto_df[cat_vars] = auto_df[cat_vars].astype(str)
    # split train and test subsets
    X_train, X_test, y_train, y_test = model_selection.train_test_split( 
                                            auto_df[num_vars+cat_vars],
                                            auto_df[target],
                                            random_state=1992,
                                            test_size=0.25)
    # train model
    ## Define o transformador do transformação logaritmica
    log = vt.LogTransformer(variables=num_vars) 
    onehot = ce.OneHotCategoricalEncoder(variables=cat_vars,
                                         drop_last=True) # Cria Dummys
    model = linear_model.Lasso() # Definição do modelo

    full_pipeline = Pipeline( steps=[
        ("log", log),
        ("onehot", onehot),
        ('model', model) ] )

    param_grid = { 'model__alpha':[0.0167, 0.0001, 0.001, 0.01, 0.1, 
                                   0.2, 0.5, 0.8, 1], # linspace
                   'model__normalize':[True],
                   'model__random_state':[1992]}

    search = model_selection.GridSearchCV(full_pipeline,
                                param_grid,
                                cv=5,
                                n_jobs=-1,
                                scoring='neg_root_mean_squared_error')

    search.fit(X_train, y_train) # Executa o treinamento!!

    best_model = search.best_estimator_
    # save test results
    cv_result = pd.DataFrame(search.cv_results_) # Pega resultdos do grid
    cv_result = cv_result.sort_values(by='mean_test_score', 
                                         ascending = False)
    # Assess model performance
    y_test_pred = best_model.predict(X_test)
    root_mean_squadred_erro = (
              metrics.mean_squared_error( y_test, y_test_pred) ** (1/2))
    print( "Raiz do Erro Quadrático Médio:", root_mean_squadred_erro)
    # Fit for all data
    best_model.fit( auto_df[num_vars+cat_vars], auto_df[target] )
    # create pandas Series model package
    model_s = pd.Series( {"cat_vars":cat_vars,
                      "num_vars":num_vars,
                      "fit_vars": X_train.columns.tolist(),
                      "model":best_model,
                      "rmse":root_mean_squadred_erro} )
    # save model
    save_model(model_s, model_path, model_name, container_url, key)

if __name__ == '__main__':
    run_train(data_path, data_file, model_path, model_name,
              container_url, key)
    elapsed_time = datetime.datetime.now() - start_time

    print(f'full process completed in {elapsed_time.total_seconds()} \
          seconds')