import datetime
import os

from azure.storage.blob import ContainerClient
import pandas as pd
import numpy as np
from sklearn import linear_model

key = os.environ.get('STORAGE_ACCOUNT_KEY')
container_url = 'https://asnrocks.blob.core.windows.net/auto'

start_time = datetime.datetime.now()

model_path = 'models/'

def get_model(model_path, container_url, key):
    ''' List all models and downloads last one, assumes order by name
        import dowloaded model
    '''
    # Instanciated a container client
    container_client = ContainerClient.from_container_url(
                                        container_url=container_url,
                                        credential=key)
    # List all files on models_path
    models_list = []
    for model in container_client.list_blobs(
                                        name_starts_with=model_path):
        models_list.append(model['name'])
    # Sort all models
    models_list.sort()
    last_model_name = models_list[-1]
    # Instaciate a blob cliente
    blob_client = container_client.get_blob_client(blob=last_model_name)
    # Download blob to local disk
    with open(last_model_name, 'wb') as my_file:
        blob_data = blob_client.download_blob()
        blob_data.readinto(my_file)
    # Close handlers
    blob_client.close()
    container_client.close()
    # Read model from local disk
    model = pd.read_pickle(last_model_name)
    return model, last_model_name

def get_score(params_dict, model_package):
    ''' Receiva all features in a dic,
        Predicts score and return it
    '''
    # Extract from package
    model = model_package.model
    columns = model_package.fit_vars
    
    # Check if calculated feature is used by model, if yes calc it
    if 'cylinder_displacement' in columns:
        params_dict['cylinder_displacement'] = (params_dict['cylinders'] 
                                          * params_dict['displacement'])
    if 'specific_torque' in columns:
        params_dict['specific_torque'] = (params_dict['horsepower'] 
                                 / params_dict['cylinder_displacement'])
    if 'fake_torque' in columns:
        params_dict['fake_torque'] = (params_dict['weight'] 
                                         / params_dict['acceleration'])
    # Predict score
    score = model.predict(pd.DataFrame.from_dict({1:params_dict}, 
                                                        orient='index'))
    return score

if __name__ == '__main__':
    # create test case
    input_dict = {'cylinders': 8,
              'displacement': 320,
              'horsepower': 150,
              'weight': 3449,
              'acceleration': 11.0,
              'year': 70,
             'origin': 1}

    model_package, model_version = get_model(model_path, container_url, 
                                             key)
    
    score = get_score(input_dict, model_package)

    elapsed_time = datetime.datetime.now() - start_time

    results_dict = {
            'Start time': start_time.strftime('%Y-%m-%d--%H-%M-%S'),
            'model version': model_version,
            'input data': input_dict,
            'predicted score': score[0],
            'scoring time': elapsed_time.total_seconds()}
    print(results_dict)
