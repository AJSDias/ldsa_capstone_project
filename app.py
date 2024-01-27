import os
import json
import pickle
import joblib
import pandas as pd
from datetime import datetime
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, BooleanField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect

########################################
# Create Database to store "good" predictions

#DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')
DB = connect(os.environ.get('DATABASE_URL'))

class Prediction(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    proba = FloatField()
    pred = BooleanField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)

# End database stuff
########################################


########################################
# Create Database to store BAD requests

#ERROR_DB = connect(os.environ.get('ERROR_DATABASE') or 'sqlite:///error_log.db')
ERROR_DB = connect(os.environ.get('ERROR_DATABASE'))

class ErrorLog(Model):
    observation_id = TextField(unique=True)
    observation = TextField()
    error_message = TextField()

    class Meta:
        database = ERROR_DB

ERROR_DB.create_tables([ErrorLog], safe=True)

# End database stuff
########################################

def log_error(observation, error_msg):
    el = ErrorLog(
        observation_id=observation.get('observation_id', None),
        observation=json.dumps(observation),
        error_message=error_msg
    )

    try:
        el.save()
    except IntegrityError:
        ERROR_DB.rollback()

########################################
# Unpickle the previously-trained model

with open('columns_capstone_v2.json') as fh:
    columns = json.load(fh)

with open('pipeline_capstone_v2.pickle', 'rb') as fh:
    pipeline = joblib.load(fh)

with open('dtypes_capstone_v2.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)

@app.route('/should_search/', methods=['POST'])
def should_search():

    obs_dict = request.get_json()
    
    # test presence of 'observation_id' in request
    if "observation_id" not in obs_dict:
        error = "Field `observation_id` missing from request: {}".format(obs_dict)
        log_error(obs_dict, error)
        return {"observation_id":None, "error": error}

    # get ID of observation
    _id = obs_dict['observation_id']

    # test presence of 'Type' in request
    if "Type" not in obs_dict:
        error = "Field `Type` missing from request: {}".format(obs_dict)
        log_error(obs_dict, error)
        return {"observation_id": _id, "error": error}

    # test presence of 'Date' in request
    if "Date" not in obs_dict:
        error = "Field `Date` missing from request: {}".format(obs_dict)
        log_error(obs_dict, error)
        return {"observation_id": _id, "error": error}

    # get Date from observation
    date = obs_dict['Date']

    # test presence of 'Part of a policing operation' in request
    if "Part of a policing operation" not in obs_dict:
        error = "Field `Part of a policing operation` missing from request: {}".format(obs_dict)
        log_error(obs_dict, error)
        return {"observation_id": _id, "error": error}

    # get Part of a Policing Operation from observation
    policing_operation = obs_dict['Part of a policing operation']

    # test presence of 'Latitude' in request
    if "Latitude" not in obs_dict:
        error = "Field `Latitude` missing from request: {}".format(obs_dict)
        log_error(obs_dict, error)
        return {"observation_id": _id, "error": error}

    # get Latitude from observation
    latitude = obs_dict['Latitude']

    # test presence of 'Longitude' in request
    if "Longitude" not in obs_dict:
        error = "Field `Longitude` missing from request: {}".format(obs_dict)
        log_error(obs_dict, error)
        return {"observation_id": _id, "error": error}

    # get Longitude from observation
    longitude = obs_dict['Longitude']

    # test presence of 'Gender' in request
    if "Gender" not in obs_dict:
        error = "Field `Gender` missing from request: {}".format(obs_dict)
        log_error(obs_dict, error)
        return {"observation_id": _id, "error": error}

    # get Gender from observation
    gender = obs_dict['Gender']


    # test presence of 'Age range' in request
    if "Age range" not in obs_dict:
        error = "Field `Age range` missing from request: {}".format(obs_dict)
        log_error(obs_dict, error)
        return {"observation_id": _id, "error": error}

    # get Age range from observation
    age = obs_dict['Age range']

    # test presence of 'Ethinicity' in request
    if "Officer-defined ethnicity" not in obs_dict:
        error = "Field `Officer-defined ethnicity` missing from request: {}".format(obs_dict)
        log_error(obs_dict, error)
        return {"observation_id": _id, "error": error}

    # get Ehtnicity from observation
    ethnicity = obs_dict['Officer-defined ethnicity']

    # test presence of 'Legislation' in request
    if "Legislation" not in obs_dict:
        error = "Field `Legislation` missing from request: {}".format(obs_dict)
        log_error(obs_dict, error)
        return {"observation_id": _id, "error": error}

    # get Legislation from observation
    legislation = obs_dict['Legislation']


    # test presence of 'Object of search' in request
    if "Object of search" not in obs_dict:
        error = "Field `Object of search` missing from request: {}".format(obs_dict)
        log_error(obs_dict, error)
        return {"observation_id": _id, "error": error}

    # get Object of search from observation
    object_search = obs_dict['Object of search']
    
    # test presence of 'station' in request
    if "station" not in obs_dict:
        error = "Field `station` missing from request: {}".format(obs_dict)
        log_error(obs_dict, error)
        return {"observation_id": _id, "error": error}

    # get Station from observation
    station_search = obs_dict['station']

    # get data from observation
    observation = {k: v for k, v in obs_dict.items() if k != 'observation_id'}

    #check that there are no invalid columns in request
    valid_columns = ['observation_id','Type','Date','Part of a policing operation',
                     'Latitude','Longitude','Gender','Age range','Officer-defined ethnicity',
                     'Legislation','Object of search','station']
    observation_columns = obs_dict.keys()
    #print ('obs_cols:',observation_columns)
    #print ('cols_pickle:',valid_columns)

    for column in observation_columns:
        if column not in valid_columns:
            error = "Unrecognized columns provided: {}".format(column)
            log_error(obs_dict, error)
            return {"observation_id": _id, "error": error}

    # # create aditional Date features
    time = pd.to_datetime(date)
    hour = time.hour
    month = time.month
    day_of_week = time.day_name()

    obs_dict['hour']=hour
    obs_dict['month']=month
    obs_dict['day_of_week']=day_of_week

    # make prediction
    obs = pd.DataFrame([obs_dict], columns=columns).astype(dtypes)
    proba = pipeline.predict_proba(obs)[0, 1]
    prediction = pipeline.predict(obs)[0]
    response = {'outcome': bool(prediction)}
    #response = {'outcome': bool(prediction),'probability':str(proba)}
    p = Prediction(
                    observation_id=_id,
                    proba=proba,
                    observation=observation,
                    pred=bool(prediction)
                   )
    try:
        p.save()
    except IntegrityError:
        error_msg = "ERROR: Observation ID: '{}' already exists".format(_id)
        response["error"] = error_msg
        print(error_msg)
        DB.rollback()
        log_error(obs_dict, error_msg)
    return jsonify(response)


@app.route('/search_result/', methods=['POST'])
def search_result():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['observation_id'])
        p.true_class = obs['outcome'] # actual outcome of search
        p.save()
        return jsonify({
                        'observation_id': p.observation_id,
                        'outcome': p.true_class,
                        'predicted_outcome': p.pred
                        })

    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['observation_id'])
        return jsonify({'error': error_msg})


@app.route('/list-db-contents/')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])


# End webserver stuff
########################################
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)