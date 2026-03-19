Class MLpipeline.PredictionService Extends %RegisteredObject
{

/*
Extracts data from IRIS for prediction based on the input filters. The filters should be a valid SQL WHERE clause without the 'WHERE' keyword.
e.g:
    set df = ##class(MLpipeline.PredictionService).PredDataExtraction("datetime > '2026-02-26 17:50:00'")
*/
ClassMethod PredDataExtraction(filters As %String = "") As %Status [ Language = python ]
{
    import iris
    import pandas as pd
    import datetime as dt
    from python_utils.utils import IRIS_DBQuery

    try:
        df = IRIS_DBQuery(
            schema="MLpipeline", 
            tablename="PointSamples", 
            columns="*", 
            filters=filters
        )
        iris._SYS.System.WriteToConsoleLog(f"DataExtraction completed.", 0, 0)
        return df

    except Exception as e:
        print(f"PredDataExtraction Error: {e}")
        iris._SYS.System.WriteToConsoleLog(f"PredDataExtraction Error: {e}", 0, 2)
        return None
}

/*
Loads the model and serves it for inference. 
Caches the model in memory and reloads it only when the path changes, as determined by a global variable.
e.g:
    set model = ##class(MLpipeline.PredictionService).CachedModelLoading()
*/
ClassMethod CachedModelLoading() As %SYS.Python [ Language = python ]
{
    import iris
    from python_utils.utils import safe_model_load

    global _cached_model, _current_path

    try:

        gref = iris.gref("^ML.Config")
        model_path = gref["ModelPath"]

        if '_cached_model' not in globals() or _current_path != model_path or not model_path:
            _cached_model = safe_model_load(model_path)
            _current_path = model_path

        iris._SYS.System.WriteToConsoleLog(f"Model loaded from {model_path}", 0, 0)
        return _cached_model
    except Exception as e:
        print(f"CachedModelLoading Error: {str(e)}")
        iris._SYS.System.WriteToConsoleLog(f"CachedModelLoading Error: {str(e)}", 0, 2)
        return None
}

/*
Uploads a CSV file to an IRIS table. It performs basic validation and type inference, including date parsing.
*/
ClassMethod SavePredictions(predictionlist As %List, namespace As %String = "USER", package As %String = "MLpipeline") As %Status [ Language = python ]
{
    from sqlalchemy import create_engine
    import pandas as pd
    import iris
    
    try:
        # Standard columns defined in your table creation
        cols = ['PointsIDForeignKey', 'YPred', 'Modelrunid', 'metrics']
        
        # Convert incoming data to DataFrame
        # Assumes predictionlist is a list of dicts: [{'PointsIDForeignKey': 1, 'YPred': 0.5, ...}, ...]
        # Or a list of lists: [[1, 0.5, 'run_1', '{}'], ...]
        df = pd.DataFrame(predictionlist)
        
        # If passed as a list of lists without keys, ensure column names match
        if not isinstance(predictionlist[0], dict):
            df.columns = cols
            
        engine = create_engine(f"iris+emb:///{namespace}")
        
        # Append data to the existing table
        df.to_sql("Predictions", engine, schema=package, if_exists='append', index=False, method='multi')
        
        iris._SYS.System.WriteToConsoleLog(f"Successfully saved {len(df)} predictions to {package}.Predictions", 0, 0)
        return True

    except Exception as e:
        print(f"SavePredictions Error: {e}")
        iris._SYS.System.WriteToConsoleLog(f"SavePredictions Error: {e}", 0, 2)
        return False
}

/*
Predicts over the data specified by the filters using the current model in production and saves the predictions back to IRIS. The filters should be a valid SQL WHERE clause without the 'WHERE' keyword.
e.g:
    set predictionsjson = ##class(MLpipeline.PredictionService).Predict("datetime > '2026-02-26 17:50:00'")
*/
ClassMethod Predict(filters As %String = "") As %SYS.Python [ Language = python ]
{
    import iris
    import mlflow
    import json
    import pandas as pd
    from python_utils.utils import measure_time_decorator

    try:

        df = iris.MLpipeline.PredictionService.PredDataExtraction(filters=filters)
        model = iris.MLpipeline.PredictionService.CachedModelLoading()
        X = df[["x"]]
        y_preds = model.predict(X)
        y_pred = pd.DataFrame(y_preds, columns=['y_pred'], index=X.index)

        iris.MLpipeline.PredictionService.SavePredictions(
            predictionlist=y_pred.reset_index().to_dict(orient='records')
        )

        return y_pred.to_json(orient='split')

    except Exception as e:
        err_msg = f"Predict Error: {str(e)}"
        print(f"Predict Error: {err_msg}")
        iris._SYS.System.WriteToConsoleLog(f"Predict Error: {err_msg}", 0, 2)
        return json.dumps({"error": err_msg})
}

}
