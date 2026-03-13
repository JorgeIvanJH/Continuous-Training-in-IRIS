Class MLpipeline.ModelServing Extends %RegisteredObject
{

/*
Loads the model and serves it for inference. 
Caches the model in memory and reloads it only when the path changes, as determined by a global variable.
e.g:
    do ##class(MLpipeline.ModelServing).Predict(X)
*/
ClassMethod Predict(X As %SYS.Python) As %SYS.Python [ Language = python ]
{
    import iris
    import mlflow
    global _cached_model, _current_path

    try:
        gref = iris.gref("^ML.Config")
        new_model_path = gref["ModelPath"]

        if not new_model_path:
            iris._SYS.System.WriteToConsoleLog("No model path found in ^ML.Config('ModelPath')", 0, 2)
            raise Exception("No model path found in ^ML.Config('ModelPath')")

        # Load the model if it's not cached or if the path has changed
        if '_cached_model' not in globals() or _current_path != new_model_path:
            _cached_model = mlflow.sklearn.load_model(new_model_path)
            _current_path = new_model_path
            iris._SYS.System.WriteToConsoleLog(f"New model loaded from {new_model_path}", 0, 0)

        iris._SYS.System.WriteToConsoleLog(f"Predicting with model from {new_model_path}", 0, 0)
        return _cached_model.predict(X)
    except Exception as e:
        print(f"Predict Error: {str(e)}")
        iris._SYS.System.WriteToConsoleLog(f"Predict Error: {str(e)}", 0, 2)
        return None
}

}
