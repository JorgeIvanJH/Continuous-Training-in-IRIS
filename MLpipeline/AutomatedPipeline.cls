Class MLpipeline.AutomatedPipeline Extends %RegisteredObject
{

Method %OnNew() As %Status [ Private ]
{
    do ##class(%SYS.System).WriteToConsoleLog("Created instance of MLpipeline.AutomatedPipeline", 0, 0)
    Quit $$$OK
}

/// Leverages the FeatureStore to extract data and transform it to pandas DataFrame.
/// e.g:
///   set AP = ##class(MLpipeline.AutomatedPipeline).%New()
///   set df = AP.DataExtraction("2026-02-26 17:50:00")
Method DataExtraction(datetime As %String) As %SYS.Python [ Language = python ]
{
    import iris
    import pandas as pd
    
    try:
        FS = iris.MLpipeline.FeatureStore._New()
        os_rs = FS.DataExtraction("MLpipeline", "PointSamples", "*", f"datetime > '{datetime}'")
        print("Data Extraction ResultSet:", os_rs)
        py_rs = iris.cls("%SYS.Python.SQLResultSet")._New(os_rs)
        df = py_rs.dataframe()
        shape = df.shape
        iris._SYS.System.WriteToConsoleLog(f"DataFrame ready. Shape: {shape}", 0, 0)
        if shape == (0, 0): # Warning for empty result
            iris._SYS.System.WriteToConsoleLog("Query returned an empty DataFrame.", 0, 1)
        return df
    except Exception as e:
        print("Error during Data Extraction:", e)
        iris._SYS.System.WriteToConsoleLog(f"Extraction Error: {str(e)}", 0, 2)
        return None
}

}
