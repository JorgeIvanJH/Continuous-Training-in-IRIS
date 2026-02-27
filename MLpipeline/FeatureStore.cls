Class MLpipeline.FeatureStore Extends %Persistent
{

// DEFINED PROPERTIES FOR THE CT RUN (instantaneous)
Property WindowDays As %Integer; // Is set when instanting this FeatureStore for a CT run
// THESE HAS TO BE STORED IN THE CT RUN TABLE

/// Description
Method DataExtraction(schema As %String, tablename As %String, columns As %String = "*", filters As %String = "") As %Status [ Language = python ]
{
    """
    Extracts data from IRIS database using SQL query constructed from the input parameters.
    e.g:
        do ##class(MLpipeline.FeatureStore).DataExtraction("MLpipeline", "PointSamples", "*", "datetime > '2026-02-26 17:50:00'")
    """
    import iris
    import pandas as pd
    
    try:
        where_clause = f" WHERE {filters}" if filters else ""
        query = f"SELECT {columns} FROM {schema}.{tablename} {where_clause}"
        rs = iris.sql.exec(query)
        df = rs.dataframe()
        if df.empty:
            print("Warning: No data found for the given filters.")
            return False
        print(f"Extracted {len(df)} rows.")
        print(f"Columns: {list(df.columns)}")
        # TODO: Log extraction details to a monitoring system
        return True

    except Exception as e:
        print(f"Extraction Error: {str(e)}")
        return False
}

/// Description
Method LogMetadata() As %Status
{
    Set run.WindowDays = ..WindowDays



    Set sc = $$$OK
    // do something
    Return sc
}
// TODO: Implement processing pipelines, e.g. data cleaning, feature engineering, etc.


// TODO: Have Table: MLpipeline.CTRun Set run = ##class(MLpipeline.CTRun).%New()
    // Set run.WindowDays = ..WindowDays
    // Set run.Query = query
    // Set run.RowCount = rowcount
    // Do run.%Save()


    //     Field	Why
    // RunId	Unique identifier
    // CreatedAt	When it ran
    // FeatureVersion	Which feature logic version
    // WindowStart	Extraction window start
    // WindowEnd	Extraction window end
    // QueryHash	Reproducibility
    // RowCount	Sanity check
    // RandomSeed	Dataset split reproducibility
    // ModelVersion	Resulting model
    // Status	Success / Failed
}
