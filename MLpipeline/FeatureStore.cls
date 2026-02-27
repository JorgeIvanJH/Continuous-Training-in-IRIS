Class MLpipeline.FeatureStore Extends %RegisteredObject
{

// Example of a constant parameter that never changes

Parameter ConstVar = "CONSTANT VAR FOR THE PROJECT";

// Purpose of the instantiation of FeatureStore

Property Action As %String;

Method %OnNew() As %Status [ Private ]
{
    do ##class(%SYS.System).WriteToConsoleLog("Created instance of MLpipeline.FeatureStore with ConstVar: "_..#ConstVar, 0, 0)
    Quit $$$OK
}

/// Description
Method DataExtraction(schema As %String, tablename As %String, columns As %String = "*", filters As %String = "") As %Status [ Language = python ]
{
    """
    Extracts data from IRIS database using SQL query constructed from the input parameters.
    e.g:
        set FS = ##class(MLpipeline.FeatureStore).%New()
        do FS.DataExtraction("MLpipeline", "PointSamples", "*", "datetime > '2026-02-26 17:50:00'")
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
        iris._SYS.System.WriteToConsoleLog(f"Executed query: {query} - Extracted {len(df)} rows.", 0, 0)
        return True

    except Exception as e:
        print(f"Extraction Error: {str(e)}")
        return False
}

// TODO: Implement processing pipelines, e.g. data cleaning, feature engineering, etc.

}
