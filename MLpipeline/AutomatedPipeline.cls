Class MLpipeline.AutomatedPipeline Extends %RegisteredObject
{

/// Description
ClassMethod DataExtraction(schema As %String, tablename As %String, columns As %String = "*", filters As %String = "") As %Status [ Language = python ]
{
    """
    Extracts data from IRIS database using SQL query constructed from the input parameters.
    e.g:
        do ##class(MLpipeline.AutomatedPipeline).DataExtraction("MLpipeline", "PointSamples", "*", "datetime > '2026-02-26 17:50:00'")
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


}
