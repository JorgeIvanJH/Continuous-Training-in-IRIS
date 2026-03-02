Class MLpipeline.FeatureStore Extends %RegisteredObject
{

// Example of a constant parameter that never changes

Parameter ConstVar = "CONSTANT VAR FOR THE PROJECT";

Method %OnNew() As %Status [ Private ]
{
    do ##class(%SYS.System).WriteToConsoleLog("Created instance of MLpipeline.FeatureStore with ConstVar: "_..#ConstVar, 0, 0)
    Quit $$$OK
}

/// Extracts data from IRIS database using SQL query constructed from the input parameters.
/// e.g:
///    set FS = ##class(MLpipeline.FeatureStore).%New()
///    set rs = FS.DataExtraction("MLpipeline", "PointSamples", "*", "datetime > '2026-02-26 17:50:00'")
Method DataExtraction(schema As %String, tablename As %String, columns As %String = "*", filters As %String = "") As %SQL.StatementResult
{
    set query = "SELECT " _ columns _ " FROM " _ schema _ "." _ tablename
    if (filters '= "") set query = query _ " WHERE " _ filters

    set tStatement = ##class(%SQL.Statement).%New()
    set tStatus = tStatement.%Prepare(query)
    
    // Handle Preparation Errors (e.g., syntax errors)
    if $$$ISERR(tStatus) {
        do ##class(%SYS.System).WriteToConsoleLog("SQL Prepare Failed: " _ query, 0, 2)
        quit ""
    }

    set rs = tStatement.%Execute()

    // Handle Execution Errors
    if (rs.%SQLCODE < 0) {
        do ##class(%SYS.System).WriteToConsoleLog("SQL Execution Error: " _ rs.%Message, 0, 2)
        quit rs // Return the result set anyway; Python will see the error code
    }
    
    do ##class(%SYS.System).WriteToConsoleLog("Executed query: " _ query, 0, 0)
    
    return rs
}

}
