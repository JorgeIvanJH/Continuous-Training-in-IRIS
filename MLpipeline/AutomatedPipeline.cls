Class MLpipeline.AutomatedPipeline Extends %RegisteredObject
{

// Seed for reproducibility in data splitting and model training

Property SEED As %Integer;

// Scaler object for data normalization (e.g., StandardScaler from sklearn)

Property SCALER As %SYS.Python;

// Test size for train-test split (e.g., 0.2 for 20% test data)

Property TESTSIZE As %Numeric(SCALE = 2);

Method %OnNew(seed As %Integer = 42, testSize As %Numeric(SCALE=2) = 0.2) As %Status [ Private ]
{
    set ..SEED = seed
    set ..TESTSIZE = testSize
    do ##class(%SYS.System).WriteToConsoleLog("Created instance of MLpipeline.AutomatedPipeline with SEED=" _ seed _ " and TESTSIZE=" _ testSize, 0, 0)
    Quit $$$OK
}

/*
Leverages the FeatureStore to extract data and transform it to pandas DataFrame.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New()
    set df = AP.DataExtraction("2026-02-26 17:50:00")
*/
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

/*
Validates the extracted DataFrame for missing values and logs the results.
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New()
    set df = AP.DataExtraction("2026-02-26 17:50:00")
    set validDf = AP.DataValidation(df)
*/
Method DataValidation(df As %SYS.Python) As %SYS.Python [ Language = python ]
{
    import iris
    import pandas as pd
    
    try:
        if df is None:
            iris._SYS.System.WriteToConsoleLog("Received None DataFrame for validation.", 0, 2)
            raise ValueError("Input DataFrame is None")
        
        # Missing Values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            iris._SYS.System.WriteToConsoleLog(f"Missing values in DataFrame: {missing_values}", 0, 1)
        
        # Additional validations can be added here (e.g. OUTLIERS)


        # End of validation
        iris._SYS.System.WriteToConsoleLog("DataFrame validation completed successfully.", 0, 0)
        return df
    except Exception as e:
        print("Error during Data Validation:", e)
        iris._SYS.System.WriteToConsoleLog(f"Validation Error: {str(e)}", 0, 2)
        return None
}

/* 
Prepares the validated DataFrame for model training
e.g:
    set AP = ##class(MLpipeline.AutomatedPipeline).%New()
    set df = AP.DataExtraction("2026-02-26 17:50:00")
    set validDf = AP.DataValidation(df)
    set prepDf = AP.DataPreparation(validDf)
*/
Method DataPreparation(df As %SYS.Python) As %SYS.Python [ Language = python ]
{
    import iris
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    try:
        if df is None:
            iris._SYS.System.WriteToConsoleLog("Received None DataFrame for preparation.", 0, 2)
            raise ValueError("Input DataFrame is None")
        
        iris._SYS.System.WriteToConsoleLog("Splitting DataFrame into training and test sets.", 0, 0)
        df_train, df_test = train_test_split(df, test_size=self.TESTSIZE, random_state=self.SEED)
        X_train = df_train[["x"]]
        y_train = df_train["y"]
        X_test = df_test[["x"]]
        y_test = df_test["y"]

        
        if not isinstance(self.SCALER, StandardScaler):
            iris._SYS.System.WriteToConsoleLog("No existing StandardScaler found. Initializing new scaler and fitting to training data.", 0, 0)
            self.SCALER = StandardScaler()
            self.SCALER.fit(X_train)
        else:
            iris._SYS.System.WriteToConsoleLog("Using existing StandardScaler for normalization.", 0, 0)
        
        iris._SYS.System.WriteToConsoleLog("Applying StandardScaler transformation to training and test data.", 0, 0)
        X_train = self.SCALER.transform(X_train)
        X_test = self.SCALER.transform(X_test)

        iris._SYS.System.WriteToConsoleLog("Data preparation completed successfully.", 0, 0)
        return (X_train, y_train, X_test, y_test)
    except Exception as e:
        print("Error during Data Preparation:", e)
        iris._SYS.System.WriteToConsoleLog(f"Preparation Error: {str(e)}", 0, 2)
        return None
}

Method ModelTraining(df As %SYS.Python) As %SYS.Python [ Language = python ]
{

    import iris

    import pandas as pd

    from sklearn.linear_model import LinearRegression

    try:

        if df is None:

            iris._SYS.System.WriteToConsoleLog("Received None DataFrame for model training.", 0, 2)

            raise ValueError("Input DataFrame is None")

        X = df.drop('target', axis=1)  # Assuming 'target' is the label column

        y = df['target']

        model = LinearRegression()

        model.fit(X, y)

        iris._SYS.System.WriteToConsoleLog("Model training completed successfully.", 0, 0)

        return model

    except Exception as e:

        print("Error during Model Training:", e)

        iris._SYS.System.WriteToConsoleLog(f"Model Training Error: {str(e)}", 0, 2)

        return None
}

}
