Class MLpipeline.DataManager Extends %Persistent
{

ClassMethod UploadCSVtoIRIS(csvpath As %String, namespace As %String, package As %String, tablename As %String) As %Status  [ Language = python ]
{
    # Populates an IRIS table data from CSV file at csvpath. Automatically detects and converts date columns. 
    # e.g:
    #   do ##class(MLpipeline.DataManager).UploadCSVtoIRIS("/dur/data/point_samples.csv","USER", "MLpipeline","PointSamples")
    from sqlalchemy import create_engine
    import pandas as pd
    
    try:
        engine = create_engine(f"iris+emb:///{namespace}")

        df = pd.read_csv(csvpath)

        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna().astype(str).head(20)
            try:
                parsed = pd.to_datetime(sample, errors="raise") # just to validate
                df[col] = pd.to_datetime(df[col], errors="coerce") # actual conversion
            except Exception:
                continue
        df.to_sql(tablename, engine, schema=package, if_exists='append', method='multi')
        return True

    except Exception as e:
        print(f"Error uploading CSV to IRIS: {e}")
        return False
}

}

