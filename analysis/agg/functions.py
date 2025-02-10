import pandas as pd
import os

def import_data(model, type, mac_model):
    # load the dataset
    path = os.path.join("exports", f"metrics_{model}_{type}_{mac_model}.csv")
    dataframe = pd.read_csv(path, engine="python")
    return dataframe

def join_data(df1, df2, name1, name2, name="name"):
    if name1 is not None:
        df1[name] = name1
    df2[name] = name2
    # join dataframes
    df = pd.concat([df1, df2])
    return df

def aggregate_data(df, name="name"):
    # aggregate data cpu_percent (mean), temperature (mean), power (sum), batch_time (sum), loss (lowest), memory (mean)
    df_agg = df.groupby(name).agg(
        {
            "cpu_percent": "mean",
            "temperature": "mean",
            "power_consumption": "sum",
            "batch_time": "sum",
            "loss": "min",
            "memory_usage": "mean",
        }
    )
    return df_agg

def get_joined_and_aggregated_data(model, mac_model):
    df1 = import_data(model=model, type="CPU", mac_model=mac_model)
    df1 = aggregate_data(df1, name="training_uuid")
    df2 = import_data(model=model, type="GPU", mac_model=mac_model)
    df2 = aggregate_data(df2, name="training_uuid")

    df = join_data(df1, df2, "CPU", "GPU")
    df_agg = aggregate_data(df)
    return df_agg
