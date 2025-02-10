import sys
sys.path.append(".")

from analysis.agg.functions import join_data, get_joined_and_aggregated_data

model = "MLP"

def main():
    df_1 = get_joined_and_aggregated_data(model=model, mac_model="M1 Pro")
    df_2 = get_joined_and_aggregated_data(model=model, mac_model="M2 Pro")
    df = join_data(df_1, df_2, "M1 Pro", "M2 Pro", "mac_model")

    df_3 = get_joined_and_aggregated_data(model=model, mac_model="M3 pro")
    df = join_data(df, df_3, None, "M3 Pro", "mac_model")
    print(df)
    df.to_csv(f"exports/agg/{model}.csv")


if __name__ == "__main__":
    main()