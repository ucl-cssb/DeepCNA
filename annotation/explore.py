import pandas as pd


if 1:
    ## metadata analysis
    data = pd.read_csv('metadata.csv')
    print(data.head())
    print(data['primary_site_text'].value_counts())

    # write out a summary csv file 
    summary_df = data.groupby(["primary_site_code", "primary_site_text"]).size().reset_index(name="number_of_samples")

    # sort by number of samples
    summary_df = summary_df.sort_values(by="number_of_samples", ascending=False)

    summary_df.to_csv('primary_site_info.csv', index=False)

if 1:
    ## read pga data
    pga = pd.read_csv('PGA_dataframe_filtered.csv')
    print(pga.head())

    df_pga = pga.groupby("Disease Type", as_index=False)["PGA"].median()

    # sort by PGA
    df_pga = df_pga.sort_values(by="PGA", ascending=False)
    print(df_pga)
    
    # merge with primary_site_info
    primary_site_info = pd.read_csv('primary_site_info.csv')
    
    df_pga = pd.merge(df_pga, primary_site_info, left_on="Disease Type", right_on="primary_site_text")

    # sort by number of samples
    df_pga = df_pga.sort_values(by="number_of_samples", ascending=False)

    # round pga to three dp
    df_pga["PGA"] = df_pga["PGA"].round(3)

    print(df_pga)
