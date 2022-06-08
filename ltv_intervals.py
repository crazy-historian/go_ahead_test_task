import pandas as pd

if __name__ == "__main__":
    users_df = pd.read_csv('task.csv')
    users_df = users_df.rename(columns={'Unnamed: 0': 'id'})
    users_df = users_df.drop(users_df.loc[users_df.ltv == 0].index)

    intervals = list()
    num_of_dates = users_df.install_date.nunique()
    ltv_start = 0
    for ltv_end in sorted(users_df.ltv.unique()):
        average_count = round(users_df.id.loc[
                                  (users_df.ltv > ltv_start) & (users_df.ltv <= ltv_end)
                                  ].count() / num_of_dates)
        if average_count == 10:
            intervals.append((ltv_start, ltv_end))
            print(f'({ltv_start}, {ltv_end}]')
            ltv_start = ltv_end
        elif average_count > 10:
            ltv_start = ltv_end
