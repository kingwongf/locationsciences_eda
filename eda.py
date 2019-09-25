import pandas as pd
import numpy as np
from functools import reduce
import operator
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', 199)  # or 199

xls = pd.ExcelFile("data/Location Sciences - REIT insights.xlsx")

visits = pd.read_excel(xls, "visits")
distance = pd.read_excel(xls, "distance")
demographics = pd.read_excel(xls,"demo")
REIT_price = pd.read_excel(xls,"SHB.L")

add_month_xls = pd.ExcelFile("data/Location Sciences - REIT insights - addendum.xlsx")
add_month_distance = pd.read_excel(add_month_xls, "distance")
add_month_demographics = pd.read_excel(add_month_xls,"demo")


# REIT_price abd visits are daily data while distance and demographics are monthly data(both requires pivoting to join on YYYY-MM)

## let's first consider all tables in the model
#print(visits.columns, visits.head)
## TODO produce visits_pivot_by_area and visits_pivot_by_area_monhtly

visits.date = visits["date"].map(lambda x: x.strftime('%Y-%m-%d'))
visits.index = pd.to_datetime(visits.date)
visits.columns = visits.columns.map(lambda x: "visits_" + x)
visits_pivot_by_area = visits.pivot_table(index=visits.index, values=['visits_avg_dwell_mins','visits_visits'], columns=["visits_area"], aggfunc=np.sum)
visits_pivot_by_area.columns = visits_pivot_by_area.columns.to_series().str.join('_')
#print(visits_pivot_by_area)
visits_pivot_by_area_monthly = visits_pivot_by_area.resample('1M').sum()


## TODO produce REIT_price and REIT_price_monthly
REIT_price.date = REIT_price["Date"].map(lambda x: x.strftime('%Y-%m-%d'))
REIT_price.index = pd.to_datetime(REIT_price.date)
REIT_price.columns = REIT_price.columns.map(lambda x: "price_" + x)
REIT_price_monthly = REIT_price.resample('1M').first()
REIT_price_monthly.index = pd.to_datetime(REIT_price_monthly.price_Date)

## since we are missing 2018-10-01 to 2018-12-01 data for distance and demographics. We'll drop the correspondings in price and visits
#REIT_price_monthly_dropped = REIT_price_monthly.loc[(REIT_price_monthly.index < "2018-09-30") |(REIT_price_monthly.index > "2018-12-30")]

#print(REIT_price_monthly_dropped)
#REIT_price_monthly_dropped['monthly_lookahead_ret'] = np.log(REIT_price_monthly_dropped['price_Adj Close']).diff(1).shift(-1)

# bin = np.sign(REIT_price_monthly_dropped['monthly_lookahead_ret'])
#print(bin)



def month_format_index(df, df_name):
    ''' turns df month column to date index'''
    df.month = df["month"].map(lambda x: x[0:4] + "-" + x[5:])
    df.month = pd.to_datetime(df["month"])
    df['date'] = df.month
    df.index = df.date
    df.columns = df.columns.map(lambda x: df_name + "_" + x)
    return df.sort_index()



distance = month_format_index(distance, "distance")
add_month_distance = month_format_index(add_month_distance, "distance")
distance = pd.concat([distance, add_month_distance], axis=0)

distance_travelled_pivot_by_area = distance.pivot_table(index=['date'], values=['distance_avg_distance_travelled'], columns=["distance_area"], aggfunc=np.sum)
distance_percent_visits_pivot_by_area = distance.pivot_table(index=['date'], values=['distance_%visitors'], columns=["distance_area"], aggfunc=np.average)
distance_travelled_pivot_by_borough = distance.pivot_table(index=['date'], values=['distance_avg_distance_travelled'], columns=["distance_borough"], aggfunc=np.sum)
distance_percent_visits_pivot_by_borough = distance.pivot_table(index=['date'], values=['distance_%visitors'], columns=["distance_borough"], aggfunc=np.average)

## TODO produce distance_travelled_pivot_by_area, distance_percent_visits_pivot_by_area

distance_travelled_pivot_by_area.columns = distance_travelled_pivot_by_area.columns.to_flat_index()
distance_travelled_pivot_by_area.columns = [reduce(operator.add, tup) for tup in distance_travelled_pivot_by_area.columns]

distance_percent_visits_pivot_by_area.columns = distance_percent_visits_pivot_by_area.columns.to_flat_index()
distance_percent_visits_pivot_by_area.columns = [reduce(operator.add, tup) for tup in distance_percent_visits_pivot_by_area.columns]

print(distance_travelled_pivot_by_area.columns)
print(distance_percent_visits_pivot_by_area.columns)

distance_list = [distance_travelled_pivot_by_area, distance_percent_visits_pivot_by_area, distance_travelled_pivot_by_borough, distance_percent_visits_pivot_by_borough]

## TODO produce demographics_percent_visitors_pivot_by_area
demographics = month_format_index(demographics, "demographics")
add_month_demographics = month_format_index(add_month_demographics, "demographics")

demographics = pd.concat(([demographics, add_month_demographics]), axis=0)

demographics_percent_visitors_pivot_by_area = demographics.pivot_table(index=['date'], values=['demographics_%visitors'], columns=["demographics_area", "demographics_demo_group"], aggfunc=np.average)
demographics_percent_visitors_pivot_by_area.columns = demographics_percent_visitors_pivot_by_area.columns.to_flat_index()
demographics_percent_visitors_pivot_by_area.columns = [reduce(operator.add, tup) for tup in demographics_percent_visitors_pivot_by_area.columns]

print(demographics_percent_visitors_pivot_by_area.columns)


## TODO produce all outputs

li_location_science = [visits_pivot_by_area, REIT_price, distance_travelled_pivot_by_area, distance_percent_visits_pivot_by_area, demographics_percent_visitors_pivot_by_area]

location_science = reduce(lambda X, x: pd.merge_asof(X.sort_index(), x.sort_index(), left_index=True, right_index=True,
                                                  direction='forward', tolerance=pd.Timedelta('1d')), li_location_science)

print(location_science.columns)

location_science.to_pickle("data/location_science.pkl")
location_science.to_csv("data/location_science.csv")