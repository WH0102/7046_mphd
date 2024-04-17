import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
import warnings

# setting ignore as a parameter and further adding category
warnings.simplefilter(action='ignore', category=FutureWarning) 

# Setting the nominatim user agent
geolocator = Nominatim(user_agent="str_address_geocoding")

# Set the rate limiters
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

def getting_str_geolocation(df) -> None:
    # To run the program in loop of 100
    for num in range(1183700, len(df), 100):
        # For monitoring in the terminal
        print(f"{num:,d}")
        # To avoid unnecessary ping
        time.sleep(1)
        # To slice the dataframe
        temp_df = df.iloc[num:num+100]
        try:
            # To get the geolocation of the address
            temp_df.loc[:,"location"] = temp_df.loc[:,"address"].apply(geocode)
            temp_df.loc[:,'point'] = temp_df.loc[:,'location'].apply(lambda loc: tuple(loc.point) if loc else None)
        except:
            continue
        # To save as csv file in slice manner.
        temp_df = temp_df.loc[temp_df.loc[:,"location"].notnull()]

        # To save the file only if the temp_df is more the 1
        if len(temp_df) > 0:
            temp_df.to_csv(f"str_partition/sheet{int(num/100)}.csv", sep = "|", index = False, errors = "ignore")
        else:
            print(f"Address from {num} - {num+100} have no geolocation returned.")
        
        # Print a separator
        print("---------------------------------")



# To read the csv file and run the program
if __name__ == "__main__":
    getting_str_geolocation(df = pd.read_csv("str_address.csv", sep="|"))