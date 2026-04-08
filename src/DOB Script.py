import pandas as pd
import openpyxl
from datetime import date

df = pd.read_excel("TC_ALL_DOB.xlsx")

df = df.drop(columns= ["Unnamed: 3", "Unnamed: 4", "Unnamed: 5", "Unnamed: 6", "Unnamed: 7", "Unnamed: 8", "Unnamed: 9", "Unnamed: 10", "Unnamed: 11", "Unnamed: 12", "Unnamed: 13", "Unnamed: 14", "Unnamed: 15"])


df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
today = date.today()
df["Age of Participant"] = df["DOB"].apply(
    lambda dob: today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
)

output_file = "TC_ALL_DOB_CLEANED.xlsx"
df.to_excel(output_file, index=False)