import requests
import base64
import json

output = "output.csv"                                                           # Output File:

url = "https://api.ignatius.io/api/report/export?reportId=5358&tableId=2362&exportType=csv&size=-1&tblName=1"

payload={}


headers={"Authorization": "Bearer X3p-Hum47YMiY8dBEw-OsQpSnVPcZRFdqtSRpx9eEdY"}
input = requests.request("GET", url, headers=headers, data=payload).


decodedBytes = base64.b64decode()
decodedStr = decodedBytes.decode("ascii")
print(decodedStr)