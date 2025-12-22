import requests
# 实时天气
url = "https://devapi.qweather.com/v7/weather/now"
params = {
    "key": "03446121c814474a9ec6536505afe1dc",
    "location": "101010100"  # 城市ID
}
response = requests.get(url, params=params)
print(response.text)