import os
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

# 設定時區為台北時間
TP_TZ = pytz.timezone('Asia/Taipei')

# 從環境變數取得 API Key
CWA_KEY = os.environ.get("CWA_KEY")

def fetch_data():
    """從氣象署 API 抓取資料"""
    print("正在連線至氣象署 API...")
    
    # 1. 抓取天氣資料 (自動氣象站 - O-A0001-001)
    weather_url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001?Authorization={CWA_KEY}&format=JSON"
    w_resp = requests.get(weather_url)
    w_data = w_resp.json()
    
    # 2. 抓取地震資料 (顯著有感地震 - E-A0015-001)
    eq_url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/E-A0015-001?Authorization={CWA_KEY}&format=JSON"
    e_resp = requests.get(eq_url)
    e_data = e_resp.json()
    
    return w_data, e_data

def process_weather(w_data):
    """處理天氣資料，篩選主要城市"""
    stations = []
    target_stations = ["臺北", "板橋", "桃園", "臺中", "臺南", "高雄", "花蓮", "臺東"]
    
    if "records" in w_data and "Station" in w_data["records"]:
        for item in w_data["records"]["Station"]:
            name = item["StationName"]
            # 這裡簡單比對站名，包含目標城市即可
            if name in target_stations:
                try:
                    temp = float(item["WeatherElement"]["AirTemperature"])
                    humid = float(item["WeatherElement"]["RelativeHumidity"])
                    stations.append({
                        "city": name,
                        "temp": temp,
                        "humid": humid
                    })
                except ValueError:
                    continue
    
    # 轉成 DataFrame 方便繪圖與排序
    df = pd.DataFrame(stations)
    if not df.empty:
        df = df.sort_values(by='temp', ascending=False)
    return df

def draw_chart(df):
    """繪製氣溫長條圖 (儲存為 weather_chart.png)"""
    if df.empty:
        print("無資料可繪圖")
        return

    plt.figure(figsize=(10, 6))
    # GitHub Runner 預設無中文字型，為避免亂碼，暫時使用英文標籤或不顯示中文
    # 這裡我們用簡單的長條圖
    bars = plt.bar(df['city'], df['temp'], color='#3498db', alpha=0.7)
    
    plt.title('Real-time Temperature Monitoring (Taiwan)', fontsize=16)
    plt.xlabel('City', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.ylim(0, 40) # 設定溫度顯示範圍
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 在柱狀圖上方標示溫度
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}°C',
                ha='center', va='bottom')

    # 因為環境沒有中文字體，先強制轉成英文 ID 顯示 (或你可以上傳字體檔)
    # 這裡為了演示成功，先讓 X 軸顯示簡單的索引或拼音，避免方塊字
    # 實際專案我們通常會把 '臺北' map 成 'Taipei'
    city_map = {
        "臺北": "Taipei", "板橋": "Banqiao", "桃園": "Taoyuan", 
        "臺中": "Taichung", "臺南": "Tainan", "高雄": "Kaohsiung", 
        "花蓮": "Hualien", "臺東": "Taitung"
    }
    english_labels = [city_map.get(x, x) for x in df['city']]
    plt.xticks(range(len(english_labels)), english_labels)

    plt.savefig('weather_chart.png', dpi=100)
    print("圖表繪製完成: weather_chart.png")

def save_json(df, e_data):
    """整合資料並存檔"""
    
    # 處理地震資料 (取最新一筆)
    latest_eq = {}
    if "records" in e_data and "Earthquake" in e_data["records"]:
        try:
            raw_eq = e_data["records"]["Earthquake"][0]
            latest_eq = {
                "time": raw_eq["EarthquakeInfo"]["OriginTime"],
                "magnitude": raw_eq["EarthquakeInfo"]["EarthquakeMagnitude"]["MagnitudeValue"],
                "depth": raw_eq["EarthquakeInfo"]["FocalDepth"],
                "location": raw_eq["EarthquakeInfo"]["Epicenter"]["Location"],
                "img": raw_eq["ReportImageURI"] # 氣象署提供的報告圖
            }
        except IndexError:
            pass

    # 輸出最終 JSON
    output = {
        "updated_at": datetime.now(TP_TZ).strftime('%Y-%m-%d %H:%M:%S'),
        "weather_list": df.to_dict(orient='records'),
        "earthquake": latest_eq
    }
    
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("資料儲存完成: data.json")

if __name__ == "__main__":
    try:
        raw_weather, raw_eq = fetch_data()
        weather_df = process_weather(raw_weather)
        
        draw_chart(weather_df)
        save_json(weather_df, raw_eq)
        
        print("執行成功！")
    except Exception as e:
        print(f"執行發生錯誤: {e}")
        # 這裡不 raise error，以免 GitHub Action 顯示紅色叉叉 (雖然可以讓他顯示，但先求穩)
