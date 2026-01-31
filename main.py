import os
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz

# 設定時區為台北時間
TP_TZ = pytz.timezone('Asia/Taipei')

# 從環境變數取得 API Key
CWA_KEY = os.environ.get("CWA_KEY")

def fetch_data():
    """從氣象署 API 抓取資料"""
    print("正在連線至氣象署 API...")
    
    # [修改點] 1. 改用 O-A0003-001 (局屬氣象站 - 現在天氣觀測報告)
    # 備註：局屬站數量較少但較為核心 (如台北、台中、高雄等)
    weather_url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0003-001?Authorization={CWA_KEY}&format=JSON"
    w_resp = requests.get(weather_url)
    w_data = w_resp.json()
    
    # 2. 抓取地震資料 (顯著有感地震 - E-A0015-001)
    eq_url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/E-A0015-001?Authorization={CWA_KEY}&format=JSON"
    e_resp = requests.get(eq_url)
    e_data = e_resp.json()
    
    return w_data, e_data

def process_weather(w_data):
    """處理天氣資料"""
    stations = []
    # 局屬氣象站通常位於主要城市，我們可以直接列出想抓的站名
    target_stations = ["臺北", "基隆", "板橋", "新屋", "新竹", "臺中", "嘉義", "臺南", "高雄", "恆春", "花蓮", "臺東", "澎湖", "金門", "馬祖"]
    
    if "records" in w_data and "Station" in w_data["records"]:
        for item in w_data["records"]["Station"]:
            name = item["StationName"]
            
            # 如果是目標城市之一
            if name in target_stations:
                try:
                    # 抓取溫度與濕度
                    temp = float(item["WeatherElement"]["AirTemperature"])
                    humid = float(item["WeatherElement"]["RelativeHumidity"])
                    
                    # [新增] 抓取該筆資料的觀測時間，方便確認是否即時
                    obs_time = item["ObsTime"]["DateTime"] 
                    
                    stations.append({
                        "city": name,
                        "temp": temp,
                        "humid": humid,
                        "obs_time": obs_time
                    })
                except (ValueError, KeyError):
                    continue
    
    df = pd.DataFrame(stations)
    # 依緯度或自定義順序可能會比較好看，這裡先依溫度排序
    if not df.empty:
        df = df.sort_values(by='temp', ascending=False)
    return df

def draw_chart(df):
    """繪製氣溫長條圖"""
    if df.empty:
        print("無資料可繪圖")
        return

    plt.figure(figsize=(12, 6)) # 加寬一點
    
    bars = plt.bar(df['city'], df['temp'], color='#e67e22', alpha=0.8) # 換個顏色區分新版
    
    plt.title('Real-time Temperature (Manned Stations)', fontsize=16)
    plt.xlabel('Station', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.ylim(0, 40)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 標示數值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height}',
                ha='center', va='bottom', fontsize=9)

    # 處理中文字型問題 (對應表)
    city_map = {
        "臺北": "Taipei", "基隆": "Keelung", "板橋": "Banqiao", "新屋": "Xinwu", 
        "新竹": "Hsinchu", "臺中": "Taichung", "嘉義": "Chiayi", "臺南": "Tainan", 
        "高雄": "Kaohsiung", "恆春": "Hengchun", "花蓮": "Hualien", "臺東": "Taitung",
        "澎湖": "Penghu", "金門": "Kinmen", "馬祖": "Matsu"
    }
    english_labels = [city_map.get(x, x) for x in df['city']]
    plt.xticks(range(len(english_labels)), english_labels, rotation=45) # 稍微旋轉標籤以免重疊

    plt.tight_layout() # 自動調整版面
    plt.savefig('weather_chart.png', dpi=100)
    print("圖表繪製完成: weather_chart.png")

def save_json(df, e_data):
    """整合資料並存檔"""
    
    latest_eq = {}
    if "records" in e_data and "Earthquake" in e_data["records"]:
        try:
            raw_eq = e_data["records"]["Earthquake"][0]
            latest_eq = {
                "time": raw_eq["EarthquakeInfo"]["OriginTime"],
                "magnitude": raw_eq["EarthquakeInfo"]["EarthquakeMagnitude"]["MagnitudeValue"],
                "depth": raw_eq["EarthquakeInfo"]["FocalDepth"],
                "location": raw_eq["EarthquakeInfo"]["Epicenter"]["Location"],
                "img": raw_eq["ReportImageURI"]
            }
        except IndexError:
            pass

    # [新增] 加上資料來源標記，方便前端 Debug
    output = {
        "meta": {
            "updated_at": datetime.now(TP_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            "source": "O-A0003-001 (Manned Stations)",
            "note": "Data updates every 30 mins"
        },
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
