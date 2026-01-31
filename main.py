import os
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import geopandas as gpd
from datetime import datetime
import pytz
import math
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# =================設定區=================
plt.rcParams['font.family'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
TP_TZ = pytz.timezone('Asia/Taipei')
CWA_KEY = os.environ.get("CWA_KEY")
MAP_URL = "https://raw.githubusercontent.com/bernie23361/optix-map/refs/heads/main/COUNTY_MOI_1140318%20(1).json"

# 客製化色階 (0~40度)
custom_colors = ['#2c7bb6', '#00a6ca', '#00ccbc', '#90eb9d', '#e4e897', '#ffff8c', '#f9d057', '#f29e2e', '#d7191c', '#a31f7b', '#6e208a']
cmap_custom = LinearSegmentedColormap.from_list("taiwan_temp", custom_colors, N=256)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# =================功能函數=================

def calculate_apparent_temp(temp, humid, wind_speed):
    try:
        humid = max(0, min(100, humid))
        e = (humid / 100.0) * 6.105 * math.exp((17.27 * temp) / (237.7 + temp))
        ws = max(wind_speed, 0)
        return temp + 0.33 * e - 0.70 * ws - 4.00
    except:
        return None

def fetch_data():
    log("下載氣象資料...")
    try:
        url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001?Authorization={CWA_KEY}&format=JSON"
        # 增加 timeout 確保不會卡住
        resp = requests.get(url, timeout=20)
        return resp.json()
    except Exception as e:
        log(f"API 錯誤: {e}")
        return None

def process_data(raw_data):
    log("處理數據中...")
    stations = []

    if raw_data and "records" in raw_data and "Station" in raw_data["records"]:
        for item in raw_data["records"]["Station"]:
            try:
                lat = float(item["GeoInfo"]["Coordinates"][0]["StationLatitude"])
                lon = float(item["GeoInfo"]["Coordinates"][0]["StationLongitude"])
                temp = float(item["WeatherElement"]["AirTemperature"])
                humid = float(item["WeatherElement"]["RelativeHumidity"])
                wind = float(item["WeatherElement"]["WindSpeed"])
                
                # 基本過濾
                if temp < -20 or temp > 50 or humid < 0 or humid > 100: continue
                
                at = calculate_apparent_temp(temp, humid, wind)
                if at is None: continue

                stations.append({
                    "name": item["StationName"], 
                    "lat": lat, 
                    "lon": lon, 
                    "temp": temp, 
                    "at": at, 
                    "city": item["GeoInfo"]["CountyName"], 
                    "town": item["GeoInfo"]["TownName"]
                })
            except: continue
            
    return pd.DataFrame(stations)

def generate_dots_map(df):
    log("繪製圓點地圖...")
    
    # 讀取地圖
    taiwan_map = gpd.read_file(MAP_URL)
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # 1. 繪製底圖 (淺灰色陸地，深灰色邊框)
    taiwan_map.plot(ax=ax, facecolor='#f5f5f5', edgecolor='#999999', linewidth=0.8, zorder=1)
    
    # 設定顏色規範
    norm = Normalize(vmin=0, vmax=40)
    
    # 2. 繪製測站圓點 (Scatter Plot)
    # s=50: 圓點大小
    # edgecolor='black': 圓點加上黑色細邊框，增加對比度
    sc = ax.scatter(
        df['lon'], df['lat'], 
        c=df['at'], 
        cmap=cmap_custom, 
        norm=norm, 
        s=60, 
        edgecolor='black', 
        linewidth=0.6, 
        alpha=0.9,
        zorder=5
    )

    # 3. 標示前三高溫的站點
    top3 = df.nlargest(3, 'at')
    for i, (_, row) in enumerate(top3.iterrows()):
        # 稍微錯開文字位置避免重疊
        offset_y = 0.05 + (i * 0.02)
        ax.text(
            row['lon'], row['lat'] + offset_y, 
            f"{row['name']}\n{row['at']:.1f}", 
            fontsize=10, fontweight='bold', ha='center', zorder=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2')
        )
        # 畫一條細線連到點
        ax.plot([row['lon'], row['lon']], [row['lat'], row['lat'] + offset_y], color='black', linewidth=0.5, zorder=9)

    # 標題與修飾
    current_time = datetime.now(TP_TZ).strftime('%Y-%m-%d %H:%M')
    plt.title(f"全臺體感溫度監測網\n{current_time}", fontsize=20, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('體感溫度 (°C)', fontsize=12)
    cbar.set_ticks(np.arange(0, 41, 5))
    
    # 設定顯示範圍 (包含金馬與本島)
    ax.set_xlim(118.0, 122.3)
    ax.set_ylim(21.7, 26.5)
    
    # 移除經緯度刻度 (讓畫面更乾淨)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.savefig('apparent_temp_map.png', dpi=150, bbox_inches='tight')
    log("圖表已更新: apparent_temp_map.png")

def save_json(df):
    output = {
        "meta": {
            "updated_at": datetime.now(TP_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            "source": "CWA Auto Stations",
            "count": len(df)
        },
        "data": df[['city', 'town', 'name', 'temp', 'at', 'lat', 'lon']].to_dict(orient='records')
    }
    with open('data_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log("JSON 已更新")

if __name__ == "__main__":
    log("程式啟動")
    
    try:
        raw = fetch_data()
        if raw:
            df = process_data(raw)
            if not df.empty:
                save_json(df)
                generate_dots_map(df)
            else:
                log("無有效資料")
        else:
            log("API 下載失敗")
    except Exception as e:
        import traceback
        traceback.print_exc()
        log(f"執行錯誤: {e}")
