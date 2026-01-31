import os
import requests
import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
import pytz
import math
import warnings

# 忽略 Geopandas/Shapely 的一些版本相容性警告
warnings.filterwarnings("ignore")

# =================設定區=================
# 設定中文字型 (使用 Linux 系統的 WenQuanYi Zen Hei)
plt.rcParams['font.family'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

TP_TZ = pytz.timezone('Asia/Taipei')
CWA_KEY = os.environ.get("CWA_KEY")

# 縣市界線 GeoJSON
MAP_URL = "https://raw.githubusercontent.com/bernie23361/optix-map/refs/heads/main/COUNTY_MOI_1140318%20(1).json"

# =================功能函數=================

def calculate_apparent_temp(temp, humid, wind_speed):
    """計算體感溫度 (Australian Apparent Temperature)"""
    try:
        e = (humid / 100.0) * 6.105 * math.exp((17.27 * temp) / (237.7 + temp))
        ws = max(wind_speed, 0)
        at = temp + 0.33 * e - 0.70 * ws - 4.00
        return at
    except Exception:
        return None

def fetch_data():
    """抓取自動氣象站資料 (O-A0001-001)"""
    print("正在下載氣象署自動測站資料...")
    url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001?Authorization={CWA_KEY}&format=JSON"
    resp = requests.get(url)
    return resp.json()

def process_data(raw_data):
    """處理資料並計算體感溫"""
    print("正在處理數據...")
    stations = []
    
    if "records" in raw_data and "Station" in raw_data["records"]:
        for item in raw_data["records"]["Station"]:
            try:
                lat = float(item["GeoInfo"]["Coordinates"][0]["StationLatitude"])
                lon = float(item["GeoInfo"]["Coordinates"][0]["StationLongitude"])
                temp = float(item["WeatherElement"]["AirTemperature"])
                humid = float(item["WeatherElement"]["RelativeHumidity"])
                wind = float(item["WeatherElement"]["WindSpeed"])
                
                # 排除極端異常值
                if temp < -10 or temp > 50 or humid < 0 or humid > 100:
                    continue

                at = calculate_apparent_temp(temp, humid, wind)
                if at is None:
                    continue

                stations.append({
                    "name": item["StationName"],
                    "lat": lat,
                    "lon": lon,
                    "temp": temp,
                    "at": at,
                    "city": item["GeoInfo"]["CountyName"],
                    "town": item["GeoInfo"]["TownName"]
                })
            except (ValueError, KeyError, IndexError):
                continue
                
    return pd.DataFrame(stations)

def generate_heatmap(df):
    """繪製體感溫度分佈圖"""
    print("正在繪製地圖...")
    
    # 讀取地圖
    taiwan_map = gpd.read_file(MAP_URL)
    
    # 設定範圍 (鎖定台灣本島與主要離島)
    min_lon, max_lon = 119, 122.5
    min_lat, max_lat = 21.8, 25.4
    
    df = df[(df['lon'] >= min_lon) & (df['lon'] <= max_lon) & 
            (df['lat'] >= min_lat) & (df['lat'] <= max_lat)]

    # 建立網格 (Grid)
    grid_x, grid_y = np.mgrid[min_lon:max_lon:400j, min_lat:max_lat:400j]
    
    # 插值運算 (Linear)
    grid_z = griddata(
        (df['lon'], df['lat']), 
        df['at'], 
        (grid_x, grid_y), 
        method='linear' 
    )

    # 開始繪圖
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # 設定色階與顏色表 (修正舊版 get_cmap 警告)
    levels = np.linspace(0, 40, 80)
    try:
        cmap = matplotlib.colormaps['nipy_spectral'].resampled(200)
    except AttributeError:
        cmap = plt.get_cmap('nipy_spectral')

    # 繪製等高線填色
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap, alpha=0.9, extend='both')
    
    # 疊加縣市界線 (黑色邊框)
    taiwan_map.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8, alpha=0.5)
    
    # 標示最高溫
    top1 = df.iloc[df['at'].argmax()]
    ax.text(top1['lon'], top1['lat'], f"最高體感\n{top1['city']}\n{top1['at']:.1f}°C", 
            fontsize=12, color='black', fontweight='bold', ha='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # 標題與資訊
    current_time = datetime.now(TP_TZ).strftime('%Y-%m-%d %H:%M')
    plt.title(f"全臺體感溫度分佈圖 (Apparent Temp)\n{current_time}", fontsize=18)
    plt.xlabel('經度')
    plt.ylabel('緯度')
    
    cbar = plt.colorbar(contour, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('體感溫度 (°C)', fontsize=12)
    
    # 鎖定顯示範圍
    ax.set_xlim(119.3, 122.3)
    ax.set_ylim(21.8, 25.4)
    
    plt.savefig('apparent_temp_map.png', dpi=150, bbox_inches='tight')
    print("圖表存檔成功: apparent_temp_map.png")

def save_json(df):
    """儲存詳細資料 JSON"""
    output_list = df[['city', 'town', 'name', 'temp', 'at', 'lat', 'lon']].to_dict(orient='records')
    
    output = {
        "meta": {
            "updated_at": datetime.now(TP_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            "source": "CWA O-A0001-001 (Auto Stations)",
            "count": len(output_list)
        },
        "data": output_list
    }
    
    with open('data_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("JSON 存檔成功: data_detailed.json")

if __name__ == "__main__":
    try:
        data = fetch_data()
        df = process_data(data)
        
        if not df.empty:
            generate_heatmap(df)
            save_json(df)
            print("全部執行完成！")
        else:
            print("抓取到的資料為空，請檢查 API。")
            
    except Exception as e:
        print(f"發生錯誤: {e}")
