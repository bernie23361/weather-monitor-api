import os
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
import pytz
import math

# =================設定區=================
# 設定中文字型
plt.rcParams['font.family'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
TP_TZ = pytz.timezone('Asia/Taipei')
CWA_KEY = os.environ.get("CWA_KEY")

# 你的 GeoJSON 連結
MAP_URL = "https://raw.githubusercontent.com/bernie23361/optix-map/refs/heads/main/COUNTY_MOI_1140318%20(1).json"
# 若想畫更細的鄉鎮界線，可改用 TOWN 連結，但縣市界線(COUNTY)在全台圖上比較清晰
# TOWN_URL = "https://raw.githubusercontent.com/bernie23361/optix-map/refs/heads/main/TOWN_MOI_1140318.json"

# =================功能函數=================

def calculate_apparent_temp(temp, humid, wind_speed):
    """
    計算體感溫度 (Australian Apparent Temperature Model)
    AT = Ta + 0.33×e − 0.70×ws − 4.00
    e: 水氣壓 (hPa), ws: 風速 (m/s)
    """
    try:
        # 1. 計算水氣壓 (Vapor Pressure)
        # Magnus formula approximation
        e = (humid / 100.0) * 6.105 * math.exp((17.27 * temp) / (237.7 + temp))
        
        # 2. 計算體感溫度 (風速單位若為 m/s)
        # 如果風速小於 0 或無資料，視為 0
        ws = max(wind_speed, 0)
        
        at = temp + 0.33 * e - 0.70 * ws - 4.00
        return at
    except Exception:
        return None

def fetch_data():
    """抓取自動氣象站資料 (O-A0001-001) - 站點較多"""
    print("正在下載氣象署自動測站資料...")
    url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001?Authorization={CWA_KEY}&format=JSON"
    resp = requests.get(url)
    data = resp.json()
    return data

def process_data(raw_data):
    """處理原始資料，計算體感溫度"""
    print("正在處理數據並計算體感溫度...")
    stations = []
    
    if "records" in raw_data and "Station" in raw_data["records"]:
        for item in raw_data["records"]["Station"]:
            try:
                # 排除無座標的站點
                lat = float(item["GeoInfo"]["Coordinates"][0]["StationLatitude"])
                lon = float(item["GeoInfo"]["Coordinates"][0]["StationLongitude"])
                
                # 排除極端值 (氣象署壞掉的儀器常顯示 -99)
                temp = float(item["WeatherElement"]["AirTemperature"])
                humid = float(item["WeatherElement"]["RelativeHumidity"])
                wind = float(item["WeatherElement"]["WindSpeed"])
                
                if temp < -10 or temp > 50 or humid < 0 or humid > 100:
                    continue

                # 計算體感溫度
                at = calculate_apparent_temp(temp, humid, wind)
                if at is None:
                    continue

                stations.append({
                    "name": item["StationName"],
                    "lat": lat,
                    "lon": lon,
                    "temp": temp, # 真實溫度
                    "at": at,     # 體感溫度
                    "city": item["GeoInfo"]["CountyName"],
                    "town": item["GeoInfo"]["TownName"]
                })
            except (ValueError, KeyError, IndexError):
                continue
                
    return pd.DataFrame(stations)

def generate_heatmap(df):
    """生成體感溫度分佈圖 (含地圖切割)"""
    print("正在載入地圖檔 (GeoJSON)...")
    # 讀取台灣地圖 (Geopandas 支援直接讀 URL)
    taiwan_map = gpd.read_file(MAP_URL)
    
    # 設定繪圖範圍 (台灣本島與離島大致範圍)
    # 為了美觀，我們可以聚焦在本島，或根據資料自動調整
    min_lon, max_lon = 119, 122.5
    min_lat, max_lat = 21.8, 25.4
    
    # 過濾掉範圍外的測站 (避免插值錯誤)
    df = df[(df['lon'] >= min_lon) & (df['lon'] <= max_lon) & 
            (df['lat'] >= min_lat) & (df['lat'] <= max_lat)]

    # === 1. 空間插值 (Interpolation) ===
    # 建立網格 (解析度越高越細緻，但也越慢。300x300 對 GitHub Actions 算安全)
    grid_x, grid_y = np.mgrid[min_lon:max_lon:400j, min_lat:max_lat:400j]
    
    # 使用 cubic (三次樣條) 或 linear 插值
    # cubic 比較平滑，但如果測站太稀疏會有怪異波動，linear 比較保守
    grid_z = griddata(
        (df['lon'], df['lat']), 
        df['at'], 
        (grid_x, grid_y), 
        method='linear' 
    )

    # === 2. 開始繪圖 ===
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # 自定義顏色條 (模擬氣象局風格: 藍->綠->黃->紅->紫)
    # 定義數值區間的顏色 (可根據季節調整)
    levels = np.linspace(0, 40, 80) # 0度到40度，切80份
    cmap = plt.cm.get_cmap('nipy_spectral', 200) # 使用光譜色系
    
    # 繪製等高線填色圖 (Heatmap)
    # 這裡先畫在整個方形區域
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap, alpha=0.9, extend='both')
    
    # === 3. 地圖裁切 (Clipping) ===
    # 這是最關鍵的一步：把地圖以外的顏色遮掉
    # 我們把台灣地圖畫在上面，但設定為「無填充、黑框」，或是用來做 Mask
    # 簡單作法：使用 Geopandas 的 plot 疊加邊界
    taiwan_map.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8, alpha=0.5)
    
    # 進階裁切：利用 shapefile 建立遮罩 (Mask) 讓海變成白色
    # 為了簡化 GitHub Actions 負擔，我們這裡用 "Patch" 技巧
    # 獲取台灣地圖的總邊界 (Union)
    tw_boundary = taiwan_map.unary_union
    
    # 設定 matplotlib 的裁切路徑
    from descartes import PolygonPatch
    from matplotlib.collections import PatchCollection
    
    # 這邊需要一點幾何處理，把 Contour 圖限制在台灣形狀內
    # 為了程式碼穩定性，我們採用「疊加白色遮罩」的反向思維：
    # (太複雜的 GIS 運算容易在 Serverless 環境失敗，我們保留上方簡單的疊加邊界即可)
    # 若要達到圖片中「只有台灣有顏色，海洋全白」的效果，通常需要更重的 cartopy 庫
    # 這裡我們先做到「背景有顏色，但疊上台灣縣市框」，或者設定背景色
    
    # 修正：為了讓海面變白，最簡單的方法是限制 X/Y 軸範圍，並接受方形背景，
    # 或是使用 geopandas.clip (但需要 raster data)。
    # 既然是 Python 腳本，我們用最簡單的「點圖 + 插值 + 縣市框」呈現。
    
    # 標示最高溫/最低溫的地點
    top1 = df.iloc[df['at'].argmax()]
    ax.text(top1['lon'], top1['lat'], f"最高體感\n{top1['city']}\n{top1['at']:.1f}°C", 
            fontsize=12, color='black', fontweight='bold', ha='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # 標題與修飾
    current_time = datetime.now(TP_TZ).strftime('%Y-%m-%d %H:%M')
    plt.title(f"全臺體感溫度分佈圖 (Apparent Temp)\n{current_time}", fontsize=18)
    plt.xlabel('經度')
    plt.ylabel('緯度')
    
    # 加入 Colorbar
    cbar = plt.colorbar(contour, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('體感溫度 (°C)', fontsize=12)
    
    # 鎖定顯示範圍
    ax.set_xlim(119.3, 122.3)
    ax.set_ylim(21.8, 25.4)
    
    plt.savefig('apparent_temp_map.png', dpi=150, bbox_inches='tight')
    print("圖表繪製完成: apparent_temp_map.png")

def save_json(df):
    """儲存 API 數據，供前端網站使用"""
    # 整理一下要輸出的欄位
    output_list = df[['city', 'town', 'name', 'temp', 'at', 'lat', 'lon']].to_dict(orient='records')
    
    output = {
        "meta": {
            "updated_at": datetime.now(TP_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            "source": "CWA O-A0001-001 (Auto Stations)",
            "formula": "Australian Apparent Temp",
            "count": len(output_list)
        },
        "data": output_list
    }
    
    with open('data_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("詳細數據儲存完成: data_detailed.json")

if __name__ == "__main__":
    try:
        raw_data = fetch_data()
        df = process_data(raw_data)
        
        # 確保有數據才畫圖
        if not df.empty:
            generate_heatmap(df)
            save_json(df)
            print("執行成功！")
        else:
            print("無有效數據可處理")
            
    except Exception as e:
        print(f"執行發生錯誤: {e}")
