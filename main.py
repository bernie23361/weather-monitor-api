import os
import requests
import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.path as mpl_path
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import griddata
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
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

# =================核心加速演算法=================

def fast_mask_grid(grid_x, grid_y, geo_df):
    """
    [極速版] 陸地遮罩運算
    使用 Matplotlib Path (C後端) 取代 Shapely within (Python迴圈)，
    效能提升約 100 倍。
    """
    print("啟動極速遮罩運算...")
    # 將網格展平為點陣列
    points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    
    # 建立一個全為 False 的遮罩
    final_mask = np.zeros(len(points), dtype=bool)
    
    # 取出台灣地圖的所有多邊形
    # 為了加速，我們直接操作幾何物件
    polys = geo_df.geometry
    
    for geom in polys:
        # 處理 MultiPolygon 與 Polygon
        if geom.geom_type == 'Polygon':
            geoms = [geom]
        elif geom.geom_type == 'MultiPolygon':
            geoms = geom.geoms
        else:
            continue
            
        for poly in geoms:
            # 建立邊界路徑
            path = mpl_path.Path(np.array(poly.exterior.coords))
            # 判斷點是否在多邊形內 (核心加速點)
            mask = path.contains_points(points)
            # 聯集運算 (只要在任一縣市內就算在陸地)
            final_mask = final_mask | mask
            
    # 轉回網格形狀
    return final_mask.reshape(grid_x.shape)

# =================常規功能函數=================

def calculate_apparent_temp(temp, humid, wind_speed):
    try:
        humid = max(0, min(100, humid))
        e = (humid / 100.0) * 6.105 * math.exp((17.27 * temp) / (237.7 + temp))
        ws = max(wind_speed, 0)
        return temp + 0.33 * e - 0.70 * ws - 4.00
    except:
        return None

def fetch_data():
    print("下載氣象資料...")
    try:
        url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001?Authorization={CWA_KEY}&format=JSON"
        resp = requests.get(url, timeout=30)
        return resp.json()
    except Exception as e:
        print(f"API 錯誤: {e}")
        return None

def process_data(raw_data):
    print("處理氣象數據...")
    mainland, km_matsu = [], []
    km_counties = ["金門縣", "連江縣"]

    if raw_data and "records" in raw_data and "Station" in raw_data["records"]:
        for item in raw_data["records"]["Station"]:
            try:
                lat = float(item["GeoInfo"]["Coordinates"][0]["StationLatitude"])
                lon = float(item["GeoInfo"]["Coordinates"][0]["StationLongitude"])
                temp = float(item["WeatherElement"]["AirTemperature"])
                humid = float(item["WeatherElement"]["RelativeHumidity"])
                wind = float(item["WeatherElement"]["WindSpeed"])
                city = item["GeoInfo"]["CountyName"]
                
                if temp < -15 or temp > 55 or humid < 0 or humid > 100: continue
                at = calculate_apparent_temp(temp, humid, wind)
                if at is None: continue

                data = {"name": item["StationName"], "lat": lat, "lon": lon, "temp": temp, "at": at, "city": city, "town": item["GeoInfo"]["TownName"]}
                
                if city in km_counties:
                    km_matsu.append(data)
                else:
                    if 119.0 <= lon <= 122.5 and 21.5 <= lat <= 25.5:
                        mainland.append(data)
            except: continue
            
    return pd.DataFrame(mainland), pd.DataFrame(km_matsu)

def generate_heatmap(df_main, df_km):
    """繪圖函數"""
    print("正在繪製地圖 (這一步現在會很快)...")
    taiwan_map = gpd.read_file(MAP_URL)
    
    # 解析度維持高畫質 (500x500)
    min_lon, max_lon = 119.9, 122.1
    min_lat, max_lat = 21.8, 25.4
    grid_x, grid_y = np.mgrid[min_lon:max_lon:500j, min_lat:max_lat:500j]
    
    # 1. 插值
    grid_z = griddata((df_main['lon'], df_main['lat']), df_main['at'], (grid_x, grid_y), method='linear', fill_value=np.nan)

    # 2. [加速版] 應用遮罩
    mask = fast_mask_grid(grid_x, grid_y, taiwan_map)
    grid_z[~mask] = np.nan

    # 3. 繪圖
    fig, ax = plt.subplots(figsize=(10, 12))
    levels = np.linspace(0, 40, 200)
    norm = Normalize(vmin=0, vmax=40)
    
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap_custom, norm=norm, extend='both')
    taiwan_map.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.6, alpha=0.6, zorder=2)
    
    # 金馬圓點
    if not df_km.empty:
        km_colors = cmap_custom(norm(df_km['at']))
        ax.scatter(df_km['lon'], df_km['lat'], c=km_colors, s=120, edgecolor='black', linewidth=1, zorder=5)
        for _, row in df_km.iterrows():
            if row['name'] in ['金門', '馬祖', '東引']:
                 ax.text(row['lon']+0.05, row['lat'], f"{row['name']}\n{row['at']:.1f}", fontsize=10, fontweight='bold', zorder=6, bbox=dict(facecolor='white', alpha=0.5, pad=0.5))

    current_time = datetime.now(TP_TZ).strftime('%Y-%m-%d %H:%M')
    plt.title(f"全臺體感溫度分佈圖\n{current_time}", fontsize=18)
    
    cbar = plt.colorbar(contour, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_ticks(np.arange(0, 41, 5))
    
    ax.set_xlim(118.0, 122.3)
    ax.set_ylim(21.8, 26.5)
    plt.savefig('apparent_temp_map.png', dpi=150, bbox_inches='tight')
    print("圖表已更新: apparent_temp_map.png")

def save_json(df_main, df_km):
    full_df = pd.concat([df_main, df_km], ignore_index=True)
    output = {
        "meta": {
            "updated_at": datetime.now(TP_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            "source": "CWA Auto Stations",
            "note": "Data updates every 30m, Map updates hourly."
        },
        "data": full_df[['city', 'town', 'name', 'temp', 'at', 'lat', 'lon']].to_dict(orient='records')
    }
    with open('data_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("數據已更新: data_detailed.json")

if __name__ == "__main__":
    # 取得現在的「分鐘數」
    now_minute = datetime.now(TP_TZ).minute
    
    # 判斷邏輯：
    # 1. 如果分鐘數 < 15 (接近整點)，跑全套 (更新資料 + 畫圖)
    # 2. 否則 (例如 30 分)，只跑半套 (只更新資料)
    # 備註：你也可以強制設為 True 來測試
    should_draw_map = (now_minute < 15)
    
    print(f"現在時間: {datetime.now(TP_TZ).strftime('%H:%M')}")
    print(f"動作模式: {'[全套更新: 圖+資料]' if should_draw_map else '[快速更新: 僅資料]'}")

    try:
        raw = fetch_data()
        if raw:
            df_main, df_km = process_data(raw)
            if not df_main.empty:
                # 永遠更新 JSON 數據
                save_json(df_main, df_km)
                
                # 只有整點時才畫圖
                if should_draw_map:
                    generate_heatmap(df_main, df_km)
                else:
                    print("非整點時段，跳過繪圖步驟。")
            else:
                print("無有效資料。")
    except Exception as e:
        print(f"執行錯誤: {e}")
