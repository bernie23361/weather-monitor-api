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
import time

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

# =================極速遮罩=================

def fast_mask_grid(grid_x, grid_y, geo_df):
    log("啟動極速遮罩運算...")
    points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
    final_mask = np.zeros(len(points), dtype=bool)
    
    # 優化：直接迭代幾何物件
    for geom in geo_df.geometry:
        if geom.geom_type == 'Polygon':
            geoms = [geom]
        elif geom.geom_type == 'MultiPolygon':
            geoms = geom.geoms
        else:
            continue
            
        for poly in geoms:
            # Bounding Box 快速篩選 (加速關鍵)
            minx, miny, maxx, maxy = poly.bounds
            # 擴大一點邊界以免切到邊緣
            if maxx < 119 or minx > 123 or maxy < 21 or miny > 27:
                continue

            path = mpl_path.Path(np.array(poly.exterior.coords))
            mask = path.contains_points(points)
            final_mask = final_mask | mask
            
    return final_mask.reshape(grid_x.shape)

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
        resp = requests.get(url, timeout=20)
        return resp.json()
    except Exception as e:
        log(f"API 錯誤: {e}")
        return None

def process_data(raw_data):
    log("處理數據中...")
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
                
                if temp < -20 or temp > 50 or humid < 0 or humid > 100: continue
                at = calculate_apparent_temp(temp, humid, wind)
                if at is None: continue

                data = {"name": item["StationName"], "lat": lat, "lon": lon, "temp": temp, "at": at, "city": city, "town": item["GeoInfo"]["TownName"]}
                
                # 分流邏輯
                if city in km_counties:
                    km_matsu.append(data)
                else:
                    # 這裡放寬範圍，確保澎湖(約119.5)能被納入 mainland 進行插值
                    if 119.0 <= lon <= 122.5 and 21.5 <= lat <= 26.0:
                        mainland.append(data)
            except: continue
            
    return pd.DataFrame(mainland), pd.DataFrame(km_matsu)

def generate_heatmap(df_main, df_km):
    log("開始繪圖程序...")
    
    taiwan_map = gpd.read_file(MAP_URL)
    
    # [升級] 解析度提升至 800x800，讓邊緣更細緻
    # 範圍微調以涵蓋澎湖
    min_lon, max_lon = 119.0, 122.2
    min_lat, max_lat = 21.8, 25.4
    grid_x, grid_y = np.mgrid[min_lon:max_lon:800j, min_lat:max_lat:800j]
    
    # ================= 關鍵技術升級：混合插值法 =================
    log("執行混合插值 (Hybrid Interpolation)...")
    
    # 1. Cubic (平滑層)：畫出漂亮的漸層，但離島會是空白(NaN)
    grid_cubic = griddata(
        (df_main['lon'], df_main['lat']), 
        df_main['at'], 
        (grid_x, grid_y), 
        method='cubic', 
        fill_value=np.nan
    )
    
    # 2. Nearest (基底層)：填補所有空隙，確保澎湖有顏色
    grid_nearest = griddata(
        (df_main['lon'], df_main['lat']), 
        df_main['at'], 
        (grid_x, grid_y), 
        method='nearest'
    )
    
    # 3. 合併：若 Cubic 是 NaN (例如澎湖或邊緣)，就用 Nearest 補上
    # 這樣既有平滑的漸層，又不會讓離島消失
    mask_nan = np.isnan(grid_cubic)
    grid_cubic[mask_nan] = grid_nearest[mask_nan]
    grid_z = grid_cubic

    # ==========================================================

    log("應用地圖遮罩...")
    mask = fast_mask_grid(grid_x, grid_y, taiwan_map)
    grid_z[~mask] = np.nan

    log("輸出圖片...")
    fig, ax = plt.subplots(figsize=(10, 12))
    levels = np.linspace(0, 40, 200)
    norm = Normalize(vmin=0, vmax=40)
    
    # 使用 contourf 繪製平滑漸層
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap_custom, norm=norm, extend='both')
    
    # 繪製縣市框線
    taiwan_map.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.5, alpha=0.5, zorder=2)
    
    # 繪製金馬圓點
    if not df_km.empty:
        km_colors = cmap_custom(norm(df_km['at']))
        ax.scatter(df_km['lon'], df_km['lat'], c=km_colors, s=150, edgecolor='black', linewidth=1, zorder=5)
        for _, row in df_km.iterrows():
            if row['name'] in ['金門', '馬祖', '東引']:
                 ax.text(row['lon']+0.06, row['lat'], f"{row['name']}\n{row['at']:.1f}", fontsize=11, fontweight='bold', zorder=6, 
                         bbox=dict(facecolor='white', alpha=0.6, pad=1, edgecolor='none'))

    current_time = datetime.now(TP_TZ).strftime('%Y-%m-%d %H:%M')
    plt.title(f"全臺體感溫度分佈圖\n{current_time}", fontsize=20, fontweight='bold')
    
    cbar = plt.colorbar(contour, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_ticks(np.arange(0, 41, 5))
    cbar.ax.tick_params(labelsize=10)
    
    # 設定顯示範圍
    ax.set_xlim(118.0, 122.3)
    ax.set_ylim(21.7, 26.5)
    
    plt.savefig('apparent_temp_map.png', dpi=150, bbox_inches='tight')
    log("圖表已更新: apparent_temp_map.png")

def save_json(df_main, df_km):
    full_df = pd.concat([df_main, df_km], ignore_index=True)
    output = {
        "meta": {
            "updated_at": datetime.now(TP_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            "source": "CWA Auto Stations",
            "count": len(full_df)
        },
        "data": full_df[['city', 'town', 'name', 'temp', 'at', 'lat', 'lon']].to_dict(orient='records')
    }
    with open('data_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log("數據已更新: data_detailed.json")

if __name__ == "__main__":
    log("程式啟動")
    
    # [強制繪圖模式] 方便你驗收
    should_draw_map = True 

    try:
        raw = fetch_data()
        if raw:
            df_main, df_km = process_data(raw)
            if not df_main.empty:
                save_json(df_main, df_km)
                if should_draw_map:
                    generate_heatmap(df_main, df_km)
                else:
                    log("跳過繪圖")
            else:
                log("無有效資料")
        else:
            log("API 下載失敗")
    except Exception as e:
        import traceback
        traceback.print_exc()
        log(f"執行錯誤: {e}")
