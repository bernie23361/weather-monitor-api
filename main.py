import os
import requests
import json
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import griddata
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
import pytz
import math
import warnings

# 忽略部分 GIS 運算警告
warnings.filterwarnings("ignore")

# =================設定區=================
# 設定中文字型
plt.rcParams['font.family'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

TP_TZ = pytz.timezone('Asia/Taipei')
CWA_KEY = os.environ.get("CWA_KEY")

# 縣市界線 GeoJSON
MAP_URL = "https://raw.githubusercontent.com/bernie23361/optix-map/refs/heads/main/COUNTY_MOI_1140318%20(1).json"

# =================客製化色階定義=================
# 依據參考圖檔建立 0度~40度 的漸層顏色
# 順序：深藍(冷) -> 藍綠 -> 黃綠 -> 黃 -> 橘 -> 紅 -> 紫(熱)
custom_colors = [
    '#2c7bb6', # 0度 - 深藍
    '#00a6ca', # ~5度
    '#00ccbc', # ~10度
    '#90eb9d', # ~15度
    '#e4e897', # ~20度
    '#ffff8c', # ~23度
    '#f9d057', # ~26度
    '#f29e2e', # ~30度
    '#d7191c', # ~35度
    '#a31f7b', # ~38度
    '#6e208a'  # 40度 - 深紫
]
# 建立 Matplotlib Colormap 物件 (切分為 256 階)
cmap_custom = LinearSegmentedColormap.from_list("taiwan_temp", custom_colors, N=256)

# =================功能函數=================

def calculate_apparent_temp(temp, humid, wind_speed):
    """計算體感溫度 (Australian Apparent Temperature)"""
    try:
        # 確保濕度在合理範圍
        humid = max(0, min(100, humid))
        # 計算水氣壓
        e = (humid / 100.0) * 6.105 * math.exp((17.27 * temp) / (237.7 + temp))
        # 風速不能為負
        ws = max(wind_speed, 0)
        # 公式
        at = temp + 0.33 * e - 0.70 * ws - 4.00
        return at
    except Exception:
        return None

def fetch_data():
    """抓取自動氣象站資料 (O-A0001-001)"""
    print("正在下載氣象署自動測站資料...")
    url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/O-A0001-001?Authorization={CWA_KEY}&format=JSON"
    # 設定 timeout 避免卡住
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status() # 檢查 HTTP 錯誤
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"下載失敗: {e}")
        return None

def process_data(raw_data):
    """處理資料，計算體感溫，並分離金馬地區"""
    print("正在處理數據並分離金馬地區...")
    mainland_data = []
    km_matsu_data = []
    
    # 金馬地區縣市名清單
    km_matsu_counties = ["金門縣", "連江縣"]

    if not raw_data or "records" not in raw_data or "Station" not in raw_data["records"]:
        return pd.DataFrame(), pd.DataFrame()

    for item in raw_data["records"]["Station"]:
        try:
            lat = float(item["GeoInfo"]["Coordinates"][0]["StationLatitude"])
            lon = float(item["GeoInfo"]["Coordinates"][0]["StationLongitude"])
            temp = float(item["WeatherElement"]["AirTemperature"])
            humid = float(item["WeatherElement"]["RelativeHumidity"])
            wind = float(item["WeatherElement"]["WindSpeed"])
            city = item["GeoInfo"]["CountyName"]
            
            # 排除儀器故障的極端值
            if temp < -15 or temp > 55 or humid < 0 or humid > 100:
                continue

            at = calculate_apparent_temp(temp, humid, wind)
            if at is None:
                continue

            data_point = {
                "name": item["StationName"],
                "lat": lat,
                "lon": lon,
                "temp": temp,
                "at": at,
                "city": city,
                "town": item["GeoInfo"]["TownName"]
            }
            
            # 資料分流
            if city in km_matsu_counties:
                km_matsu_data.append(data_point)
            else:
                # 簡單過濾台灣本島大致範圍外的錯誤站點
                if 119.0 <= lon <= 122.5 and 21.5 <= lat <= 25.5:
                    mainland_data.append(data_point)

        except (ValueError, KeyError, IndexError, TypeError):
            continue
                
    return pd.DataFrame(mainland_data), pd.DataFrame(km_matsu_data)

def generate_heatmap(df_main, df_km):
    """繪製專業體感溫度分佈圖 (含精確陸地遮罩)"""
    print("正在載入地圖並準備空間運算 (這需要一點時間)...")
    
    # 1. 讀取地圖資料
    taiwan_map = gpd.read_file(MAP_URL)
    # 將所有縣市合併成一個大的台灣形狀 (用於計算遮罩)
    taiwan_union = taiwan_map.union_all()
    
    # 2. 設定繪圖範圍與網格 (解析度設為 500x500 以求精細)
    # 範圍稍微縮窄聚焦本島
    min_lon, max_lon = 119.9, 122.1
    min_lat, max_lat = 21.8, 25.4
    
    grid_x, grid_y = np.mgrid[min_lon:max_lon:500j, min_lat:max_lat:500j]
    
    print("正在進行空間插值運算...")
    # 3. 插值運算 (計算整個方形區域的溫度)
    grid_z = griddata(
        (df_main['lon'], df_main['lat']), 
        df_main['at'], 
        (grid_x, grid_y), 
        method='linear',
        fill_value=np.nan # 延伸區域先填 NaN
    )

    print("正在應用陸地遮罩 (Masking)...")
    # 4. 精確陸地遮罩運算 (關鍵步驟)
    # 將網格點轉換為幾何點序列
    grid_points = [Point(xy) for xy in zip(grid_x.flatten(), grid_y.flatten())]
    grid_geoseries = gpd.GeoSeries(grid_points, crs=taiwan_map.crs)
    
    # 判斷哪些點在台灣陸地範圍內
    # within 運算較耗時，需耐心等待
    mask = grid_geoseries.within(taiwan_union)
    # 將布林遮罩轉回網格形狀
    mask_grid = mask.values.reshape(grid_x.shape)
    
    # 應用遮罩：將不在陸地內的點設為 NaN (透明)
    grid_z[~mask_grid] = np.nan

    # =================開始繪圖=================
    print("開始繪製圖表...")
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # 設定色階範圍固定在 0~40 度
    levels = np.linspace(0, 40, 200)
    norm = Normalize(vmin=0, vmax=40)
    
    # A. 繪製主島等高線填色圖
    # contourf 會自動忽略 NaN 值，達成海面透明效果
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap_custom, norm=norm, extend='both')
    
    # B. 疊加縣市界線 (細黑線，增加辨識度)
    taiwan_map.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.6, alpha=0.6, zorder=2)
    
    # C. 繪製金門馬祖獨立圓點
    if not df_km.empty:
        print("繪製金馬測站...")
        # 依據溫度取得對應顏色
        km_colors = cmap_custom(norm(df_km['at']))
        # 繪製圓點 (zorder設高一點確保在最上層)
        scatter = ax.scatter(
            df_km['lon'], df_km['lat'], 
            c=km_colors, s=120, edgecolor='black', linewidth=1, zorder=5
        )
        # 標示文字 (選擇代表性站點，避免重疊)
        for _, row in df_km.iterrows():
            # 只標示特定的代表站以保持整潔 (可自行調整)
            if row['name'] in ['金門', '馬祖', '東引']:
                 ax.text(row['lon']+0.02, row['lat'], 
                         f"{row['name']}\n{row['at']:.1f}°C",
                         fontsize=10, fontweight='bold', va='center', zorder=6,
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

    # D. 圖表修飾
    current_time = datetime.now(TP_TZ).strftime('%Y-%m-%d %H:%M')
    plt.title(f"全臺體感溫度分佈圖 (Apparent Temp)\n{current_time}", fontsize=18, pad=15)
    plt.xlabel('經度 E')
    plt.ylabel('緯度 N')
    
    # 調整 Colorbar
    cbar = plt.colorbar(contour, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('體感溫度 (°C)', fontsize=12)
    # 設定 Colorbar 的刻度顯示
    cbar.set_ticks(np.arange(0, 41, 5)) 
    
    # 鎖定最終顯示範圍 (包含金馬與本島)
    ax.set_xlim(118.0, 122.3)
    ax.set_ylim(21.8, 26.5)
    
    # 加個浮水印或來源標示
    plt.text(122.2, 21.9, 'Source: CWA Auto Stations\nViz: Python/GeoPandas', 
             fontsize=8, color='gray', ha='right')

    plt.savefig('apparent_temp_map.png', dpi=150, bbox_inches='tight')
    print("圖表存檔成功: apparent_temp_map.png")

def save_json(df_main, df_km):
    """儲存詳細資料 JSON (合併本島與金馬)"""
    # 合併兩個 Dataframe
    full_df = pd.concat([df_main, df_km], ignore_index=True)
    
    output_list = full_df[['city', 'town', 'name', 'temp', 'at', 'lat', 'lon']].to_dict(orient='records')
    
    output = {
        "meta": {
            "updated_at": datetime.now(TP_TZ).strftime('%Y-%m-%d %H:%M:%S'),
            "source": "CWA O-A0001-001",
            "count": len(output_list),
            "colormap_range": "0-40C"
        },
        "data": output_list
    }
    
    with open('data_detailed.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("JSON 存檔成功: data_detailed.json")

if __name__ == "__main__":
    try:
        raw_data = fetch_data()
        if raw_data:
            # 資料分流處理
            df_main, df_km = process_data(raw_data)
            
            if not df_main.empty:
                # 傳入兩組資料進行繪圖
                generate_heatmap(df_main, df_km)
                # 儲存合併資料
                save_json(df_main, df_km)
                print("全部執行完成！")
            else:
                print("本島有效資料不足，無法繪圖。")
        else:
            print("無法獲取 API 資料。")
            
    except Exception as e:
        # 捕捉並印出完整的錯誤訊息以便 Debug
        import traceback
        traceback.print_exc()
        print(f"發生嚴重錯誤: {e}")
