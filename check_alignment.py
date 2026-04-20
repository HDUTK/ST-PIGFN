# check_csv_range.py
import pandas as pd
df = pd.read_csv('./data/grotto_shape_points.csv')
print("=== 当前形状文件坐标范围 ===")
print(f"X Range: {df['x'].min()} ~ {df['x'].max()}")
print(f"Y Range: {df['y'].min()} ~ {df['y'].max()}")
print(f"Z Range: {df['z'].min()} ~ {df['z'].max()}")