import geopandas as gpd
from shapely.geometry import Point, LineString
import pandas as pd
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import math

def generate_points_along_geometry(gdf, distance):
    points_with_id = []
    for idx, row in gdf.iterrows():
        geometry = row['geometry']
        building_id = str(row['osmid'])
        if geometry.geom_type == 'Polygon':
            exterior = geometry.exterior
        elif geometry.geom_type == 'LineString':
            exterior = geometry
        else:
            continue
        length = exterior.length
        current_distance = 0
        while current_distance <= length:
            point = exterior.interpolate(current_distance)
            points_with_id.append((point, building_id))
            current_distance += distance
    return points_with_id

def create_view_lines(view_point, target_points_with_id):
    lines_with_target = []
    for point, building_id in target_points_with_id:
        line = LineString([view_point, point])
        lines_with_target.append((line, building_id))
    return lines_with_target

def remove_intersecting_lines(lines, gdf):
    non_intersecting_lines = []
    for line, building_id in lines:
        if not gdf.intersects(line).any():
            non_intersecting_lines.append((line, building_id))
    return non_intersecting_lines

def calculate_angle(view_point, target_point):
    delta_x = target_point.x - view_point.x
    delta_y = target_point.y - view_point.y
    angle_radians = math.atan2(delta_x, delta_y)
    angle_degrees = math.degrees(angle_radians)
    bearing = (angle_degrees + 360) % 360
    return bearing

def process_point(row, gdf, buffer, distance_between_points):
    fov_metrics = []
    view_point = row['geometry']
    # selected_building_id = row['building_id']
    # print(view_point)

    view_point_buffer = view_point.buffer(buffer)
    target_buildings = gdf[gdf.intersects(view_point_buffer)]
    # target_building = buildings_within_buffer[buildings_within_buffer['building_id'] == selected_building_id]
    # print(target_buildings)

    points = generate_points_along_geometry(target_buildings, distance_between_points)
    lines = create_view_lines(view_point, points)
    unobstructed_lines = remove_intersecting_lines(lines, target_buildings)
    building_lines = {}
    # print(target_buildings)

    for line, building_id in unobstructed_lines:
        view_point = Point(line.coords[0])
        target_point = Point(line.coords[1])
        angle = calculate_angle(view_point, target_point)
        if building_id not in building_lines:
            building_lines[building_id] = []
        building_lines[building_id].append(angle)
        
    for building_id, angles in building_lines.items():
        angles.sort()
        largest_fov = 0
        angle_pair = (0, 0)
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                fov = min((angles[i] - angles[j]) % 360, (angles[j] - angles[i]) % 360)
                if fov > largest_fov:
                    largest_fov = fov
                    angle_pair = (angles[i], angles[j])
        total_fov = largest_fov
        if total_fov >= 180:
            total_fov = 360 - largest_fov

        if max(angle_pair) - min(angle_pair) < 180:
            fov_metrics.append({
                "pid": row['pid'],
                "lat": row['lat'],
                "lng": row['lng'],
                "degree": row['degree'],
                "building_id": building_id,
                "left_angle_geo": min(angle_pair),
                "right_angle_geo": max(angle_pair),
                "fov_geo": total_fov
            })
        else:
            fov_metrics.append({
                "pid": row['pid'],
                "lat": row['lat'],
                "lng": row['lng'],
                "degree": row['degree'],
                "building_id": building_id,
                "left_angle_geo": max(angle_pair),
                "right_angle_geo": min(angle_pair),
                "fov_geo": total_fov})

    return fov_metrics


def process_batch(batch, gdf, buffer, distance_between_points):
    results = []
    for _, row in batch.iterrows():
        result = process_point(row, gdf, buffer, distance_between_points)
        results.extend(result)
    return results

def geo_fov_calculator_visibility_batch(df_points, json_dir, city_crs, buffer, distance_between_points, out_dir, batch_size=10):
    gdf_points = gpd.GeoDataFrame(df_points, geometry=gpd.points_from_xy(df_points.lng, df_points.lat))
    gdf_points.crs = "EPSG:4326"
    gdf_point_meter = gdf_points.to_crs(city_crs)
    # gdf_point_meter['building_id'] = gdf_point_meter['building_id's.astype(str)
    
    bdfp = gpd.read_file(json_dir)
    gdf = bdfp.to_crs(city_crs)
    gdf['osmid'] = gdf['osmid'].astype(int).astype(str)
    total_points = len(gdf_point_meter)
    processed_count = 0
    
    # Open CSV file outside the loop for appending
    with open(out_dir, 'a', newline='') as f:
        fieldnames = ["pid", "lat", "lng", "degree", "building_id", "left_angle_geo", "right_angle_geo", "fov_geo"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Check if file is empty to write header
        f.seek(0, 2)  # Move to the end of the file
        if f.tell() == 0:  # Check if file is empty
            writer.writeheader()  # Write header if empty
        
        with ProcessPoolExecutor(max_workers=8) as executor:
            # Create batches
            batches = [gdf_point_meter.iloc[i:i + batch_size] for i in range(0, total_points, batch_size)]
            future_to_batch = {executor.submit(process_batch, batch, gdf, buffer, distance_between_points): batch for batch in batches}
            
            for future in as_completed(future_to_batch):
                fov_metrics = future.result()
                for metric in fov_metrics:
                    writer.writerow(metric)
                processed_count += len(future_to_batch[future])
                # print(f"Processed {processed_count} of {total_points}")
                if processed_count % 5 == 0:
                    print(f"Processed {processed_count} of {total_points}")


if __name__ == "__main__":
    # Define paths and parameters
    degree_dir = 'data/00_img_metadata/gsv_metadata.csv' #'/path/to/your/degree.csv'
    json_dir = 'data/03_bd_footprint/Munich.geojson' #'/path/to/your/json.geojson'
    out_dir = 'data/02_geofov/geofov_new.csv' # '/path/to/your/output.csv'
    city_crs = 'EPSG:31468'   # "EPSG:YourCRS"
    df_points = pd.read_csv(degree_dir)
    buffer = 50
    distance_between_points = 0.2
    geo_fov_calculator_visibility_batch(df_points, json_dir, city_crs, buffer, distance_between_points, out_dir)
