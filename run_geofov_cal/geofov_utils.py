import os
import re
import math
import csv
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import xml.etree.ElementTree as ET


def parse_ground_surface(gml_path):
    """Extract ground surface polygon from a GML file."""
    try:
        tree = ET.parse(gml_path)
    except Exception as e:
        print(f"Error parsing {gml_path}: {e}")
        return None
    ns = {
        'bldg': 'http://www.opengis.net/citygml/building/2.0',
        'gml': 'http://www.opengis.net/gml'
    }
    elem = tree.getroot().find('.//bldg:GroundSurface', ns)
    if elem is None:
        print(f"{gml_path}: no GroundSurface found")
        return None
    pos = elem.find('.//gml:posList', ns)
    if pos is None or pos.text is None:
        print(f"{gml_path}: no posList found")
        return None
    try:
        coords = list(map(float, pos.text.split()))
    except Exception as e:
        print(f"{gml_path}: error parsing coordinates: {e}")
        return None
    points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 3)]
    if len(points) < 3:
        print(f"{gml_path}: insufficient coordinates")
        return None
    return Polygon(points)


def load_buildings(folder_path):
    """Load all building polygons from GML files in a folder."""
    buildings = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith('.gml'):
            path = os.path.join(folder_path, fname)
            poly = parse_ground_surface(path)
            if poly:
                bid = os.path.splitext(fname)[0]
                m = re.search(r'(\d+)$', bid)
                bid = m.group(1) if m else bid
                buildings.append({'building_id': bid, 'geometry': poly})
            else:
                print(f"Skipping {fname}")
    if not buildings:
        print("No building data found.")
    return buildings


def is_vertex_visible(view, vertex, target_id, buildings, eps=1e-6):
    """Check if a vertex is visible from a viewpoint."""
    ray = LineString([view, vertex])
    dist = view.distance(vertex)
    for b in buildings:
        if b['building_id'] == target_id:
            continue
        inter = ray.intersection(b['geometry'])
        if not inter.is_empty:
            if inter.geom_type == 'Point':
                if view.distance(inter) < dist - eps:
                    return False
            elif hasattr(inter, 'geoms'):
                if any(view.distance(pt) < dist - eps for pt in inter.geoms if pt.geom_type=='Point'):
                    return False
            else:
                first = Point(list(inter.coords)[0])
                if view.distance(first) < dist - eps:
                    return False
    return True


def calculate_bearing(origin, point):
    """Compute bearing angle from origin to point."""
    dx = point.x - origin.x
    dy = point.y - origin.y
    ang = math.degrees(math.atan2(dx, dy))
    return (ang + 360) % 360


def compute_point_fov(row, buildings, buffer_dist):
    """Compute FOV metrics for a single point."""
    metrics = []
    view_pt = row['geometry']
    buf = view_pt.buffer(buffer_dist)
    for b in buildings:
        geom = b['geometry']
        bid = b['building_id']
        if not buf.intersects(geom):
            continue
        coords = []
        if geom.geom_type == 'Polygon':
            coords = geom.exterior.coords
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                coords.extend(poly.exterior.coords)
        else:
            continue
        angles = []
        for coord in coords:
            pt = Point(coord)
            angle = calculate_bearing(view_pt, pt)
            if is_vertex_visible(view_pt, pt, bid, buildings):
                angles.append(angle)
        if not angles:
            continue
        angles.sort()
        n = len(angles)
        max_gap = -1
        idx = 0
        for i in range(n):
            gap = (angles[(i+1)%n] - angles[i]) % 360
            if gap > max_gap:
                max_gap = gap
                idx = i
        left = angles[(idx+1)%n]
        right = angles[idx]
        fov = (360 - max_gap) % 360
        metrics.append({
            'pid': row['pid'],
            'lat': row['lat'],
            'lng': row['lng'],
            'degree': row['degree'],
            'building_id': bid,
            'left_angle_geo': left,
            'right_angle_geo': right,
            'fov_geo': fov
        })
    return metrics


def compute_batch_fov(batch, buildings, buffer_dist):
    results = []
    for _, row in batch.iterrows():
        results.extend(compute_point_fov(row, buildings, buffer_dist))
    return results


def run_fov_processing(df_points, buildings, city_crs, buffer_dist, output_csv, batch_size=10):
    gdf = gpd.GeoDataFrame(
        df_points,
        geometry=gpd.points_from_xy(df_points.lng, df_points.lat),
        crs='EPSG:4326'
    ).to_crs(city_crs)
    total = len(gdf)
    processed = 0
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pid','lat','lng','degree',
            'building_id','left_angle_geo','right_angle_geo','fov_geo'
        ])
        writer.writeheader()
        from concurrent.futures import ProcessPoolExecutor, as_completed
        batches = [gdf.iloc[i:i+batch_size] for i in range(0, total, batch_size)]
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(compute_batch_fov, batch, buildings, buffer_dist): batch for batch in batches}
            for future in as_completed(futures):
                for metric in future.result():
                    writer.writerow(metric)
                processed += len(futures[future])
                print(f"Processed {processed} of {total}")


def generate_ray_line(origin, bearing, length=80):
    """Create a line for a given bearing."""
    rad = math.radians(bearing)
    dx = length * math.sin(rad)
    dy = length * math.cos(rad)
    return LineString([origin, Point(origin.x + dx, origin.y + dy)])


def linewidth_from_distance(dist, min_dist, max_dist):
    """Determine line width based on distance."""
    if math.isclose(min_dist, max_dist):
        return 5.0
    return 1.0 + 4.0 * (max_dist - dist) / (max_dist - min_dist)


def plot_building_rays(csv_path, buildings_folder, city_crs, output_folder):
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df = pd.read_csv(csv_path, dtype={'building_id': str})
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.lng, df.lat),
        crs='EPSG:4326'
    ).to_crs(city_crs)
    buildings = load_buildings(buildings_folder)
    bdict = {b['building_id']: b['geometry'] for b in buildings}
    all_geoms = [b['geometry'] for b in buildings]
    all_gdf = gpd.GeoDataFrame({'geometry': all_geoms}, crs=city_crs)
    for bid, grp in gdf.groupby('building_id'):
        geom = bdict.get(bid)
        if geom is None:
            print(f"{bid} missing, skipping")
            continue
        centroid = geom.centroid
        grp['dist'] = grp.geometry.distance(centroid)
        grp = grp[~grp.geometry.within(geom)]
        grp = grp.nsmallest(3, 'dist')
        if grp.empty:
            print(f"{bid} no valid points")
            continue
        min_d, max_d = grp.dist.min(), grp.dist.max()
        fig, ax = plt.subplots(figsize=(12,12))
        all_gdf.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.3)
        gpd.GeoSeries([geom], crs=city_crs).plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.7, linewidth=2)
        for _, row in grp.iterrows():
            lw = linewidth_from_distance(row.dist, min_d, max_d)
            for key in ['left_angle_geo', 'right_angle_geo']:
                line = generate_ray_line(row.geometry, row[key])
                xs, ys = line.xy
                ax.plot(xs, ys, linewidth=lw)
            ax.plot(row.geometry.x, row.geometry.y, 'ro', markersize=8)
        ax.set_title(f"Camera Angle Rays for Building {bid}")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        outp = os.path.join(output_folder, f"camera_rays_{bid}.png")
        plt.savefig(outp, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {outp}")
