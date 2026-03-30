
from ipyleaflet import Map, GeoJSON, LayersControl, SearchControl, Marker
from ipywidgets import Button, VBox, Label, HTML
from shapely.geometry import box
import datetime

def create_interactive_map(
    gdf,
    layer_name="Layer",
    zoom=10,
    center=None,
    filter_zoom=10,
    point_style=None,
    log_file=None,
    show_search=False,
    info_popup=False,
    extra_properties=None,
    style_callback=None,
    dynamic_radius=False
):
    # Compute center if not provided
    if center is None:
        bounds = gdf.total_bounds
        center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]

    # Create map
    m = Map(center=center, zoom=zoom, scroll_wheel_zoom=True, layout={'height': '700px'})
    geo_json_layer = GeoJSON(data={"type": "FeatureCollection", "features": []}, name=layer_name)
    if point_style:
        geo_json_layer.point_style = point_style
    m.add_layer(geo_json_layer)
    m.add_control(LayersControl(position='topright'))

    # Optional search control
    if show_search:
        search_marker = Marker(location=center)
        search_control = SearchControl(
            position="topleft",
            url="https://nominatim.openstreetmap.org/search?format=json&q={s}",
            zoom=12,
            property_name="display_name",
            marker=search_marker
        )
        m.add_control(search_control)

    # Status and info widgets
    status_label = Label(value="Ready")
    info_box = HTML("<b>Select a feature</b>", layout={'width': '300px', 'height': '150px'}) if info_popup else None

    # Logging function
    def log_debug(message):
        if log_file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, "a") as f:
                f.write(f"[{timestamp}] {message}\n")

    # Click handler for info popup
    def on_feature_click(event, feature, **kwargs):
        props = feature.get("properties", {})
        info_html = "<b>Feature Info:</b><br>" + "<br>".join([f"{k}: {v}" for k, v in props.items() if k != "style"])
        if info_box:
            info_box.value = info_html  # ✅ This is correct for updating HTML content

    # Manual update function
    def manual_update(_):
        bounds = m.bounds
        zoom_level = m.zoom
        status_label.value = f"Running for zoom level: {zoom_level}."
        log_debug(f"Manual update triggered. Zoom: {zoom_level}, Bounds: {bounds}")

        if zoom_level >= filter_zoom:
            bbox = box(bounds[0][1], bounds[0][0], bounds[1][1], bounds[1][0])
            filtered = gdf[gdf.intersects(bbox)]
            log_debug(f"Filtered features: {len(filtered)}")

            features = []
            for _, row in filtered.iterrows():
                props = {}
                if extra_properties:
                    props.update(extra_properties(row) if callable(extra_properties) else extra_properties)
                props["style"] = {"color": "blue", "weight": 2, "fillOpacity": 0.5}
                features.append({"type": "Feature", "geometry": row.geometry.__geo_interface__, "properties": props})

            geo_json_layer.data = {"type": "FeatureCollection", "features": features}

            # Dynamic radius for point layers
            if dynamic_radius and point_style:
                geo_json_layer.point_style["radius"] = max(2, zoom_level - 9.5)

            # Apply custom style callback if provided
            if style_callback:
                geo_json_layer.style_callback = style_callback

            # Enable info popup
            if info_popup:
                geo_json_layer.on_click(on_feature_click)

            status_label.value = f"Updated map with {len(filtered)} features."
        else:
            geo_json_layer.data = {"type": "FeatureCollection", "features": []}
            status_label.value = f"Zoom further in to see {layer_name}."

    # Button for manual trigger
    btn = Button(description="Update Map")
    btn.on_click(manual_update)

    widgets = [m, btn, status_label]
    if info_popup:
        widgets.append(info_box)

    return VBox(widgets)
