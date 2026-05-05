import pandas as pd
import umap
import plotly.graph_objects as go
import plotly.express as px  # Added for high-contrast color palettes
import webbrowser
import os
import numpy as np
import sklearn.cluster as cluster
import threading
import http.server
import socketserver
from database import PersonDatabase

db = PersonDatabase()
if db.table is None or db.table.count_rows() == 0:
    print("Database is empty. Nothing to visualize.")
    exit()

# 1. DATA PREPARATION
df = db.table.to_pandas()
embeddings = np.stack(df["embedding"].values)

# 2. DIMENSIONALITY REDUCTION
print("Computing UMAP projection...")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
umap_2d = reducer.fit_transform(embeddings)
df["x"], df["y"] = umap_2d[:, 0], umap_2d[:, 1]

# 3. AUTO-LABELING
if df["label"].isnull().all() or (df["label"] == "None").all():
    print("Labels are missing. Running HDBSCAN...")
    df["label"] = cluster.HDBSCAN(min_cluster_size=5).fit_predict(umap_2d).astype(str)

# 4. BASE64 IMAGE FORMATTING
df["image_src"] = "data:image/jpeg;base64," + df["thumbnail"]

# 5. BUILD PLOTLY FIGURE (White Theme & High Contrast)
# Using 'Alphabet' or 'Bold' ensures colors are as far apart as possible
color_sequence = px.colors.qualitative.Alphabet 

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["x"],
    y=df["y"],
    mode="markers",
    marker=dict(
        size=10,
        color=pd.factorize(df["label"])[0],
        colorscale=color_sequence,
        line=dict(width=0.5, color="#444"), # Subtle border for marker definition
        opacity=0.9
    ),
    customdata=np.stack((df["label"], df["id"], df["image_src"]), axis=-1),
    hovertemplate=(
        "<b>Label:</b> %{customdata[0]}<br>"
        "<b>ID:</b> %{customdata[1]}"
        "<extra></extra>"
    )
))

fig.update_layout(
    title=dict(text="Latent Space Analysis", font=dict(color="#111", size=20)),
    template="plotly_white", # Force white background
    paper_bgcolor="white",
    plot_bgcolor="white",
    hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial", font_color="black"),
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    width=1200,
    height=850
)

# 6. BUILD FULL HTML MANUALLY
plotly_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Embedding Explorer</title>
</head>
<body style="background:#ffffff; margin:0; padding:0;">

<!-- High-Contrast Floating thumbnail overlay -->
<div id="hover-img-box" style="
    position: fixed;
    display: none;
    pointer-events: none;
    z-index: 9999;
    border: 1px solid #ccc;
    border-radius: 8px;
    background: #ffffff;
    padding: 8px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
">
    <img id="hover-img" src="" width="200" style="display:block; border-radius:4px;">
    <div id="hover-label" style="
        color: #111;
        font-size: 12px;
        font-weight: bold;
        font-family: sans-serif;
        text-align: center;
        padding-top: 8px;
        max-width: 200px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    "></div>
</div>

{plotly_html}

<script>
(function() {{
    var checkInterval = setInterval(function() {{
        var gd = document.querySelector('.js-plotly-plot');
        if (!gd) return;
        clearInterval(checkInterval);

        var box   = document.getElementById('hover-img-box');
        var img   = document.getElementById('hover-img');
        var label = document.getElementById('hover-label');

        gd.on('plotly_hover', function(data) {{
            var pt = data.points[0];
            var cd = pt.customdata;
            img.src           = cd[2];
            label.textContent = cd[0] + ' | ID: ' + cd[1];
            box.style.display = 'block';
        }});

        gd.on('plotly_unhover', function() {{
            box.style.display = 'none';
        }});

        document.addEventListener('mousemove', function(e) {{
            var offset = 25;
            var bw = box.offsetWidth  || 220;
            var bh = box.offsetHeight || 250;
            var x  = e.clientX + offset;
            var y  = e.clientY + offset;

            if (x + bw > window.innerWidth)  x = e.clientX - bw - offset;
            if (y + bh > window.innerHeight) y = e.clientY - bh - offset;

            box.style.left = x + 'px';
            box.style.top  = y + 'px';
        }});
    }}, 100);
}})();
</script>
</body>
</html>"""

# 7. EXPORT & SERVE
output_file = "embedding_explorer.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(full_html)

PORT = 8050
def serve():
    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *args: None
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        httpd.serve_forever()

threading.Thread(target=serve, daemon=True).start()
webbrowser.open(f"http://localhost:{PORT}/{output_file}")
input("Server active on port 8050. Press Enter to terminate.\n")