import os
import io
import base64
import lancedb
import pyarrow as pa
import pandas as pd
import numpy as np
from PIL import Image

def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def b64_to_img(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))

class PersonDatabase:
    def __init__(self, db_uri="database/lancedb", table_name="identities", threshold=0.67):
        self.db_uri = db_uri
        self.table_name = table_name
        self.threshold = threshold
        os.makedirs(db_uri, exist_ok=True)
        self.db = lancedb.connect(db_uri)
        self.table = None
        self._load_table()

    def _load_table(self):
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)

    def _next_id(self) -> int:
        if self.table is None or self.table.count_rows() == 0:
            return 1
        df = self.table.to_pandas()
        return int(df["id"].max()) + 1

    def search(self, emb: np.ndarray):
        if self.table is None or self.table.count_rows() == 0:
            return None, None, None

        results = (
            self.table
            .search(np.array(emb, dtype=np.float32), vector_column_name="embedding")
            .metric("dot")
            .limit(1)
            .to_pandas()
        )

        if results.empty:
            return None, None, None

        top = results.iloc[0]
        similarity = 1 - float(top["_distance"])

        if similarity >= self.threshold:
            return int(top["id"]), top["label"], similarity

        return None, None, similarity

    def create_identity(self, emb: np.ndarray, thumbnail: Image.Image) -> int:
        new_id = self._next_id()
        emb_f32 = np.array(emb, dtype=np.float32)
        dim = len(emb_f32)

        record = {
            "id": new_id,
            "label": None,
            "thumbnail": img_to_b64(thumbnail),
            "embedding": emb_f32.tolist(),
        }

        if self.table is None:
            schema = pa.schema([
                pa.field("id", pa.int32()),
                pa.field("label", pa.string()),
                pa.field("thumbnail", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), dim)),
            ])
            self.table = self.db.create_table(self.table_name, data=[record], schema=schema)
        else:
            self.table.add([record])

        return new_id

    def add_embedding(self, uid: int, emb: np.ndarray, thumbnail: Image.Image):
        if self.table is None:
            return

        df = self.table.search().where(f"id = {uid}").limit(10).to_pandas()
        label = df.iloc[0]["label"] if not df.empty else None

        record = {
            "id": uid,
            "label": label,
            "thumbnail": img_to_b64(thumbnail),
            "embedding": np.array(emb, dtype=np.float32).tolist(),
        }
        self.table.add([record])

    def assign_label_and_merge(self, source_uid: int, new_label: str):
        if self.table is None or not new_label.strip():
            return

        label = new_label.strip()
        df_all = self.table.to_pandas()

        existing = df_all[
            (df_all["label"] == label) & (df_all["id"] != source_uid)
        ]

        if not existing.empty:
            other_uid = int(existing.iloc[0]["id"])
            keep_uid = min(source_uid, other_uid)
            drop_uid = max(source_uid, other_uid)

            df_keep = df_all[df_all["id"] == keep_uid].copy()
            df_drop = df_all[df_all["id"] == drop_uid].copy()

            df_keep["label"] = label
            df_drop["id"] = keep_uid
            df_drop["label"] = label

            self.table.delete(f"id = {keep_uid}")
            self.table.delete(f"id = {drop_uid}")

            combined = pd.concat([df_keep, df_drop], ignore_index=True)
            self.table.add(combined.to_dict(orient="records"))

        else:
            self.table.update(where=f"id = {source_uid}", values={"label": label})

    def get_ui_options(self) -> list:
        if self.table is None or self.table.count_rows() == 0:
            return []
        df = self.table.to_pandas()
        labels = df["label"].dropna().unique().tolist()
        return sorted([str(l) for l in labels if str(l).strip()])