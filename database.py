import os
import io
import base64
import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
from PIL import Image

THRESHOLD_CONFIDENT  = 0.80
THRESHOLD_UNCERTAIN  = 0.69
MIN_EMBEDDINGS_UNCERTAIN = 5
TOP_K_VOTE           = 5

def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def b64_to_img(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))

class PersonDatabase:

    def __init__(
        self,
        db_uri: str    = "database/lancedb",
        table_name: str = "identities",
    ) -> None:
        self.db_uri     = db_uri
        self.table_name = table_name

        os.makedirs(db_uri, exist_ok=True)
        self.db    = lancedb.connect(db_uri)
        self.table = None
        self._load_table()

    def _load_table(self) -> None:
        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)

    def _next_id(self) -> int:
        if self.table is None or self.table.count_rows() == 0:
            return 1
        return int(self.table.to_pandas()["id"].max()) + 1

    def _create_table_with_record(self, record: dict, dim: int) -> None:
        schema = pa.schema([
            pa.field("id",        pa.int32()),
            pa.field("label",     pa.string()),
            pa.field("thumbnail", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), dim)),
        ])
        self.table = self.db.create_table(
            self.table_name, data=[record], schema=schema
        )

    def _embedding_count(self, uid: int, df: pd.DataFrame | None = None) -> int:
        if self.table is None or self.table.count_rows() == 0:
            return 0
        if df is None:
            df = self.table.to_pandas()
        return int((df["id"] == uid).sum())

    def get_embedding_count(self, uid: int) -> int:
        return self._embedding_count(uid)

    def search(
        self,
        emb: np.ndarray,
        exclude_uids: set[int] | list[int] | None = None,
    ) -> tuple[int | None, str | None, float | None]:
        if self.table is None or self.table.count_rows() == 0:
            return None, None, None

        k = max(TOP_K_VOTE, 1)
        query = (
            self.table
            .search(np.array(emb, dtype=np.float32), vector_column_name="embedding")
            .metric("dot")
        )

        if exclude_uids:
            exclude_str = ", ".join(map(str, exclude_uids))
            if exclude_str:
                query = query.where(f"id NOT IN ({exclude_str})")

        results = query.limit(k).to_pandas()

        if results.empty:
            return None, None, None

        results["similarity"] = 1.0 - results["_distance"].astype(float)
        top1       = results.iloc[0]
        best_sim   = float(top1["similarity"])

        if best_sim >= THRESHOLD_CONFIDENT:
            return int(top1["id"]), top1["label"], best_sim

        if best_sim < THRESHOLD_UNCERTAIN:
            return None, None, best_sim

        candidates = results[results["similarity"] >= THRESHOLD_UNCERTAIN]

        if candidates.empty:
            return None, None, best_sim

        vote_counts = candidates["id"].value_counts()
        winner_uid  = int(vote_counts.idxmax())
        winner_rows = candidates[candidates["id"] == winner_uid]
        winner_sim  = float(winner_rows["similarity"].max())

        emb_count = self._embedding_count(winner_uid)
        if emb_count < MIN_EMBEDDINGS_UNCERTAIN:
            return None, None, best_sim

        winner_label = winner_rows.iloc[0]["label"]
        return winner_uid, winner_label, winner_sim

    def create_identity(self, emb: np.ndarray, thumbnail: Image.Image) -> int:
        new_id  = self._next_id()
        emb_f32 = np.array(emb, dtype=np.float32)
        record  = {
            "id":        new_id,
            "label":     None,
            "thumbnail": img_to_b64(thumbnail),
            "embedding": emb_f32.tolist(),
        }

        if self.table is None:
            self._create_table_with_record(record, dim=len(emb_f32))
        else:
            self.table.add([record])

        return new_id

    def add_embedding(self, uid: int, emb: np.ndarray, thumbnail: Image.Image) -> None:
        if self.table is None:
            return

        rows  = self.table.search().where(f"id = {uid}").limit(1).to_pandas()
        label = rows.iloc[0]["label"] if not rows.empty else None

        record = {
            "id":        uid,
            "label":     label,
            "thumbnail": img_to_b64(thumbnail),
            "embedding": np.array(emb, dtype=np.float32).tolist(),
        }
        self.table.add([record])

    def assign_label_and_merge(self, source_uid: int, new_label: str) -> None:
        if self.table is None or not new_label.strip():
            return

        label  = new_label.strip()
        df_all = self.table.to_pandas()

        existing = df_all[
            (df_all["label"] == label) & (df_all["id"] != source_uid)
        ]

        if not existing.empty:
            other_uid = int(existing.iloc[0]["id"])
            keep_uid  = min(source_uid, other_uid)
            drop_uid  = max(source_uid, other_uid)

            df_keep = df_all[df_all["id"] == keep_uid].copy()
            df_drop = df_all[df_all["id"] == drop_uid].copy()

            df_keep["label"] = label
            df_drop["id"]    = keep_uid
            df_drop["label"] = label

            self.table.delete(f"id = {keep_uid}")
            self.table.delete(f"id = {drop_uid}")

            combined = pd.concat([df_keep, df_drop], ignore_index=True)
            self.table.add(combined.to_dict(orient="records"))
        else:
            self.table.update(where=f"id = {source_uid}", values={"label": label})

    def get_ui_options(self) -> list[str]:
        if self.table is None or self.table.count_rows() == 0:
            return []

        df     = self.table.to_pandas()
        labels = df["label"].dropna().unique().tolist()
        return sorted(str(l) for l in labels if str(l).strip())