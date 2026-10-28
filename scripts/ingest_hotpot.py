"""Resumable/idempotent upload of prepared HotpotQA passages to Qdrant Cloud.

Use a DIFFERENT collection from the earlier infra_probe prototype. Upload from
your local computer, not from a Render Free service's ephemeral filesystem.
"""
from __future__ import annotations
import argparse, json, os, uuid
from pathlib import Path


def make_point(row: dict, model_name: str):
    from qdrant_client import models
    return models.PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, "hotpotqa:" + row["source_id"])),
        vector={ "dense": models.Document(text=f"{row['title']}\n{row['text']}", model=model_name),
                 "bm25": models.Document(text=f"{row['title']}\n{row['text']}", model="qdrant/bm25"),
        },
        payload={
            "source_id": row["source_id"],
            "title": row["title"],
            "text": row["text"],
            "source_type": row["source_type"],
            "origin_split": row.get("origin_split", ""),
            "origin_example_id": row.get("origin_example_id", ""),
        },
    )


def progress_filename(records_path: Path, collection: str) -> Path:
    return records_path.with_name(f".{records_path.stem}.{collection}.upload.json")


def save_progress(path: Path, state: dict):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp.replace(path)


def upload(records_path: Path, *, collection: str, batch_size: int, 
           max_records: int | None = None, restart: bool = False):
    from dotenv import load_dotenv
    from qdrant_client import QdrantClient, models
    load_dotenv()
    url = os.environ.get("QDRANT_URL")
    key = os.environ.get("QDRANT_API_KEY")
    if not url or not key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env")
    model_name = "sentence-transformers/all-minilm-l6-v2"
    client = QdrantClient(url=url, api_key=key, cloud_inference=True, timeout=120)

    if not client.collection_exists(collection):
        client.create_collection(collection_name=collection,
            vectors_config={"dense": models.VectorParams(
                size=384, distance=models.Distance.COSINE, on_disk=True
            )},
            sparse_vectors_config={"bm25": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
                index=models.SparseIndexParams(on_disk=True),
            )},
        )
        print(f"Created collection {collection!r}", flush=True)
    else:
        info = client.get_collection(collection)
        print(f"Reusing collection {collection!r}; points: {info.points_count}", flush=True)

    state_path = progress_filename(records_path, collection)
    fingerprint = {"path": str(records_path.resolve()),
                   "size": records_path.stat().st_size,
                   "mtime_ns": records_path.stat().st_mtime_ns,
                   "collection": collection}
    state = {"next_line": 0, **fingerprint}
    if state_path.exists() and not restart:
        old = json.loads(state_path.read_text(encoding="utf-8"))
        if any(old.get(k) != v for k, v in fingerprint.items()):
            raise RuntimeError("Input JSONL changed since last upload. Use --restart to replay safely.")
        state = old
    elif restart:
        print("Replaying from the first line; existing IDs will be overwritten, not duplicated.", flush=True)

    if max_records is not None and max_records <= 0:
        raise ValueError("--max-records must be positive")

    batch = []
    completed = int(state["next_line"])
    start = completed
    last_line = start

    def flush(next_line: int):
        nonlocal completed, batch
        if not batch:
            return
        # same point UUID on retries -> upserts are idempotent
        client.upload_points(
            collection_name=collection,
            points=batch,
            batch_size=batch_size,
            parallel=1,
            max_retries=3,
            wait=True,
        )
        completed = next_line
        save_progress(state_path, {"next_line": completed, **fingerprint})
        print(f"Uploaded through record {completed:,}", flush=True)
        batch = []

    with records_path.open(encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx < start:
                continue
            if max_records is not None and line_idx >= max_records:
                break
            row = json.loads(line)
            last_line = line_idx + 1
            batch.append(make_point(row, model_name))
            if len(batch) >= batch_size:
                flush(line_idx + 1)
        flush(last_line)

    cloud_count = client.count(collection_name=collection, exact=True).count
    print(f"Qdrant exact count: {cloud_count:,}. Local progress: {completed:,} lines.")
    if max_records is None and completed >= 50_000 and cloud_count < 50_000:
        print("WARNING: Qdrant has fewer than 50k points; investigate before claiming compliance.")
    return cloud_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=Path("data/processed/hotpot_records.jsonl"))
    parser.add_argument("--collection", default="hotpot_evidence_v1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-records", type=int, default=None,
                        help="Initial smoke test e.g. 100; omit to upload all")
    parser.add_argument("--restart", action="store_true",
                        help="Replay all records using stable IDs; does NOT delete collection")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    upload(args.records, collection=args.collection,
           batch_size=args.batch_size, max_records=args.max_records,
           restart=args.restart)


if __name__ == "__main__":
    main()
