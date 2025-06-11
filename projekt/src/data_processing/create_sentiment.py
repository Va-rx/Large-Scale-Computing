import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch
import os
import ast
from collections import defaultdict


def parse_hashtags(raw):
    if isinstance(raw, str) and raw.startswith("[{"):
        try:
            parsed = ast.literal_eval(raw)
            return [tag["text"] for tag in parsed if isinstance(tag, dict) and "text" in tag]
        except:
            return []
    return []


def get_existing_dates_for_month(file_path):
    if not os.path.exists(file_path):
        return set()
    try:
        existing_df = pd.read_parquet(file_path, columns=["date"])
        return set(existing_df["date"].dt.date.unique())
    except Exception as e:
        print(f"[WARN] Error while reading {file_path}: {e}")
        return set()


def save_monthly_partitioned(results, output_dir, prefix="tweet_sentiments"):
    monthly_data = defaultdict(list)
    for row in results:
        month_key = row["date"].strftime("%Y-%m")
        monthly_data[month_key].append(row)

    for month_key, month_rows in monthly_data.items():
        file_path = os.path.join(output_dir, f"{prefix}_{month_key}.parquet")
        existing_dates = get_existing_dates_for_month(file_path)

        filtered = [r for r in month_rows if r["date"].date() not in existing_dates]
        if not filtered:
            print(f"[SKIP] {month_key}: all days have been saved.")
            continue

        df_month = pd.DataFrame(filtered)
        table = pa.Table.from_pandas(df_month)

        if os.path.exists(file_path):
            with pq.ParquetWriter(file_path, table.schema, use_dictionary=True, compression='snappy',
                                  use_deprecated_int96_timestamps=True) as writer:
                writer.write_table(table)
        else:
            pq.write_table(table, file_path, compression='snappy', use_deprecated_int96_timestamps=True)

        print(f"[SAVE] {month_key}: saved {len(filtered)} records to {file_path}")


def prepare_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    print(f"CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA'}")
    return pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, device=device)


def analyze_sentiment(df, classifier, hypotheses, sample_size=500):
    results = []
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42).copy()
    sample_df["text_cleaned"] = sample_df["text"].str[:512].fillna("")

    outputs = classifier(sample_df["text_cleaned"].tolist(), candidate_labels=hypotheses, batch_size=8)
    if isinstance(outputs, dict):
        outputs = [outputs]

    for i, output in tqdm(enumerate(outputs), total=len(sample_df), desc="Sentyment"):
        row = sample_df.iloc[i]
        scores = dict(zip(output["labels"], output["scores"]))

        results.append({
            "tweetid": row.get("tweetid", None),
            "date": pd.to_datetime(row.get("tweetcreatedts")).date(),
            "rushate": round(scores.get(hypotheses[0], 0), 3),
            "ukrhate": round(scores.get(hypotheses[1], 0), 3),
            "location": row.get("location", None),
            "language": row.get("language", None),
            "following": row.get("following", None),
            "followers": row.get("followers", None),
            "retweetcount": row.get("retweetcount", None),
            "favoritecount": row.get("favorite_count", None),
            "hashtags": parse_hashtags(row.get("hashtags", ""))
        })

    return results


def process_batches(input_path, output_dir, batch_size=64000, min_tweets_per_day=10, days_threshold=30):
    file = pq.ParquetFile(input_path)
    classifier = prepare_model("joeddav/xlm-roberta-large-xnli")
    hypotheses = [
        "This text speaks negatively about Russia.",
        "This text speaks negatively about Ukraine."
    ]

    grouped_texts_by_day = {}
    results = []

    columns_to_read = [
        "tweetid", "tweetcreatedts", "location", "following", "followers",
        "retweetcount", "hashtags", "language", "favorite_count", "text"
    ]

    for batch in file.iter_batches(batch_size=batch_size, columns=columns_to_read):
        df = batch.to_pandas().dropna(subset=["text"])

        # Poland filter data
        # df = df[
        #     (df["language"] == "pl") |
        #     (df["location"].astype(str).str.startswith("Poland")) |
        #     (df["location"].astype(str).str.startswith("Polska"))
        # ]

        df["tweetcreatedts"] = pd.to_datetime(df["tweetcreatedts"], format="mixed", errors="coerce")
        grouped = df.groupby(df["tweetcreatedts"].dt.date)

        for date, group_df in grouped:
            grouped_texts_by_day.setdefault(date, pd.DataFrame())
            grouped_texts_by_day[date] = pd.concat([grouped_texts_by_day[date], group_df], ignore_index=True)

        print(f"[INFO] Collected data for {len(grouped_texts_by_day)} days.")

        if len(grouped_texts_by_day) >= days_threshold:
            print("[INFO] Analyzing sentiment...")

            for date, group_df in list(grouped_texts_by_day.items()):
                if len(group_df) < min_tweets_per_day:
                    print(f"[SKIP] {date}: not enough tweets ({len(group_df)}), skipping.")
                    continue

                print(f"[PROCESS] {date}: analyzing {len(group_df)} tweets...")
                daily_results = analyze_sentiment(group_df, classifier, hypotheses)
                results.extend(daily_results)

            if results:
                save_monthly_partitioned(results, output_dir)
                results = []
                grouped_texts_by_day = {}
                print("[RESET] Saved all the processed data and cleared set buff.\n")

    if results:
        save_monthly_partitioned(results, output_dir)
        print(f"[FINAL SAVE] Saved last {len(results)} records.")


def analyze_specific_tweet_ids(input_path, tweet_ids):
    file = pq.ParquetFile(input_path)
    classifier = prepare_model("joeddav/xlm-roberta-large-xnli")
    hypotheses = [
        "This text speaks negatively about Russia.",
        "This text speaks negatively about Ukraine."
    ]
    columns_to_read = [
        "tweetid", "tweetcreatedts", "location", "following", "followers",
        "retweetcount", "hashtags", "language", "favorite_count", "text"
    ]

    all_results = []

    for batch in file.iter_batches(columns=columns_to_read, batch_size=32_000):
        df = batch.to_pandas()
        df = df[df["tweetid"].isin(tweet_ids)].dropna(subset=["text"])
        if df.empty:
            continue

        df["tweetcreatedts"] = pd.to_datetime(df["tweetcreatedts"], format="mixed", errors="coerce")
        df["text_cleaned"] = df["text"].str[:512].fillna("")

        outputs = classifier(df["text_cleaned"].tolist(), candidate_labels=hypotheses, batch_size=8)

        if isinstance(outputs, dict):
            outputs = [outputs]

        for i, output in enumerate(outputs):
            row = df.iloc[i]
            scores = dict(zip(output["labels"], output["scores"]))

            all_results.append({
                "tweetid": row.get("tweetid", None),
                "date": pd.to_datetime(row.get("tweetcreatedts")).date(),
                "rushate": round(scores.get(hypotheses[0], 0), 3),
                "ukrhate": round(scores.get(hypotheses[1], 0), 3),
                "location": row.get("location", None),
                "language": row.get("language", None),
                "following": row.get("following", None),
                "followers": row.get("followers", None),
                "retweetcount": row.get("retweetcount", None),
                "favoritecount": row.get("favorite_count", None),
                "hashtags": parse_hashtags(row.get("hashtags", ""))
            })

        found_ids = set(df["tweetid"].unique())
        tweet_ids = [tid for tid in tweet_ids if tid not in found_ids]
        if not tweet_ids:
            break

    if not all_results:
        print("[WARN] No tweets found.")

    return all_results


if __name__ == "__main__":
    INPUT_PATH = "../../ukr_rus_tweets.parquet"
    OUTPUT_DIR = "../../parquets"
    process_batches(INPUT_PATH, OUTPUT_DIR)
