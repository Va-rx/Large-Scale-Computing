import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq


dtypes = {
    "Unnamed: 0": "Int64",
    "userid": "Int64",
    "username": "string",
    "acctdesc": "string",
    "location": "string",
    "following": "Int64",
    "followers": "Int64",
    "totaltweets": "Int64",
    "usercreatedts": "string",
    "tweetid": "Int64",
    "tweetcreatedts": "string",
    "retweetcount": "Int64",
    "text": "string",
    "hashtags": "string",
    "language": "string",
    "favorite_count": "Int64",
    "is_retweet": "boolean",
    "original_tweet_id": "Int64",
    "original_tweet_userid": "Int64",
    "original_tweet_username": "string",
    "in_reply_to_status_id": "Int64",
    "in_reply_to_user_id": "Int64",
    "in_reply_to_screen_name": "string",
    "is_quote_status": "boolean",
    "quoted_status_id": "Int64",
    "quoted_status_userid": "Int64",
    "quoted_status_username": "string",
    "extractedts": "string"
}
desired_column_order = list(dtypes.keys())


def enforce_dtypes(df, dtypes):
    for col, dtype in dtypes.items():
        if col not in df.columns:
            df[col] = pd.NA
        try:
            if dtype == "string":
                df[col] = df[col].astype(pd.StringDtype())
            elif dtype == "Int64":
                df[col] = df[col].astype("Int64")
            elif dtype == "boolean":
                df[col] = df[col].astype("boolean")
        except Exception as e:
            print(f"[Error] Converting column '{col}' failed: {e}")
            df[col] = pd.NA
            df[col] = df[col].astype(pd.StringDtype() if dtype == "string" else dtype)
    return df


def csv_to_parquet_chunked(folder_path, output_file, chunk_size, dtypes):
    first_file = True
    schema = None
    writer = None

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                        chunk = chunk.drop(columns=['coordinates'], errors='ignore')

                        for col in desired_column_order:
                            if col not in chunk.columns:
                                chunk[col] = pd.NA

                        chunk = chunk[desired_column_order]

                        chunk = enforce_dtypes(chunk, dtypes)

                        table = pa.Table.from_pandas(chunk)

                        if first_file:
                            schema = table.schema
                            writer = pq.ParquetWriter(output_file, schema)
                            first_file = False

                        if table.schema != schema:
                            print(f"[Error]: File {file_path} skipped because of different schema.")
                            continue

                        writer.write_table(table)
                        print(f"[Processed]: Chunk from file {file_path} ({len(chunk)} rows)")

                except Exception as e:
                    print(f"[Error]: While loading file {file_path}: {str(e)}")

    if writer:
        writer.close()
        print(f"\n[Success]: All data saved to file {output_file}")
    else:
        print("[Error]: No data to save.")


folder_path = "../../ukr_rus_tweets/"
output_file = "../../ukr_rus_tweets.parquet"
chunk_size = 100000
csv_to_parquet_chunked(folder_path, output_file, chunk_size, dtypes)
