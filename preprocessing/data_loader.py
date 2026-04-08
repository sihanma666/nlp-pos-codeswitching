import pandas as pd

def load_ascend_parquet(path):
    df = pd.read_parquet(path)
    return df


def get_transcriptions(path, limit=None):
    df = load_ascend_parquet(path)

    texts = df["transcription"].tolist()

    if limit:
        texts = texts[:limit]

    return texts
