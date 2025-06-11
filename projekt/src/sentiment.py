import pandas as pd


def get_rus_ukr_parts(df):
    """
    Function to get Russian and Ukrainian parts of the dataframe based on sentiment analysis.
    :param df: DataFrame containing sentiment analysis results
    :return: Two DataFrames: one for Russian sentiment and one for Ukrainian sentiment
    """
    rus_neg = (df["rushate"] >= 0.5) & (df["rushate"] > df["ukrhate"])
    ukr_neg = (df["ukrhate"] >= 0.5) & (df["ukrhate"] > df["rushate"])

    df_rus_neg = df[rus_neg]
    df_ukr_neg = df[ukr_neg]

    return df_rus_neg, df_ukr_neg


def get_rus_ukr_percentages(df):
    """
    Function to calculate the percentage of negative tweets for Russian and Ukrainian sentiment.
    :param df: DataFrame containing sentiment analysis results
    :return: Dictionary with overall sentiment analysis results
    """
    df_rus_neg, df_ukr_neg = get_rus_ukr_parts(df)

    return {
        "rus_neg_perc": len(df_rus_neg) / len(df),
        "ukr_neg_perc": len(df_ukr_neg) / len(df),
    }


def get_rus_ukr_percentages_daily(df):
    """
    Function to analyze daily sentiment statistics for Russian and Ukrainian tweets.
    :param df: DataFrame containing sentiment analysis results
    :return: DataFrame with daily sentiment statistics
    """
    df["date_only"] = df["date"].dt.date

    df_rus_neg, df_ukr_neg = get_rus_ukr_parts(df)
    df_rus_neg["date_only"] = df_rus_neg["date"].dt.date
    df_ukr_neg["date_only"] = df_ukr_neg["date"].dt.date

    daily_total = df.groupby("date_only").size().rename("total")

    daily_rus_neg = df_rus_neg.groupby("date_only").size().rename("rus_neg")
    daily_ukr_neg = df_ukr_neg.groupby("date_only").size().rename("ukr_neg")

    result = pd.concat([daily_total, daily_rus_neg, daily_ukr_neg], axis=1).fillna(0)

    result["rus_neg_avg"] = result["rus_neg"] / result["total"]
    result["ukr_neg_avg"] = result["ukr_neg"] / result["total"]

    result = result.reset_index()

    return result[["date_only", "rus_neg_avg", "ukr_neg_avg"]]


def analyze_file_sentiment_stats(df):
    df_rus_neg, df_ukr_neg = get_rus_ukr_parts(df)

    return {
        "rus_neg_likes": df_rus_neg["favoritecount"].median(),
        "ukr_neg_likes": df_ukr_neg["favoritecount"].median(),
        "rus_neg_retweets": df_rus_neg["retweetcount"].median(),
        "ukr_neg_retweets": df_ukr_neg["retweetcount"].median(),
        "rus_neg_followers": df_rus_neg["followers"].median(),
        "ukr_neg_followers": df_ukr_neg["followers"].median(),
        "rus_neg_following": df_rus_neg["following"].median(),
        "ukr_neg_following": df_ukr_neg["following"].median(),
    }