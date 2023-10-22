def handle_null_values(data):
    """
    Handle null values in a DataFrame by filling them with the median of each column.

    Parameters:
        data (pd.DataFrame): Input DataFrame with potentially missing values.

    Returns:
        pd.DataFrame: DataFrame with null values replaced by column medians.
    """
    data = data.fillna(data.median())
    return data
