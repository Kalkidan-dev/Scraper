def add_production_budget_scale(df, features):
    """
    Add a new feature for Production Budget Scale to the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        features (list): The list of feature column names.

    Returns:
        pd.DataFrame, list: Updated DataFrame and features list.
    """
    def classify_budget(budget):
        """Classify movies into budget categories based on production cost."""
        if budget < 10:
            return 'Low'
        elif 10 <= budget < 50:
            return 'Medium'
        else:
            return 'High'

    # Apply the function to create a new column for budget scale
    df['Budget_Scale'] = df['Production_Budget_Millions'].apply(classify_budget)

    # One-hot encode the 'Budget_Scale' feature to treat it as categorical data
   
