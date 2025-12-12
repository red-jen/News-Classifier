class EDA:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def visualize_distribution(self, column):
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.dataframe, x=column)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()

    def generate_summary_statistics(self):
        return self.dataframe.describe(include='all')