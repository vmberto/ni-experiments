import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


df = pd.read_csv(f"{os.getcwd()}/output/output.csv")

plt.figure(figsize=(12, 6))

sns.set_theme(style="whitegrid")

g = sns.catplot(
    data=df, kind="bar",x="Augment Layers", y="Accuracy", hue="Name",
)
g.despine(left=True)
g.set_axis_labels("", "acc")
g.legend.set_title("")

plt.xticks(rotation=45, ha='right')

plt.show()
