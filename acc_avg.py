import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('output/output_old.csv')

accuracy_average = df.groupby("Augment Layers")["Accuracy"].mean().reset_index()

colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'lightcoral', 'lightblue']

plt.figure(figsize=(10, 6))
plt.bar(accuracy_average["Augment Layers"], accuracy_average["Accuracy"])
plt.title('Average Accuracy by Augment Layers')
plt.xlabel('Augment Layers')
plt.ylabel('Average Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig('./output/avg.png')

