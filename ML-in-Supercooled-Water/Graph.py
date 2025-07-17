import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data from CSV (replace with your file path)
file_path = 'Results.csv'
df = pd.read_csv(file_path)

# Extract data
models = df['Model']
cat0_test = df['Cat0_test']
cat1_test = df['Cat1_test']

# Plotting
fig, ax = plt.subplots(figsize=(14, 9))

# Number of models
n = len(models)

# Bar width and index
bar_width = 0.35
index = np.arange(n)

# Create bars
bar1 = ax.bar(index - bar_width/2, cat0_test, bar_width, label='cat0_test', color='forestgreen')
bar2 = ax.bar(index + bar_width/2, cat1_test, bar_width, label='cat1_test', color='darkviolet')

# Annotate values on top of bars
def annotate_bars(bars):
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom', fontsize=10)

annotate_bars(bar1)
annotate_bars(bar2)

# Labeling and customization
ax.set_xlabel('Models')
ax.set_ylabel('F1-Scores')
ax.set_title('F1-Scores for Different Models')
ax.set_xticks(index)
ax.set_xticklabels(models)
ax.legend()

plt.tight_layout()
plt.savefig('F1-Scores.png')
plt.show()
# Save the figure as an image
