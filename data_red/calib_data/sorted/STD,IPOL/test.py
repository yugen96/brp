import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

planets = sns.load_dataset("planets")

sns.violinplot("orbital_period", "method", data=planets.query("orbital_period < 1000"))

# Get the ax object to use later.
ax = sns.boxplot(x="day", y="total_bill", hue="smoker",
data=tips,palette="Set2",fliersize=0)

# Get the handles and labels. For this example it'll be 2 tuples
# of length 4 each.
handles, labels = ax.get_legend_handles_labels()

# When creating the legend, only use the first two elements
# to effectively remove the last two.
l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)




