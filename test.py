import matplotlib.pyplot as plt
import numpy as np

# Sample data
categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5', 'Category 6']
part_labels = ['Part 1', 'Part 2', 'Part 3']
num_categories = len(categories)
num_parts = len(part_labels)
values = np.random.randint(1, 20, size=(num_categories, num_parts))
print(values)

# Create figure and axis
fig, ax = plt.subplots()

# Plot each category as a stacked bar
bottom = np.zeros(num_categories)
colors = ['red', 'green', 'blue']  # Define colors for each part
for i in range(num_parts):
    ax.bar(categories, values[:, i], bottom=bottom, label=f'{part_labels[i]}', color=colors[i])
    bottom += values[:, i]

# Customize plot
ax.set_ylabel('Values')
ax.set_title('Bar Plot with 6 Categories and 3 Parts Each')
ax.legend()

plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

plt.show()