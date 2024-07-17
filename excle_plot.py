import pandas as pd
import matplotlib.pyplot as plt

# Load data from Excel file
excel_file = "building_comparison_results.xlsx"  # Change this to your file path
sheet_name = "Sheet3"  # Change the sheet name if it's different
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# Columns to plot
columns_to_plot = [
                   'dx_determined_absolut', 

                   'dy_determined_absolut']
labels = [
                   'Jaccard Index ', 

                   'Advanced Jaccard Index']
# Plot histograms for each column
plt.figure(figsize=(12, 8))
i = 0
for col in columns_to_plot:
    plt.hist(df[col], bins=100, alpha=0.5, histtype='step', label=labels[i])
    i = i+1

plt.xlabel('Jaccard Index')
plt.ylabel('Frequency')
plt.title('Distribution of (Advanced) Jaccard Indices in the sampleset')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# Calculate and print the number of entries below 0.8 for each column
for col, label in zip(columns_to_plot, labels):
    count_below_0_8 = (df[col] < 0.8).sum()
    print(f'Number of entries in {label} below 0.8: {count_below_0_8}')