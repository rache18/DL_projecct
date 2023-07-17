
# C:\Users\rache\Desktop\School\Master\Year 1\Semester B\Deep Learning\DL_projecct\csv models 150 epochs\--data_augmentation --model resnet18 --shape triangle --length 16 --epochs 150\tryploting.csv

import csv
import matplotlib.pyplot as plt

# Get the CSV file path from user input
csv_file = input("Enter the CSV file path: ")

# Initialize lists to store the data
x_line = []
function1 = []
function2 = []

# Read the CSV file and extract the data
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row
    for i, row in enumerate(reader):
        x_line.append(float(row[1]))      # Add values from the second column to x_line
        function1.append(float(row[2]))  # Add values from the third column to function1
        function2.append(float(row[3]))  # Add values from the fourth column to function2
        if i == 2:
            function1_title = row[0]  # Assign the title of function1
        elif i == 3:
            function2_title = row[0]  # Assign the title of function2

# Get the title of the plot
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    title_row = list(reader)[4]
    plot_title = title_row[0]

# Plot the data
plt.plot(x_line, function1, label=function1_title)  # Use function1_title as the label for function1
plt.plot(x_line, function2, label=function2_title)  # Use function2_title as the label for function2

# Set the plot title
plt.title(plot_title)

# Set the axis labels
plt.xlabel('X line')
plt.ylabel('Function Values')

# Display the legend
plt.legend()

# Show the plot
plt.show()
