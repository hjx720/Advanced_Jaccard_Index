import sqlite3
import os

# Define the folder path
folder_path = r".\example_data"

# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('building_data.db')
cursor = conn.cursor()

# Drop the table if it exists
cursor.execute('''DROP TABLE IF EXISTS building_data''')

# Create a table to store the data
cursor.execute('''CREATE TABLE building_data
                (id INTEGER PRIMARY KEY,
                base_name TEXT,
                manipulation_type TEXT,
                manipulation_intensity TEXT,
                subversion TEXT,
                randomized_level INTEGER,
                file_path TEXT)''')

# Iterate through each file in the folder
for file_name in os.listdir(folder_path):
    # Split the file name into parts based on underscores
    parts = file_name[0:-5].split('_')

    try:
        # Extract information from the filename
        base_name = parts[0]+parts[1]
    except:
        print(f"unexpected filename structure: {file_name}")
    try:
        manipulation_type = parts[2]  # Fixed manipulation type
    except:
        manipulation_type = "None"
        print(f"no manipulation: {file_name}")
    try:
        manipulation_intensity = parts[3]  # Extract intensity level
    except:
        manipulation_intensity = "None"
        print(f"no manipulation intensity: {file_name}")
    try:
        randomized = parts[4]
    except:
        randomized = "None"
        print(f"not randomized: {file_name}")
    try:
        randomized_level = int(parts[5])
    except:
        randomized_level = "None"
        print(f"not randomized: {file_name}")
    file_path = os.path.join(folder_path, file_name)

    # Insert the data into the database
    cursor.execute('''INSERT INTO building_data
                    (base_name, manipulation_type, manipulation_intensity, subversion, randomized_level, file_path)
                    VALUES (?, ?, ?, ?, ?, ?)''',
                    (base_name, manipulation_type, manipulation_intensity, randomized, randomized_level, file_path))

# Commit changes and close the connection
conn.commit()
conn.close()

print("Data has been successfully inserted into the database.")
