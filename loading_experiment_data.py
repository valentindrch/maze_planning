import csv

def load_maze_data(file_path):
    data = {}
    with open(file_path, mode='r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            id = row['id']
            if id not in data:
                data[id] = []
            data[id].append({key: value for key, value in row.items() if key not in ['ID', 'trial']})
    return data

file_path = 'maze_data.csv'

# return the distance of a the first participants first trial
maze_data = load_maze_data(file_path)

