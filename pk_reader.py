import pickle

# Path to .pk file, change it to your path
file_path = r"D:\git_projects\mpt_tracking\mpt_tracking\randomnoise.pk"
def load_measurements(file_path):
    with open(file_path, 'rb') as file:
        measurements = pickle.load(file)
    return measurements

measurements = load_measurements(file_path)

print("Loaded Measurements:")
for measurement in measurements:
    print(measurement)


