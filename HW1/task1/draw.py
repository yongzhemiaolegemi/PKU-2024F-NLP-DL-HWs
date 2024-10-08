import matplotlib.pyplot as plt

# Data provided by the user
data = {
    "test_1_1e1": 0.0003596305847167969,
    "test_1_1e2": 0.0,
    "test_1_1e3": 0.000652313232421875,
    "test_1_1e4": 0.0006623268127441406,
    "test_1_1e5": 0.0018207073211669923,
    "test_1_1e6": 0.022583484649658203,
    "test_1_1e7": 0.22885477542877197,
    "test_2_1e1": 0.0004015684127807617,
    "test_2_1e2": 0.000355219841003418,
    "test_2_1e3": 0.00033020973205566406,
    "test_2_1e4": 0.00015707015991210936,
    "test_2_1e5": 0.0009403228759765625,
    "test_2_1e6": 0.008353710174560547,
    "test_2_1e7": 0.08222310543060303
}
# get data 2
# Test test_1e1 average running time = 0.00020091533660888673
# Test test_1e2 average running time = 0.0004008054733276367
# Test test_1e3 average running time = 0.0002504587173461914
# Test test_1e4 average running time = 0.0012414216995239257
# Test test_1e5 average running time = 0.010425591468811035
# Test test_1e6 average running time = 0.09935836791992188
# Test test_1e7 average running time = 0.9823718786239624
data_2 = {
    "test_1e1": 0.00020091533660888673,
    "test_1e2": 0.0004008054733276367,
    "test_1e3": 0.0002504587173461914,
    "test_1e4": 0.0012414216995239257,
    "test_1e5": 0.010425591468811035,
    "test_1e6": 0.09935836791992188,
    "test_1e7": 0.9823718786239624
}
# Extracting the test numbers and their corresponding running times
def get_number(name):
    if "1e1" in name:
        return 10
    elif "1e2" in name:
        return 100
    elif "1e3" in name:
        return 1000
    elif "1e4" in name:
        return 10000
    elif "1e5" in name:
        return 100000
    elif "1e6" in name:
        return 1000000
    elif "1e7" in name:
        return 10000000
test_numbers = [get_number(name) for name in data.keys()]

running_times = list(data_2.values())
running_times_1 = running_times[:7]
running_times_2 = running_times[7:]
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(test_numbers[:7], running_times_1, marker='o', linestyle='-', color='b')
# plt.plot(test_numbers[:7], running_times_2, marker='o', linestyle='-', color='r')
plt.xscale('log')
plt.xlabel('Test Number (Log Scale)')
plt.ylabel('Average Running Time (seconds)')
plt.title('Average Running Time vs Test Number')
plt.grid(True)
plt.show()
