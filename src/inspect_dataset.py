import pickle
import matplotlib.pyplot as plt

data_path = "data/RML2016.10a_dict.pkl"

print("Loading dataset...")

with open(data_path, "rb") as f:
    data = pickle.load(f, encoding="latin1")

print("Dataset loaded successfully.")

key = ('QPSK', 2)
samples = data[key]

print("Selected key:", key)
print("Shape:", samples.shape)

sample = samples[0]

i_signal = sample[0]
q_signal = sample[1]

plt.plot(i_signal, label="I")
plt.plot(q_signal, label="Q")

plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("QPSK Signal, SNR = 2 dB")
plt.legend()

plt.show()