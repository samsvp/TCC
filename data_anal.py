#%%
import matplotlib.pyplot as plt

truth_file = "src/centroid_tracker/scripts/data.txt"
pred_file = "data.txt"


with open(truth_file) as f:
    truth_data = [eval(tupl) for tupl in f.readlines()]

with open(pred_file) as f:
    pred_data = [eval(tupl) for tupl in f.readlines()]

print(truth_data)

# %%
plt.plot([t[0] for t in truth_data[4000:]])

# %%
plt.plot([t[1] for t in truth_data])

# %%
plt.plot([p[0] for p in pred_data[212:]])

# %%
plt.plot([p[1] for p in pred_data[212:]])

# %%
p_data = pred_data[212:]

# %%
from matplotlib.pyplot import figure

figure(figsize=(8, 6))
plt.title("X Distance from the origin")
plt.ylabel("Distance (meters)")
plt.xlabel("Time")
plt.plot([-(p[0] - 0.5*p[-2]) for p in p_data], label="pred")
plt.plot([t[0] for t in truth_data[-len(p_data):]], label="Truth")
plt.legend(loc="lower right")
plt.savefig("spill_x_dist.png")
# %%
figure(figsize=(8, 6))
plt.title("Y Distance from the origin")
plt.ylabel("Distance (meters)")
plt.xlabel("Time")
plt.plot([p[1] - 0.5 * p[-1] for p in p_data], label="pred")
plt.plot([t[1] for t in truth_data[-len(p_data):]], label="truth")
plt.legend(loc="lower right")
plt.savefig("spill_y_dist.png")

# %%
figure(figsize=(8, 6))
plt.title("Relative Error (Y)")
plt.ylabel("Error (%)")
plt.xlabel("Time")
plt.plot([
    abs(truth_data[-1][1] - 
       (p_data[i][1] - 0.5 * p_data[i][-1]))  * 100 /
     truth_data[-1][1]
    for i in range(len(p_data))])
plt.legend(loc="lower right")
plt.savefig("spill_y_error.png")
# %%
figure(figsize=(8, 6))
plt.title("Relative Error (X)")
plt.ylabel("Error (%)")
plt.xlabel("Time")
plt.plot([
    abs(truth_data[-1][0] + 
        (p_data[i][0] - 0.5 * p_data[i][-2]))  * 100 /
     truth_data[-1][0]
    for i in range(len(p_data))])
plt.legend(loc="lower right")
plt.savefig("spill_x_error.png")
# %%
