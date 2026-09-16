"""
Builds cov_profile.png (Figure 3) from the CoV-vs-x data written by
main_solver_0.py, main_solver_4.py, and main_solver_6.py once each has been
re-run with the CSV-export snippet added (see cov_csv_snippet.py).

Expects:
    results_0/cov_profile.csv
    results_4/cov_profile.csv
    results_6/cov_profile.csv
each with columns: x, CoV

Obstacle centre x-positions are taken from the geometry used in the paper
(4-obstacle case: x = 0.5, 0.9, 1.3, 1.7; 6-obstacle case: x = 0.4, 0.65,
0.9, 1.15, 1.4, 1.65 -- EDIT these two lists to match your actual Gmsh
geometry before running).
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    xs, covs = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            covs.append(float(row["CoV"]))
    return xs, covs


# EDIT these to match your actual obstacle centre x-coordinates
obstacle_x_4obs = [0.5, 0.9, 1.3, 1.7]
obstacle_x_6obs = [0.4, 0.65, 0.9, 1.15, 1.4, 1.65]

x0, c0 = load_csv("results_0/cov_profile.csv")
x4, c4 = load_csv("results_4/cov_profile.csv")
x6, c6 = load_csv("results_6/cov_profile.csv")

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(x0, c0, "o-", color="#2b6cb0", label="0 obstacles")
ax.plot(x4, c4, "s-", color="#c05621", label="4 obstacles")
ax.plot(x6, c6, "^-", color="#2f855a", label="6 obstacles")

for xo in obstacle_x_4obs:
    ax.axvline(xo, color="#c05621", linestyle="--", linewidth=0.7, alpha=0.5)
for xo in obstacle_x_6obs:
    ax.axvline(xo, color="#2f855a", linestyle="--", linewidth=0.7, alpha=0.5)

ax.axhline(0.05, color="gray", linestyle=":", linewidth=1)
ax.text(0.05, 0.06, "well-mixed threshold (CoV = 0.05)", fontsize=8, color="gray")

ax.set_xlabel("Axial position $x$")
ax.set_ylabel("Coefficient of Variation")
ax.set_title("Axial evolution of the outlet Coefficient of Variation")
ax.legend()
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig("cov_profile.png", dpi=200)
print("saved cov_profile.png")

