import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from soundscape_psd_functions import parse_xml_file

station = str(1199)

_, freqs, powers = parse_xml_file(station, channel="Z")

# ---- reshape flattened PSDs ----
n_freq = len(np.unique(freqs))
n_psd = len(freqs) // n_freq
#print out number of psds plotted
print(f"Number of PSDs plotted: {n_psd}")

freqs_reshaped = freqs.reshape(n_psd, n_freq)
powers_reshaped = powers.reshape(n_psd, n_freq)

freqs_hold = freqs_reshaped[0]

# ---- Low-noise PSD per station ----
median = np.array([
    np.median(np.sort(powers_reshaped[:, i]))
    for i in range(n_freq)
])
# ---- Low-noise PSD per station ----
low_noise_median = np.array([
    np.median(np.sort(powers_reshaped[:, i])[:10])
    for i in range(n_freq)
])

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 16))
# Plot all PSDs
for i in range(n_psd):
    ax[0].plot(
        freqs_hold,
        powers_reshaped[i],
        c='#999999',
        alpha=0.2,
        zorder=-1
    )

    ax[1].plot(
            freqs_hold,
            powers_reshaped[i],
            c='#999999',
            alpha=0.2,
            zorder=-1
    )
# Median line
ax[0].scatter(
    freqs_hold,
    median,
    c='#984ea3',
    s=6,
    label=f"median"
)

# Dots: 10 lowest per frequency
for i, f in enumerate(freqs_hold):
    p_sorted = np.sort(powers_reshaped[:, i])[:10]
    ax[1].scatter(
        np.full_like(p_sorted, f),
        p_sorted,
        s=6,
        c='#ff7f00',
        alpha=0.4,
        label=f"10 lowest powers/freq" if i == 0 else None
    )

# Median line
ax[1].scatter(
    freqs_hold,
    low_noise_median,
    c='#377eb8',
    s=6,
    label=f"low-noise median"
)

#Low-noise median line
ax[2].plot(
    freqs_hold,
    low_noise_median,
    c='#377eb8',
    lw=2,
    label=f"low-noise median"
)
# Median line
ax[2].plot(
    freqs_hold,
    median,
    c='#984ea3',
    lw=2,
    label=f"median"
)

for i, a in enumerate(ax):
    a.grid(True, which="both", alpha=0.3)
    a.set_ylabel("Power, dB", fontsize='x-large')
    a.set_xlim(5.2556e-3, 2.4355e2)
    handles, labels = a.get_legend_handles_labels()
    if a.get_legend_handles_labels()[0]:
            a.legend(handles[::-1], labels[::-1], fontsize='large') 

ax[-1].set_xscale('log')
ax[-1].set_xlabel("Frequency, Hz", fontsize='x-large')
plt.tight_layout()
plt.savefig("psd_summary.png", dpi=300, bbox_inches="tight")
plt.close()

