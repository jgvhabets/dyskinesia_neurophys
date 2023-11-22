import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# read X_cross_val_data.pickle
with open(
    os.path.join("lfpecog_decode_cebra", "X_cross_val_data_ECOG_AND_STN.pickle"), "rb"
) as f:
    X_cross_val_data = pickle.load(f)
sub_ids = np.unique(X_cross_val_data["sub_ids"])

plt.figure(figsize=(10, 7))
for sub_idx_plt, sub in enumerate(np.unique(X_cross_val_data["sub_ids"])):
    plt.subplot(3, 5, sub_idx_plt + 1)
    idx_sub = np.where(X_cross_val_data["sub_ids"] == sub)[0]
    plt.imshow(
        stats.zscore(X_cross_val_data["X_all"][idx_sub].T, axis=1),
        aspect="auto",
        cmap="viridis",
    )
    if sub_idx_plt in (0, 5, 10):
        plt.yticks(
            np.arange(len(X_cross_val_data["ft_names"])), X_cross_val_data["ft_names"]
        )
    plt.xticks(
        np.arange(len(X_cross_val_data["ft_times_all"][idx_sub]))[::50],
        np.rint(X_cross_val_data["ft_times_all"][idx_sub][::50]).astype(int),
    )
    plt.clim(-3, 3)
    plt.plot(-X_cross_val_data["y_all_scale"][idx_sub] + 47, c="white")
    plt.title(sub)
    plt.colorbar()
plt.tight_layout()
plt.savefig(
    "features_vs_time.pdf",
)


# with PLOTLY
y_tick_labels = X_cross_val_data["ft_names"]
fig = make_subplots(
    rows=6, cols=4, subplot_titles=[f"Subject {sub}" for sub in sub_ids]
)

for i, sub in enumerate(sub_ids, start=0):
    idx_sub = np.where(X_cross_val_data["sub_ids"] == sub)[0]
    image = X_cross_val_data["X_all"][idx_sub]
    time = X_cross_val_data["ft_times_all"][idx_sub]

    # Add a heatmap for this subject to the i-th subplot
    row = i // 4 + 1
    col = i % 4 + 1
    fig.add_trace(
        go.Heatmap(
            z=stats.zscore(image.T, axis=0),
            x=time,
            y=np.arange(len(y_tick_labels)),
            zmin=-3,
            zmax=3,
            colorscale="Viridis",
            # showscale=True,
            name=f"Subject {sub}",  # Add a name for this trace
        ),
        row=row,
        col=col,
    )
    fig.update_yaxes(
        title_text="Features",
        tickvals=np.arange(len(y_tick_labels)),
        ticktext=y_tick_labels,
        row=row,
        col=col,
    )
    fig.update_xaxes(title_text="Time [min]", row=row, col=col)

fig.update_layout(
    # xaxis_title="Time",
    # yaxis_title="Y",
    autosize=False,
    width=4000,
    height=4000,  # *len(sub_ids),  # Adjust the height based on the number of subjects
    # margin=dict(
    #    l=1,
    #    r=50,
    #    b=100,
    #    t=100,
    #    pad=4
    # ),
)

fig.show()
fig.write_image("Featurers_over_time_plotly.pdf")
