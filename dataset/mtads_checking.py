import pandas as pd
import plotly.graph_objects as go

from dataset.mtads_loading import load_all_stored_datasets
from global_utils import generator_seek

def simple_plot(data: pd.DataFrame, features: int, title: str):
    fig = go.Figure()

    for feature in range(features):
        fig.add_trace(go.Scatter(x=data.index, y=data[f"value-{feature}"], mode='lines', name=feature))

    fig.update_layout(title=title)
    fig.show()

all_iter = generator_seek(load_all_stored_datasets("fsb"), drop=True)

for i in all_iter:
    g, p, trains, tests = i
    print(g)
    try:
        print(p["to_fix"])
        # plot test sequence
        simple_plot(tests, p["channels"], f"Test sequence {g}")
    except Exception as e:
        print(f"Error in test sequence {g}: {e}")
    finally:
        input("Press Enter to continue...")
