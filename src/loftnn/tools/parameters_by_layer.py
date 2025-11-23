import matplotlib.pyplot as plt

from loftnn.model_utils import model_to_layers


def plot_parameters_by_layer(model):
    layers = model_to_layers(model, include_names=True)

    xs = []
    heights = []
    for name, layer in layers:
        xs.append(name)
        heights.append(sum(p.numel() for p in layer.parameters()))

    fig, ax = plt.subplots()
    ax.bar(xs, heights)
    ax.set_ylabel("# Parameters")
    ax.set_title("Parameters by Layer")
    plt.xticks(rotation=90)
    plt.show()
