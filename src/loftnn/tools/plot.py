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

    plt.savefig("parameters_by_layer.png")
    print("saved parameters_by_layer.png")

    plt.show()


def plot_training_times_by_epoch(times_by_label, xlabel="Epoch"):
    for label, times in times_by_label.items():
        epochs = range(1, len(times) + 1)
        plt.plot(epochs, times, label=label, linestyle="-", marker="o")

    plt.title("Training Times")
    plt.xlabel(xlabel)
    plt.ylabel("Time (s)")

    max_epochs = max(len(times) for times in times_by_label.values())
    plt.xticks(range(1, max_epochs + 1))
    plt.ylim(0, 100)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig("training_times_by_epoch.png")
    print("saved training_times_by_epoch.png")

    plt.show()


def plot_profiling_and_planning_times(labels, profiling_times, planning_times):
    fig, ax = plt.subplots()

    ax.bar(labels, profiling_times, label="Profiling")
    ax.bar(labels, planning_times, bottom=profiling_times, label="Planning")

    ax.set_ylabel("Time (s)")
    ax.set_title("Profiling and planning overhead")
    ax.legend()

    plt.savefig("profiling_and_planning_times.png")
    print("saved profiling_and_planning_times.png")

    plt.show()


def plot_planning_time_scaling(labels, worker_counts, planning_times):
    plt.plot(
        worker_counts, planning_times[0], label=labels[0], marker="o", linestyle="-"
    )
    plt.plot(
        worker_counts, planning_times[1], label=labels[1], marker="s", linestyle="-"
    )

    plt.xlabel("Number of workers")
    plt.ylabel("Time (s)")
    plt.title("Planning Time Scaling")

    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig("planning_time_scaling.png")
    print("saved planning_time_scaling.png")

    plt.show()
