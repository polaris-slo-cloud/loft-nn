import matplotlib.pyplot as plt

from loftnn.model_utils import model_to_layers


def plot_model(model, name, X, Y=None):
    from torchview import draw_graph

    draw_graph(
        model,
        input_size=X.shape if Y is None else [X.shape, Y.shape],
        dtypes=[X.dtype] if Y is None else [X.dtype, Y.dtype],
        depth=1,
        expand_nested=False,
        graph_name=f"{name}_architecture",
        save_graph=True,
        filename=f"model_{name}",
    )
    print(f"saved model_{name}.png")


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

    plt.close()


def plot_planning_time_scaling(labels, worker_counts, planning_times):
    plt.rcParams.update({"font.size": 30})
    plt.figure(figsize=(10, 6))

    plt.plot(
        worker_counts,
        planning_times[0],
        label=labels[0],
        marker="o",
        linestyle="-",
        linewidth=5,
        markersize=12,
    )
    plt.plot(
        worker_counts,
        planning_times[1],
        label=labels[1],
        marker="s",
        linestyle="-",
        linewidth=5,
        markersize=12,
    )

    plt.xlabel("Number of workers")
    plt.ylabel("Time (s)")

    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig("planning_time_scaling.png")
    print("saved planning_time_scaling.png")

    plt.close()


def plot_training_times_by_epoch(path, times_by_label, xlabel="Epoch", ylim=100):
    plt.rcParams.update({"font.size": 30})
    plt.figure(figsize=(10, 6))

    for label, times in times_by_label.items():
        epochs = range(1, len(times) + 1)
        plt.plot(
            epochs,
            times,
            label=label,
            linestyle="-",
            marker="o",
            linewidth=5,
            markersize=12,
        )

    plt.xlabel(xlabel)
    plt.ylabel("Time (s)")

    max_epochs = max(len(times) for times in times_by_label.values())
    plt.xticks(range(1, max_epochs + 1))
    plt.ylim(0, ylim)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(path)
    print(f"saved {path}")

    plt.close()


def plot_training_losses_by_epoch(path, losses_by_label, xlabel="Epoch", ylim=1):
    plt.rcParams.update({"font.size": 30})
    plt.figure(figsize=(10, 6))

    for label, losses in losses_by_label.items():
        epochs = range(1, len(losses) + 1)
        plt.plot(
            epochs,
            losses,
            label=label,
            linestyle="-",
            marker="o",
            linewidth=5,
            markersize=12,
        )

    plt.xlabel(xlabel)
    plt.ylabel("Loss")

    max_epochs = max(len(losses) for losses in losses_by_label.values())
    plt.xticks(range(1, max_epochs + 1))
    plt.ylim(0, ylim)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    plt.savefig(path)
    print(f"saved {path}")

    plt.close()


def plot_profiling_and_planning_times(
    path, labels, profiling_times, planning_times, ylim=100
):
    plt.rcParams.update({"font.size": 30})
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(labels, profiling_times, label="Profiling")

    planning_times_gte1 = [max(t, 1) for t in planning_times]
    planning_bar = ax.bar(
        labels, planning_times_gte1, bottom=profiling_times, label="Planning"
    )

    total_times_rounded = [
        round(profiling_time + planning_time, 2)
        for profiling_time, planning_time in zip(profiling_times, planning_times)
    ]
    ax.bar_label(planning_bar, labels=total_times_rounded, padding=3)

    ax.set_ylim(0, ylim)
    ax.set_ylabel("Time (s)")
    ax.legend()
    plt.tight_layout()

    plt.savefig(path)
    print(f"saved {path}")

    plt.close()


def plot_model_size_limits(path, labels, model_size_limits, ylim=1000):
    plt.rcParams.update({"font.size": 30})
    fig, ax = plt.subplots(figsize=(10, 6))

    bar = ax.bar(
        labels,
        model_size_limits,
        label="Model size limit",
        color=["skyblue", "salmon"],
    )

    ax.bar_label(bar, labels=model_size_limits, padding=3)

    ax.set_ylim(0, ylim)
    ax.set_ylabel("Parameters (M)")
    plt.tight_layout()

    plt.savefig(path)
    print(f"saved {path}")

    plt.close()
