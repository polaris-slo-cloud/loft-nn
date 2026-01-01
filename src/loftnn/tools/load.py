import pickle


def load_exact_planner_state():
    with open("exact_planner_state.pickle", "rb") as handle:
        state = pickle.load(handle)
    return state


def load_exact_profiler_caches():
    with open("exact_profiler_caches.pickle", "rb") as handle:
        caches = pickle.load(handle)
    return caches
