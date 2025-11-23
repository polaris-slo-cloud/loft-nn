from abc import ABC, abstractmethod

import json
import pprint
import time


class Experiment(ABC):

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def visualize(self, results):
        pass

    def time(self, func):
        start = time.time()

        result = func()

        end = time.time()
        duration = end - start

        return duration, result


class ExperimentRunner:

    @staticmethod
    def run_all():
        print("Running experiments...")
        experiments = Experiment.__subclasses__()
        experimental_results = {
            experiment.__name__: list(experiment().run()) for experiment in experiments
        }

        print(f"\nExperimental results:")
        pprint.pp(experimental_results)

        with open("experimental_results.json", "w") as f:
            json.dump(experimental_results, f, indent=4)
        print("\nSaved to experimental_results.json")


class ExperimentVisualizer:

    @staticmethod
    def visualize_all():
        print("Visualizing experiments...")
        experiments = Experiment.__subclasses__()

        for experiment in experiments:
            with open("experimental_results.json", "r") as f:
                experimental_results = json.load(f)
            experiment().visualize(experimental_results.get(experiment.__name__, []))
