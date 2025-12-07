from abc import ABC, abstractmethod

import logging
import json
import pprint
import time

logger = logging.getLogger(__name__)


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
        logger.info("Running experiments...")
        experiments = Experiment.__subclasses__()
        experimental_results = {
            experiment.__name__: list(experiment().run()) for experiment in experiments
        }

        logger.info(f"\nExperimental results:")
        logger.info(pprint.pformat(experimental_results))

        with open("experimental_results.json", "w") as f:
            json.dump(experimental_results, f, indent=4)
        logger.info("\nSaved to experimental_results.json")


class ExperimentVisualizer:

    @staticmethod
    def visualize_all():
        logger.info("Visualizing experiments...")
        experiments = Experiment.__subclasses__()

        for experiment in experiments:
            with open("experimental_results.json", "r") as f:
                experimental_results = json.load(f)
            experiment().visualize(experimental_results.get(experiment.__name__, []))
