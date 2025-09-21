import os


class TestParallel:

    def setup_env(self):
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "1234"

    def teardown_env(self):
        os.environ["RANK"] = "-1"
        os.environ["LOCAL_RANK"] = "-1"
        os.environ["WORLD_SIZE"] = "-1"
        os.environ["MASTER_ADDR"] = "-1"
        os.environ["MASTER_PORT"] = "-1"
