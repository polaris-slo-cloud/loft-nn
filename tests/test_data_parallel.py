import os

from loftnn import DataParallel
from loftnn.types import Device

from .sample_model import SimpleNN, get_sample_input, get_output_shape

class TestDataParallel:

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
    
    def test_data_parallel(self):
        self.setup_env()
        model = SimpleNN()
        batch_size = 5
        inp = get_sample_input(model, batch_size)
        expected_out = get_output_shape(model)

        config = DataParallel.Configuration.from_env()
        dp_model = DataParallel(model=model, config=config, device=Device.cpu)

        result = dp_model(inp)
        assert result.shape == (batch_size, expected_out)

        dp_model.cleanup()
        self.teardown_env()
    
    def test_is_available(self):
        self.setup_env()
        assert DataParallel.is_available() == True
        self.teardown_env()
        assert DataParallel.is_available() == False
