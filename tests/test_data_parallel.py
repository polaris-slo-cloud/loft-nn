from loftnn import DataParallel, ProcessConfiguration
from loftnn.types import Device

from .sample_model import SimpleNN, get_sample_input, get_output_shape
from .test_parallel import TestParallel


class TestDataParallel(TestParallel):

    def test_data_parallel(self):
        self.setup_env()
        model = SimpleNN()
        batch_size = 5
        inp = get_sample_input(model, batch_size)
        expected_out = get_output_shape(model)

        process_config = ProcessConfiguration.from_env()
        dp_model = DataParallel(
            model=model, process_config=process_config, device=Device.cpu
        )

        result = dp_model(inp)
        assert result.shape == (batch_size, expected_out)

        dp_model.cleanup()
        self.teardown_env()
