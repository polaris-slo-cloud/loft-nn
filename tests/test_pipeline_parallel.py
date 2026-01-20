from loftnn import PipelineConfiguration, PipelineParallel, ProcessConfiguration

from .sample_model import SimpleNN, get_sample_input, get_total_parameters
from .test_parallel import TestParallel


class TestPipelineParallel(TestParallel):

    def test_pipeline_parallel(self):
        self.setup_env()
        model = SimpleNN()
        batch_size = 15
        num_microbatches = 3
        inp = get_sample_input(model, batch_size)

        process_config = ProcessConfiguration.from_env()
        pp_model = PipelineParallel(
            model=model,
            process_config=process_config,
            pipeline_config=PipelineConfiguration(
                split_points=[1, 3],
                num_microbatches=num_microbatches,
                microbatch_sample=inp.chunk(num_microbatches)[0],
                loss_fn=None,
            ),
        )

        assert get_total_parameters(pp_model) < get_total_parameters(model)
        assert pp_model.training == True

        pp_model.cleanup()
        self.teardown_env()
