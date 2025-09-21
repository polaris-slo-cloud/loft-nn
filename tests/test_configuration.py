import loftnn

from .test_parallel import TestParallel


class TestConfiguration(TestParallel):

    def test_is_available(self):
        self.setup_env()
        assert loftnn.is_available() == True
        self.teardown_env()
        assert loftnn.is_available() == False
