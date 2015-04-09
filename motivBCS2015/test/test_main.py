import numpy as np
from unittest import TestCase
from motivBCS2015 import stimulus

class TestStimulus(TestCase):
    """Testing the generation of stimulus"""
    def setUp(self):
        self.duration = 100000
        self.probability = 0.5

    def test_stimulus(self):
        """Test the generation of stimulus"""
        #Given
        duration = 10000
        probability = 0.5
        #When the stimulus generation is simple
        test_stim = stimulus(duration, self.probability)
        #Then
        self.assertAlmostEqual(np.mean(test_stim), probability, 1)

