import unittest

from flightrecorder.aicpu_sampler import parse_usages


class AicpuSamplerTests(unittest.TestCase):
    def test_parse_a2_usages(self):
        text = """Aicore Usage Rate(%) : 12
	Aivector Usage Rate(%) : 17
	Aicpu Usage Rate(%) : 63
	Ctrlcpu Usage Rate(%) : 4
"""
        self.assertEqual(parse_usages(text), dict(aicore_pct=12, aivector_pct=17,
                                                  aicpu_pct=63, ctrlcpu_pct=4))


if __name__ == "__main__":
    unittest.main()
