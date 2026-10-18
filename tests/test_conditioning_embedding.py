"""Positional encoding convention and exact time derivatives, no hidden scale."""
import unittest
import torch
from rmfsr.conditioning_embedding import PositionalTimeEmbedding
from test_conditioning import ToyEmbedding


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.embedding = PositionalTimeEmbedding(ToyEmbedding().double())

    def test_geometric_angular_frequencies_and_identity_time_encoding(self):
        frequencies = 10000.**(-torch.arange(32,dtype=torch.double)/31)
        torch.testing.assert_close(self.embedding.frequencies,frequencies,rtol=0,atol=0)
        time, destination = torch.tensor([.7]).double(), torch.tensor([.2]).double()
        phases = torch.stack((time,destination),-1)[...,None]*frequencies
        expected = torch.cat((phases.sin(),phases.cos()),-1).flatten(1)
        torch.testing.assert_close(self.embedding.encode(time,destination),expected,rtol=0,atol=0)
        self.assertEqual(float(frequencies.max()),1.)

    def test_time_derivative_retains_every_frequency_without_multiplier(self):
        time, destination = torch.tensor([.7]).double(), torch.tensor([.2]).double()
        _,actual = torch.func.jvp(self.embedding.encode,(time,destination),
                                  (torch.ones_like(time),torch.zeros_like(destination)))
        phase = time[:,None]*self.embedding.frequencies
        expected_t = torch.cat((phase.cos()*self.embedding.frequencies,
                                -phase.sin()*self.embedding.frequencies),-1)
        torch.testing.assert_close(actual[:,:64],expected_t)
        torch.testing.assert_close(actual[:,64:],torch.zeros_like(expected_t))

    def test_position_basis_preserves_existing_time_mlp(self):
        original = ToyEmbedding().double()
        replacement = PositionalTimeEmbedding(original)
        self.assertIs(replacement.net,original.net)
        time, destination = torch.tensor([.7]).double(), torch.tensor([.2]).double()
        torch.testing.assert_close(replacement(time,destination),
                                   original.net(replacement.encode(time,destination)))


if __name__ == '__main__': unittest.main()
