"""Serialization, causal output, and full directional derivative checks on CPU."""
import io
import unittest
import torch
from torch import nn
from rmfsr.conditioning import install, inspect


class ToyEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('frequencies', torch.linspace(-16, 16, 32))
        self.net = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.history, self.freq_pad = 2, 1
        self.condition = nn.Linear(128, 4)
        self.conv = nn.Conv2d(4, 4, 3)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding, self.layer = ToyEmbedding(), ToyLayer()

    def forward(self, state, time, destination):
        return self.layer(state, self.embedding(time, destination))


class ConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        torch.set_num_threads(2)
        self.model = ToyModel().double()
        self.before = dict(self.model.named_parameters())
        install(self.model, ToyLayer)

    def test_preserves_old_parameters_and_initializes_actual_scale_to_one(self):
        after = dict(self.model.named_parameters())
        self.assertTrue(all(after[key] is value for key, value in self.before.items()))
        scale = self.model.layer.scale(torch.randn(2, 128, dtype=torch.double))
        torch.testing.assert_close(scale, torch.ones_like(scale), rtol=0, atol=0)
        self.assertEqual(set(after)-set(self.before), {'layer.scale.weight', 'layer.scale.bias'})

    def test_serialization_preserves_specification_frequencies_and_outputs(self):
        data = dict(state=self.model.state_dict(), conditioning=inspect(self.model))
        stream = io.BytesIO(); torch.save(data, stream); stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = install(ToyModel().double(), ToyLayer)
        restored.load_state_dict(saved['state'], strict=True)
        self.assertEqual(inspect(restored), saved['conditioning'])
        inputs = (torch.randn(2, 4, 3, 5, dtype=torch.double), torch.tensor([.8,.7]), torch.tensor([.1,.2]))
        inputs = tuple(value.double() for value in inputs)
        torch.testing.assert_close(restored(*inputs), self.model(*inputs), rtol=0, atol=0)
        with self.assertRaises(AssertionError): install(restored, ToyLayer)

    def test_complete_state_and_time_jvp_matches_finite_difference(self):
        state = torch.randn(2, 4, 3, 5, dtype=torch.double)
        time, destination = torch.tensor([.8,.7]).double(), torch.tensor([.1,.2]).double()
        inputs = state, time, destination
        tangent = torch.randn_like(state), torch.ones_like(time), torch.zeros_like(destination)
        velocity = lambda value, t, r: (value-self.model(value,t,r))/t[:,None,None,None]
        _, actual = torch.func.jvp(velocity, inputs, tangent)
        step = 1e-6
        upper = velocity(*(value+step*direction for value,direction in zip(inputs,tangent)))
        lower = velocity(*(value-step*direction for value,direction in zip(inputs,tangent)))
        torch.testing.assert_close(actual, (upper-lower)/(2*step), rtol=1e-5, atol=1e-7)
        self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_output_causality_and_streaming_match(self):
        value = torch.randn(2, 4, 3, 9, dtype=torch.double)
        embedding = torch.randn(2, 128, dtype=torch.double)
        future = value.clone(); future[...,5:] += 100
        output = self.model.layer(value, embedding)
        torch.testing.assert_close(output[...,:5], self.model.layer(future,embedding)[...,:5])
        cache = {}
        chunks = [self.model.layer(chunk,embedding,cache) for chunk in value.split(3,-1)]
        torch.testing.assert_close(torch.cat(chunks,-1), output)


if __name__ == '__main__': unittest.main()
