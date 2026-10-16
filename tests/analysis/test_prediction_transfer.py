from types import SimpleNamespace

import numpy as np
import pytest

from visioncortex.detection_inference import prediction_arrays


def test_batch_transfer_preserves_boxes_confidences_classes_and_empty_frames(monkeypatch):
    torch = pytest.importorskip('torch')
    values = [torch.tensor([[-.1, .2, 1.1, .9, .375, 2], [.1, .3, .4, .7, .8125, 20]]),
              torch.empty((0, 6)), torch.tensor([[.2, .1, .8, .9, .25, 0]])]
    predictions = [SimpleNamespace(boxes=SimpleNamespace(xyxyn=v[:, :4], conf=v[:, 4], cls=v[:, 5]))
                   for v in values]
    predictions.insert(2, SimpleNamespace(boxes=None))
    expected = [(np.clip(v[:, :4].numpy(), 0, 1), v[:, 4].numpy(), v[:, 5].numpy().astype(int))
                for v in values]
    expected.insert(2, (np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)))
    original_cpu = torch.Tensor.cpu
    transfers = []

    def host_transfer(self, *args, **kwargs):
        transfers.append(self.shape)
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'cpu', host_transfer)
    actual = prediction_arrays(predictions)
    assert len(actual) == len(expected)
    for got, wanted in zip(actual, expected, strict=True):
        for array, reference in zip(got, wanted, strict=True):
            np.testing.assert_array_equal(array, reference)
    assert transfers == [(3, 6)]


def test_optional_array_ports_remain_compatible():
    class ArrayPort:
        def __init__(self, data):
            self.data = np.asarray(data)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.data

    prediction = SimpleNamespace(boxes=SimpleNamespace(
        xyxyn=ArrayPort([[-1, .2, .9, 2]]), conf=ArrayPort([.75]), cls=ArrayPort([3])))
    actual = prediction_arrays([prediction, SimpleNamespace(boxes=None)])
    np.testing.assert_array_equal(actual[0][0], [[0, .2, .9, 1]])
    np.testing.assert_array_equal(actual[0][1], [.75])
    np.testing.assert_array_equal(actual[0][2], [3])
    assert actual[1][0].shape == (0, 4)
