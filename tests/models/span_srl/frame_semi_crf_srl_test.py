# pylint: disable=no-self-use,invalid-name
import subprocess
import os

import torch
from torch.autograd import Variable

from allennlp.common.testing import ModelTestCase
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError
from allennlp.models.span_srl.frame_semi_crf_srl import FrameSemanticRoleLabeler
from allennlp.nn.util import get_lengths_from_binary_sequence_mask


class FrameSemiCrfSemanticRoleLabelerTest(ModelTestCase):
    def testMakeSpanMask(self):
        # TODO(Swabha): Check if a tensor like this is actually getting generated in framenet-reader-test.
        valid_frame_elements = Variable(torch.LongTensor(
            [[1, 2, 3, 4, 9], [7, 7, -1, -1, -1]]).cuda())

        num_classes = 10
        batch_size = 2
        tag_mask = FrameSemanticRoleLabeler.get_tag_mask(
            num_classes, valid_frame_elements, batch_size)
        expected_tag_mask = Variable(torch.FloatTensor([[0, 1, 1, 1, 1, 0, 0, 0, 0, 1], [
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0]]).cuda())

        assert torch.eq(tag_mask, expected_tag_mask).data.all()
