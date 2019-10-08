"""Jigsaw Toxic Comment Classification Problem."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from zipfile import ZipFile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem, text_problems
from tensor2tensor.utils import registry

import tensorflow as tf
from re import compile

from multi_label import Text2MultiLabelProblem

@registry.register_problem
class JigsawToxicCommentClassification(Text2MultiLabelProblem):
  """Jigsaw Toxic Comment Classification."""
  URL = "https://drive.google.com/uc?export=download&id=1pCRlILaqd7IpGaBwa5euKd3Vbq4HQVe-"
  RE_TRAIN = compile(r'^"([\da-z]+)","("")?(.+?)"("")?,([01]),([01]),([01]),([01]),([01]),([01])\s*$')
  RE_TEST = compile(r'^"([\da-z]+)","("")?(.+?)"("")?\s*$')
  RE_TEST_LABEL = compile(r'^([\da-z]+),([01]),([01]),([01]),([01]),([01]),([01])\s*$')

  @property
  def is_generate_per_split(self):
    return True

  @property
  def dataset_splits(self):
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 10,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  @property
  def approx_vocab_size(self):
    return 2**13

  @property
  def num_classes(self):
    return 6

  def class_labels(self, data_dir):
    del data_dir
    return ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

  def doc_generator(self, jigsaw_dir, dataset, include_label=False):
    if dataset == "train":
        path = os.path.join(jigsaw_dir, "train.csv")
        skip_header = True

        with tf.gfile.Open(path) as jigsaw_f:
            doc = jigsaw_f.read()
            for match in self.RE_TRAIN.finditer(doc):
                if skip_header:
                    skip_header = False
                    continue
                text = match.group(3)
                if include_label:
                    yield text, [i for i in range(5) if match.group(5 + i) == '1']
                else:
                    yield text
    else:
        test_label_path = os.path.join(jigsaw_dir, "test_labels.csv")
        test_labels = {}
        with tf.gfile.Open(test_label_path) as jigsaw_label_f:
            doc = jigsaw_label_f.read()
            for match in self.RE_TEST_LABEL.finditer(doc):
                comment_id = match.group(1)
                test_labels[comment_id] = [i*match.group(2 + i) for i in range(6) == '1']

        test_path = os.path.join(jigsaw_dir, "test.csv")
        with tf.gfile.Open(test_path) as jigsaw_f:
            doc = jigsaw_f.read()
            for match in self.RE_TEST.finditer(doc):
                if skip_header:
                    skip_header = False
                    continue
                comment_id = match.group(1)
                if comment_id in test_labels:
                    text = match.group(3)
                    if include_label:
                        yield text, test_labels[comment_id]
                    else:
                        yield text


  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate examples."""
    # Download and extract
    download_path = generator_utils.maybe_download_from_drive(tmp_dir, 'jigsaw-toxic-comment-classification-challenge.zip',
                                                   self.URL)
    jigsaw_dir = os.path.join(tmp_dir, "jigsaw")
    if not tf.gfile.Exists(jigsaw_dir):
      with ZipFile(download_path) as zipfile:
        zipfile.extractall(jigsaw_dir)

    # Generate samples
    train = dataset_split == problem.DatasetSplit.TRAIN
    dataset = "train" if train else "test"
    for doc, labels in self.doc_generator(jigsaw_dir, dataset, include_label=True):
      yield {
          "inputs": doc,
          "labels": [int(label) for label in labels],
      }

@registry.register_problem
class JigsawToxicCommentClassificationCharacters(JigsawToxicCommentClassification):
  """Jigsaw Toxic Comment Classification, character level."""

  @property
  def vocab_type(self):
    return text_problems.VocabType.CHARACTER

  def global_task_id(self):
    return problem.TaskID.EN_CHR_SENT