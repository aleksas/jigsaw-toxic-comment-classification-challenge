from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import problem, text_problems, text_encoder
from tensor2tensor.layers import modalities
import tensorflow as tf

class Text2MultiLabelProblem(text_problems.Text2TextProblem):
  """Base class for text multi-labeling problems."""

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    """Generate samples of text and label pairs.
    Each yielded dict will be a single example. The inputs should be raw text.
    The label should be an int array with each element in [0, self.num_classes).
    Args:
      data_dir: final data directory. Typically only used in this method to copy
        over user-supplied vocab files (for example, if vocab_type ==
        VocabType.TOKEN).
      tmp_dir: temporary directory that you can use for downloading and scratch.
      dataset_split: problem.DatasetSplit, which data split to generate samples
        for (for example, training and evaluation).
    Yields:
      {"inputs": text, "labels": int[]}
    """
    raise NotImplementedError()

  # START: Additional subclass interface
  @property
  def num_classes(self):
    """The number of classes."""
    raise NotImplementedError()

  def class_labels(self, data_dir):
    """String representation of the classes."""
    del data_dir
    return ["ID_%d" % i for i in range(self.num_classes)]

  # END: Additional subclass interface

  def generate_text_for_vocab(self, data_dir, tmp_dir):
    for i, sample in enumerate(
        self.generate_samples(data_dir, tmp_dir, problem.DatasetSplit.TRAIN)):
      yield sample["inputs"]
      if self.max_samples_for_vocab and (i + 1) >= self.max_samples_for_vocab:
        break

  def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
    generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
    encoder = self.get_or_create_vocab(data_dir, tmp_dir)
    for sample in generator:
      inputs = encoder.encode(sample["inputs"])
      inputs.append(text_encoder.EOS_ID)
      labels = sample["labels"]
      yield {"inputs": inputs, "targets": labels}

  def feature_encoders(self, data_dir):
    encoder = self.get_or_create_vocab(data_dir, None, force_get=True)

    return {
        "inputs": encoder,
        "targets": [text_encoder.ClassLabelEncoder(label) for label in self.class_labels(data_dir)]
    }

  def hparams(self, defaults, unused_model_hparams):
    p = defaults
    p.modality = {"inputs": modalities.ModalityType.SYMBOL,
                  "targets": modalities.ModalityType.MULTI_LABEL}
    p.vocab_size = {"inputs": self._encoders["inputs"].vocab_size,
                    "targets": self.num_classes}

  def example_reading_spec(self):
    data_fields = {
        "inputs": tf.VarLenFeature(tf.int64),
        "targets": tf.FixedLenSequenceFeature((), tf.int64, allow_missing=True),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)