import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
import argparse
import tqdm
from allennlp.common import Params
from allennlp.data.iterators import BasicIterator
from allennlp.data import DatasetReader
from allennlp.models import Model
from typing import TextIO, Optional, List

def main(serialization_directory, device, evaluation_data_path):
    """
    serialization_directory : str, required.
        The directory containing the serialized weights.
    device: int, default = -1
        The device to run the evaluation on.
    """

    config = Params.from_file(os.path.join(serialization_directory, "model_params.json"))
    dataset_reader = DatasetReader.from_params(config['dataset_reader'])
    if not evaluation_data_path:
        evaluation_data_path = config['validation_data_path']

    print("Loading model from {}model.tar.gz.".format(serialization_directory))
    model = Model.load(config,
                       serialization_dir=serialization_directory,
                       cuda_device=device)

    # Load the evaluation data and index it.
    print("Reading evaluation data from {}".format(evaluation_data_path))
    dataset = dataset_reader.read(evaluation_data_path)
    dataset.index_instances(model.vocab)
    iterator = BasicIterator(batch_size=32)
    batches = iterator(dataset,
                       num_epochs=1,
                       shuffle=False,
                       cuda_device=device,
                       for_training=False)

    print("Decoding with loaded model.")
    model.fast_mode = False
    model_predictions = []
    for batch in tqdm.tqdm(batches):
        result = model(**batch)
        output_dict = model.decode(result)
        model_predictions.extend(output_dict["tags"])

    print("Writing model predictions to file.")
    prediction_file_path = os.path.join(serialization_directory, "predictions.txt")
    gold_file_path = os.path.join(serialization_directory, "gold.txt")
    prediction_file = open(prediction_file_path, "w+")
    gold_file = open(gold_file_path, "w+")

    for instance, prediction in tqdm.tqdm(zip(dataset.instances, model_predictions)):
        fields = instance.fields
        predicted_tags = [model.vocab.get_token_from_index(p, namespace="labels") for p in prediction]

        if "targets" in fields:
            verb_index = fields["targets"].labels.index(1)
        elif "verb_indicator" in fields:
            try:
                # Most sentences have a verbal predicate, but not all.
                verb_index = fields["verb_indicator"].labels.index(1)
            except ValueError:
                verb_index = None
        else:
            verb_index = None

        frame = None
        if "frame" in fields:
            frame = fields["frame"].tokens[0].text
        gf = None
        if "gf" in fields:
            gf = [g.text for g in fields["gf"].tokens]
        pt = None
        if "pt" in fields:
            pt = [p.text for p in fields["pt"].tokens]

        sentence = [token.text for token in fields["tokens"].tokens]
        gold_tags = convert_spans_into_sequence_of_tags(tag_matrix=fields["tags"].labels,
                                                        max_span_width=model.max_span_width,
                                                        sentence_length=len(sentence))

        write_to_conll_eval_file(prediction_file,
                                 gold_file,
                                 verb_index,
                                 sentence,
                                 predicted_tags,
                                 gold_tags,
                                 frame,
                                 gf,
                                 pt)
    prediction_file.close()
    gold_file.close()

    print("Evaluation.")
    metrics = model.get_metrics()
    metrics_file_path = os.path.join(serialization_directory, "eval_metrics.txt")
    metrics_file = open(metrics_file_path, "w+")

    for key, metric in metrics.items():
        if "overall" in key:
            print("{}: {}".format(key, metric))
        metrics_file.write(key.ljust(25))
        metrics_file.write(str(metric).rjust(15))
        metrics_file.write('\n')


def convert_spans_into_sequence_of_tags(tag_matrix: List[str],
                                        max_span_width: int,
                                        sentence_length: int) -> List[int]:
    tag_sequence = ["O" for _ in range(sentence_length)]
    for end_idx in range(sentence_length):
        for diff in range(max_span_width):
            if diff > end_idx:
                break
            span_tag = tag_matrix[end_idx*max_span_width + diff]
            if span_tag == "*":
                continue
            start_idx = end_idx - diff
            for position in range(start_idx, end_idx+1):
                # Make sure that the current position is not already assigned.
                assert tag_sequence[position] == "O"
                tag_sequence[position] = span_tag
    return tag_sequence


def write_to_conll_eval_file(prediction_file: TextIO,
                             gold_file: TextIO,
                             verb_index: Optional[int],
                             sentence: List[str],
                             prediction: List[str],
                             gold_labels: List[str],
                             frame: str = None,
                             gf: List[str] = None,
                             pt: List[str] = None):
    """
    Prints predicate argument predictions and gold labels for a single verbal
    predicate in a sentence to two provided file references.

    Parameters
    ----------
    prediction_file : TextIO, required.
        A file reference to print predictions to.
    gold_file : TextIO, required.
        A file reference to print gold labels to.
    verb_index : Optional[int], required.
        The index of the verbal predicate in the sentence which
        the gold labels are the arguments for, or None if the sentence
        contains no verbal predicate.
    sentence : List[str], required.
        The word tokens.
    prediction : List[str], required.
        The predicted BIO labels.
    gold_labels : List[str], required.
        The gold BIO labels.
    """
    assert len(sentence) == len(prediction) == len(gold_labels)
    verbs = ["-"] * len(sentence)
    if verb_index:
        verbs[verb_index] = "V-" + sentence[verb_index]
        if frame is not None:
            verbs[verb_index] = frame

    idx = 0
    for word, verb, predicted, gold in zip(sentence, verbs, prediction, gold_labels):
        fields = "{0:<20}\t{1:<10}".format(word, verb)
        if gf:
            fields = "{0:}\t{1:>10}".format(fields, gf[idx])
        if pt:
            fields = "{0:}\t{1:>10}".format(fields, pt[idx])

        gold_file.write("{}\t{}\n".format(fields, gold))
        prediction_file.write("{0:}\t{1:>25}\t{2:>25}\n".format(fields, gold, predicted))

        idx += 1

    prediction_file.write("\n")
    gold_file.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Write CONLL format semi-CRF SRL predictions"
                                                 " to file from a pretrained model.")
    parser.add_argument('--path', type=str,
                        help='The serialization directory.')
    parser.add_argument('--device', type=int, default=-1,
                        help='The device to load the model onto.')
    parser.add_argument('--eval', type=str,
                        help='File to evaluate.', default='')

    args = parser.parse_args()
    main(args.path, args.device, args.eval)
