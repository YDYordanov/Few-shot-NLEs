import torch
import statistics
import subprocess

from tqdm import tqdm
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor


def eval_nlp_scores(pred, gt, verbose=False):
    """
    evaluates the nlp scores bleu1-bleu4, and meteor

    Args:
        pred (List): List of predictions
        gt (List): List of ground truths
    """
    if len(pred) == len(gt) == 0:
        return {}

    gts = {}
    res = {}

    for imgId in range(len(pred)):
        gts[imgId] = gt[imgId]
        res[imgId] = pred[imgId]

    # Set up scorers
    if verbose: print('Setting up scorers...')
    results = {}
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), ["METEOR"])
    ]
    # Compute scores
    for scorer, method in scorers:
        if verbose:
            print('Computing %s score...'%(scorer.method()))
        try:
            # NOTE: Some scorers are very verbose :/
            corpus_score, sentence_scores = scorer.compute_score(gts, res)
            
            # iterate (for bleu)
            for ind in range(len(method)):
                cs, ss = corpus_score, sentence_scores
                if isinstance(corpus_score, list):
                    cs, ss = corpus_score[ind], sentence_scores[ind]

                results[method[ind]] = cs, ss

                if verbose:
                    print("%s: %0.3f"%(method[ind], cs))

        except subprocess.CalledProcessError:
            if verbose:
                print(f"Error during calling of {method} in local rank {self.local_rank}.")

    return results


def input_subset(gt_labels, predicted_labels, gt_nles, predicted_nles):
    """
    Get subset of examples where label was predicted correctly.
    We only measure NLG metrics for those.
    """
    if not(len(gt_labels) == len(gt_nles) ==
           len(predicted_labels) == len(predicted_nles)):
        print(len(gt_labels), len(gt_nles), len(predicted_labels),
              len(predicted_nles))
    assert len(gt_labels) == len(gt_nles) == \
        len(predicted_labels) == len(predicted_nles)
    gt_nles = [
        gt_nles[i] for i in range(len(gt_nles))
        if gt_labels[i] == predicted_labels[i]
    ]
    predicted_nles = [
        predicted_nles[i] for i in range(len(gt_nles))
        if gt_labels[i] == predicted_labels[i]
    ]
    return gt_nles, predicted_nles


def get_nlg_scores(gen_expl, gt_expl, bert_metric, device):

    # getting NLG metrics
    nlg_scores = eval_nlp_scores([[x] for x in gen_expl], gt_expl)
    nlg_global_scores = {k: v[0] for k,v in nlg_scores.items()}
    for gen_nle, gt_nle_triplet in tqdm(zip(gen_expl, gt_expl)):
        bert_metric.add_batch(predictions=[gen_nle],
                              references=[gt_nle_triplet])
    bert_scores = bert_metric.compute(
        model_type='distilbert-base-uncased',
        device=device)
    nlg_global_scores['BERTScore'] = sum(bert_scores['f1']) / len(
        bert_scores['f1'])

    return nlg_global_scores

