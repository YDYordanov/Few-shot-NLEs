import os
import copy
from tqdm import tqdm
from time import time
import jsonlines
import numpy as np

import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize

from transformers import T5ForConditionalGeneration


class WT5(nn.Module):
    def __init__(self, config, tokenizer):
        super(WT5, self).__init__()
        self.config = config
        if 't5' in self.config['lm_name']:
            self.model = T5ForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=self.config['lm_name'])
        else:
            raise NotImplementedError

        self.perpl_criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config['fp16'])
        self.tokenizer = tokenizer

        # Process special tokens of T5
        unk_token_id = self.tokenizer.unk_token_id
        print('Unk token id: ', unk_token_id)
        if 't5' in self.config['lm_name']:
            self.special_token_ids = {}
            for tok in ['<pad>', '</s>', '▁explanation', '.', ':', ',']:
                token_id = self.tokenizer.convert_tokens_to_ids(tok)
                assert token_id != unk_token_id
                self.special_token_ids[tok] = token_id
            print(self.special_token_ids, '\n')
            self.pad_id = self.special_token_ids['<pad>']
            self.eos_id = self.special_token_ids['</s>']
            self.sos_id = self.eos_id
            self.explain_id = self.special_token_ids['▁explanation']
            self.full_stop_id = self.special_token_ids['.']
            self.column_id = self.special_token_ids[':']
            self.comma_id = self.special_token_ids[',']
        else:
            raise NotImplementedError

        np.random.seed(124423)

    def send_dict_to_device(self, input_dict):
        for key in input_dict.keys():
            input_dict[key] = input_dict[key].to(self.device)
        return input_dict

    def forward(self, input_ids, labels=None):
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs[0]
        logits = outputs[1]
        return loss, logits

    def map_decode(self, sentence_ids_batch):
        if isinstance(sentence_ids_batch, torch.Tensor):
            sentence_ids_batch = sentence_ids_batch.cpu().tolist()
        return list(map(lambda x: self.tokenizer.decode(
            x, skip_special_tokens=True, clean_up_tokenization_spaces=True),
                        sentence_ids_batch))

    def remove_junk(self, id_list):
        # Strip padding
        id_list = [idx for idx in id_list if idx != self.pad_id]

        # Remove <\s> and <s> tokens
        id_list = list(
            filter(lambda a: a not in [self.eos_id, self.sos_id], id_list))

        return id_list

    def cut_label(self, id_list):
        # Cut id_list to leave only the label (rm the NLE and junk)

        original_id_list = copy.deepcopy(id_list)

        id_list = self.remove_junk(id_list)

        try:
            column_position = id_list.index(self.column_id)
        except ValueError:
            column_position = None

        if column_position is not None:
            id_list = id_list[:column_position-1]

        # Fix a bug where the WG model outputs a full stop after the label
        if len(id_list) == 0:
            print('Faulty example (code 1):', original_id_list)
            print(self.tokenizer.decode(
                original_id_list, skip_special_tokens=False,
                clean_up_tokenization_spaces=False))
            return id_list
        elif id_list[-1] == self.full_stop_id:
            id_list = id_list[:-1]
        if len(id_list) == 0:
            if not self.config['silent']:
                print('Faulty example (code 2):', original_id_list)
                print(self.tokenizer.decode(
                        original_id_list, skip_special_tokens=False,
                        clean_up_tokenization_spaces=False))
        return id_list

    def compare_labels(self, targ_ids, out_ids):
        """Compare the output ids to the target ids up to the
        [explanation] or <eos> tokens"""
        out_ids = self.cut_label(out_ids)
        targ_ids = self.cut_label(targ_ids)

        return targ_ids == out_ids

    def extract_nle_ids(self, word_ids):
        """
        Extract the ids of the NLE, and the start position
         of the NLE in the word_ids list;
         further, extract the id of the first pad token
        Note: this function is applied only to the target output
        :param word_ids:
        :return:
        """
        if word_ids[0] == self.pad_id:
            word_ids = word_ids[1:]

        try:
            column_position = word_ids.index(self.column_id)
            nle_start_position = column_position + 1
        except ValueError:
            column_position = None
            nle_start_position = None
        try:
            pad_position = word_ids.index(self.pad_id)
        except ValueError:
            pad_position = None

        if column_position is None:
            nle_ids = []
        elif len(word_ids) < column_position + 2:
            nle_ids = []
        elif pad_position is None:
            nle_ids = word_ids[column_position + 1:]
        else:
            nle_ids = word_ids[column_position + 1: pad_position]
        return nle_ids, nle_start_position, pad_position

    def evaluate(self, loader, report_bleu=False, print_predictions=False,
                 force_explain=False, save_predictions=False,
                 prediction_save_path=None):
        """
        Compute test loss and BLEU score
        :param prediction_save_path:
        :param save_predictions:
        :param force_explain:
        :param report_bleu:
        :param print_predictions:
        :param loader:
        """
        total_loss = 0
        total_full_loss = 0
        total_full_entries = 0
        total_nle_loss = 0
        total_nle_length = 0
        total_num_correct = 0
        total_num_datapoints = 0
        all_generated_sentences = []
        all_target_sentences = []
        all_input_sentences = []
        target_sentences = []
        generated_sentences = []
        all_list_predictions = []
        summary_bleu_score = None
        for step, batch_dict in tqdm(enumerate(loader), total=len(loader)):
            with torch.no_grad():
                batch_dict = self.send_dict_to_device(batch_dict)
                input_ids = batch_dict['input_tensor']
                labels = batch_dict['target_tensor']

                time1 = time()
                loss, batch_logits = self.forward(input_ids, labels)

                time1 = time()
                list_labels = labels.cpu().tolist()

                # Note: this loss is teacher-forced
                total_loss += loss.item()

                time1 = time()
                # Compute nle-only perplexity
                nle_losses_tensor = self.perpl_criterion(
                    batch_logits.permute(0, 2, 1), labels)
                # Sum the nle_losses based on the positions
                # of the labels corresponding to the nle-s
                list_nle_losses = nle_losses_tensor.cpu().tolist()
                for nle_losses, label in zip(list_nle_losses, list_labels):
                    nle_ids, nle_pos, pad_pos = self.extract_nle_ids(label)
                    if nle_pos is None:
                        selected_losses = []
                        selected_full_losses = nle_losses
                    elif pad_pos is None:
                        selected_losses = nle_losses[nle_pos:]
                        selected_full_losses = nle_losses
                    else:
                        selected_losses = nle_losses[nle_pos: pad_pos]
                        selected_full_losses = nle_losses[: pad_pos]
                    total_nle_loss += sum(selected_losses)
                    total_full_loss += sum(selected_full_losses)
                    assert len(selected_losses) == len(nle_ids)
                    total_nle_length += len(nle_ids)
                    total_full_entries += len(selected_full_losses)

                # Make predictions: beam search
                time1 = time()
                if print_predictions or report_bleu:
                    beam_size = self.config['eval_beam_size']
                else:
                    beam_size = 1  # speed-up dev evaluation

                if force_explain:
                    list_inputs = input_ids.cpu().tolist()
                    if 'wg' in loader.task_name:
                        option1_list = batch_dict['option1'].cpu().tolist()
                        option2_list = batch_dict['option2'].cpu().tolist()
                        candidates = [
                            [self.remove_junk(option1),
                             self.remove_junk(option2)] for option1, option2
                            in zip(option1_list, option2_list)]
                    elif 'comve' in loader.task_name:
                        label_list = ['1', '2']
                        candidates = [[loader.dataset.tokenizer(
                            label, return_tensors="pt"
                        ).input_ids.cpu().tolist()[0][:-1]
                                       for label in label_list
                                      ] for _ in list_inputs]
                    elif loader.task_name == 'esnli':
                        label_list = ['entailment', 'contradiction', 'neutral']
                        candidates = [[loader.dataset.tokenizer(
                            label, return_tensors="pt"
                        ).input_ids.cpu().tolist()[0][:-1]
                                       for label in label_list
                                      ] for _ in list_inputs]
                    else:
                        raise NotImplementedError
                    min_length = max(len(p) for p in candidates) + 2 + 1

                def prefix_allowed_tokens_fn(batch_id, x):
                    """
                    Constrains the next generations,
                    so that the "explanation" word is generated
                    after generating the label.
                    :param x:
                    :param batch_id:
                    :return:
                    """

                    if loader.task_name == 'ewg':
                        assert loader.dataset.input_format == 'default'
                    x_ids = x.cpu().tolist()
                    # Force to predict one of the 2 candidates,
                    # followed by "explanation:"
                    possible_labels = copy.deepcopy(candidates[batch_id])

                    if x_ids[0] in self.special_token_ids and len(x_ids) == 0:
                        return [x[0] for x in possible_labels]
                    x_ids = x_ids[1:]
                    possibilities = []

                    for candidate_id, candidate in enumerate(possible_labels):
                        candidate = copy.deepcopy(candidate)
                        candidate += [self.explain_id, self.column_id]
                        candidate_match = False
                        if len(candidate) >= len(x_ids):
                            if x_ids == candidate[:len(x_ids)]:
                                candidate_match = True
                        if candidate_match and len(candidate) > len(x_ids):
                            possibilities.append(candidate[len(x_ids)])

                    if len(possibilities) > 0:
                        return possibilities
                    else:
                        # No constraint on the next token
                        return list(range(
                            loader.dataset.tokenizer.vocab_size))

                if not force_explain:
                    prefix_allowed_tokens_fn = None
                    min_length = 0
                predictions = self.model.generate(
                    input_ids=input_ids, max_length=200,
                    min_length=min_length, num_beams=beam_size,
                    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)

                time1 = time()
                # Compute number of correctly predicted labels
                list_labels = labels.cpu().tolist()
                list_predictions = predictions.cpu().tolist()
                all_list_predictions += list_predictions

                if self.config['debug']:
                    if step == 0:
                        print('---------> labels and predictions at step 0: ')
                        for label, prediction in zip(
                                list_labels, list_predictions):
                            print(
                                self.tokenizer.convert_ids_to_tokens(label),
                                self.tokenizer.convert_ids_to_tokens(
                                    prediction))
                correctness = [self.compare_labels(
                    targ_ids=label, out_ids=prediction)
                    for label, prediction in zip(list_labels, list_predictions)
                ]

                num_correct = sum(correctness)
                total_num_correct += num_correct
                total_num_datapoints += len(correctness)

                if report_bleu or print_predictions:
                    time1 = time()
                    # Convert to sentences
                    generated_sents = self.map_decode(predictions)
                    all_generated_sentences += generated_sents
                    target_sents = self.map_decode(labels)
                    all_target_sentences += target_sents
                    input_sents = self.map_decode(input_ids)
                    all_input_sentences += input_sents

        if report_bleu or print_predictions:
            generated_sentences = [word_tokenize(sent)
                                   for sent in all_generated_sentences]
            target_sentences = [[word_tokenize(sent)]
                                for sent in all_target_sentences]

        def find_nle_token(word_list):
            # Find "explanation" token in a word list
            nle_token_id = None
            for w_id, w in enumerate(word_list):
                if w == 'explanation':
                    nle_token_id = w_id
                    break
            return nle_token_id

        def cut_nle(word_list):
            # Cut NLE from the output
            nle_token_id = find_nle_token(word_list)
            if nle_token_id is not None:
                if nle_token_id < len(word_list) - 2:
                    return word_list[nle_token_id + 2:]
                else:
                    return []
            else:
                return []

        # Print predictions
        if print_predictions:
            for i in range(len(all_generated_sentences)):
                gen_sent = all_generated_sentences[i]
                targ_sent = all_target_sentences[i]
                in_str = all_input_sentences[i]
                print('IN:', in_str, '\n', 'OUT:', gen_sent, '\n',
                      'TARGET:', targ_sent, '\n')

        # Save generated outputs (label&nle) to file for hand evaluation
        if save_predictions:
            nle_list = self.map_decode(
                [self.extract_nle_ids(pred)[0]
                 for pred in all_list_predictions])
            predicted_labels_list = self.map_decode(
                [self.cut_label(pred) for pred in all_list_predictions])
            prediction_list_to_save = [
                {'answer': label, 'nle': nle}
                for label, nle in zip(predicted_labels_list, nle_list)
            ]
            save_file = prediction_save_path
            with jsonlines.open(save_file, 'w') as dataset_file:
                dataset_file.write_all(prediction_list_to_save)

        # Compute BLEU score
        if report_bleu:
            # Remove the label, "explanation" and ":" tokens
            # from BLEU evaluation
            target_nles = [[cut_nle(x[0])] for x in target_sentences]
            generated_nles = [cut_nle(x) for x in generated_sentences]
            summary_bleu_score = corpus_bleu(target_nles, generated_nles)

        # Note: this teacher_loss is slightly imperfect, since
        # the final batch may be of different size
        teacher_loss = total_loss / len(loader)
        if total_nle_length > 0:
            nle_loss = total_nle_loss / total_nle_length
        else:
            nle_loss = None

        # Note: we cap large float values to prevent overflow
        return_dict = {
            'teacher_loss': teacher_loss,
            'full_loss': total_full_loss / total_full_entries,
            'nle_loss': nle_loss,
            'teacher_perplexity': (
                2 ** teacher_loss if teacher_loss < 1000 else 1e10 - 1),
            'nle_perplexity': (
                (2 ** nle_loss if nle_loss < 1000 else 1e10 - 1)
                if nle_loss is not None else None),
            'Accuracy': total_num_correct / total_num_datapoints
        }
        if report_bleu:
            return_dict['BLEU'] = summary_bleu_score
        return return_dict

    def run_epoch(self, epoch, epoch_size, train_loaders, valid_loaders,
                  tb_writer, save_dir, log_interval=200, save_interval=10e+10,
                  start_batch=1, print_predictions=False):
        buffer_loss = 0
        self.train()
        # Sample the datasets based on their size
        sizes = [len(loader.dataset) for loader in train_loaders]
        adjusted_sizes = [size ** 0.75 for size in sizes]
        sample_probs = [
            size / sum(adjusted_sizes) for size in adjusted_sizes]
        train_loader_iterators = [iter(loader) for loader in train_loaders]

        print('Data sizes and sample prob-s, resp.:', sizes, sample_probs)

        for step in tqdm(range(epoch_size)):
            with torch.cuda.amp.autocast(enabled=self.config['fp16']):
                task_id = int(np.nonzero(
                    np.random.multinomial(1, sample_probs))[0])
                task_choice = train_loader_iterators[task_id]
                batch_dict = next(task_choice)
                # Fast-forward the iterator until the resumption point:
                if start_batch > 1 and step < start_batch:
                    continue
                elif 1 < start_batch == step:
                    print('Resuming training...')

                batch_dict = self.send_dict_to_device(batch_dict)
                input_ids = batch_dict['input_tensor']
                labels = batch_dict['target_tensor']

                if (step + 1) % save_interval == 1 and step > 0:
                    print(0, input_ids)
                    print(labels)

                if 1 < start_batch == step:
                    print(1, input_ids)
                    print(labels)

                # Note: default behaviour is:
                # decoder_input_ids = (shifted lm labels to the right)
                loss = self.model(
                    input_ids=input_ids, labels=labels, return_dict=True).loss
                buffer_loss += loss.item()

            if (step + 1) % save_interval == 1 and step > 0:
                print(loss.item())
                print(self.model.state_dict()['lm_head.weight'])

            if 1 < start_batch == step:
                print(loss.item())
                print(self.model.state_dict()['lm_head.weight'])
                model_dir = 'saved_models/test/checkpoint.pth'
                import os
                if os.path.isfile(model_dir):
                    save_dict = torch.load(model_dir,
                                           map_location=self.device)
                    for key in self.model.state_dict().keys():
                        print(self.model.state_dict()[key].tolist() ==
                              save_dict['model_state_dict'][key].tolist())

            self.scaler.scale(loss).backward()
            if (step + 1) % self.config['grad_accum_steps'] == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            if (step + 1) % log_interval == 0:
                self.eval()
                log_step = step + epoch_size * (epoch - 1)
                for valid_loader in valid_loaders:
                    task_name = valid_loader.task_name
                    dev_results = self.evaluate(
                        valid_loader, print_predictions=print_predictions)
                    for result_key in dev_results.keys():
                        if dev_results[result_key] is not None:
                            tb_writer.add_scalar(
                                task_name + ' ' + result_key,
                                dev_results[result_key], log_step + 1
                            )
                    print('{} dev results:'.format(task_name),
                          dev_results)

                buffer_av_loss = buffer_loss / log_interval
                print('Train loss:', buffer_av_loss, '\n')
                tb_writer.add_scalar('Train loss', buffer_av_loss,
                                     log_step + 1)
                buffer_loss = 0
                self.train()

            if (step + 1) % save_interval == 0:
                print('Saving model...')
                save_dict = {
                    'epoch': epoch,
                    'mini_batch': step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict()
                }
                print('Model saved!')
                if self.scheduler is not None:
                    save_dict['scheduler_state_dict'] = \
                        self.scheduler.state_dict()
                torch.save(save_dict, '{}/checkpoint.pth'.format(save_dir))
