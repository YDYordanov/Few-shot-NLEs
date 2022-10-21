import os
import argparse
import json
import math

import torch

from transformers import AdamW, T5Tokenizer, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter

from utils import WT5Loader
from models import WT5
from gpu_utils import update_gpu_synch_file


def main(args):
    if args.use_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.use_devices
        if args.debug:
            os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)  # for debugging only
        print("Using gpus:", args.use_devices)
    if args.use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device("cuda")

    tb_writer = SummaryWriter(logdir=args.save_dir)

    if args.load_model is not None:
        config_file = os.path.join(args.load_model, 'config.txt')
        with open(config_file, 'r') as f:
            print('Opening:', config_file)
            config_dict = json.load(f)
        for key in config_dict.keys():
            if key in ['lm_name', 'input_length']:
                setattr(args, key, config_dict[key])
        config_file = os.path.join(args.save_dir, 'config.txt')
        with open(config_file, 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        print(args.__dict__)
    elif args.resume_from_checkpoint is not None:
        config_file = os.path.join(args.resume_from_checkpoint, 'config.txt')
        with open(config_file, 'r') as f:
            print('Opening:', config_file)
            config_dict = json.load(f)
        for key in config_dict.keys():
            if key not in ['resume_from_checkpoint']:
                setattr(args, key, config_dict[key])
        config_file = os.path.join(args.save_dir, 'config.txt')
        with open(config_file, 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        print(args.__dict__)
    elif args.evaluate_model is not None:
        config_file = os.path.join(args.evaluate_model, 'config.txt')
        with open(config_file, 'r') as f:
            print('Opening:', config_file)
            config_dict = json.load(f)
        for key in config_dict.keys():
            if key in ['lm_name', 'input_length']:
                setattr(args, key, config_dict[key])
        args.num_epochs = 0
        print(args.__dict__)
    elif args.dev_evaluate_model is not None:
        config_file = os.path.join(args.dev_evaluate_model, 'config.txt')
        with open(config_file, 'r') as f:
            print('Opening:', config_file)
            config_dict = json.load(f)
        for key in config_dict.keys():
            if key in ['lm_name', 'input_length']:
                setattr(args, key, config_dict[key])
        args.num_epochs = 0
        print(args.__dict__)
    else:
        # Normal training mode: only use arguments from argparse
        config_file = os.path.join(args.save_dir, 'config.txt')
        with open(config_file, 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        print(args.__dict__)

    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed(args.model_seed)
    torch.backends.cudnn.deterministic = True

    args.train_b_size = args.train_b_size // args.grad_accum_steps

    if 't5' in args.lm_name:
        tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=args.lm_name)
    else:
        raise NotImplementedError

    esnli = {
        'train_data_path': args.esnli_train_data_path,
        'dev_data_path': args.esnli_dev_data_path,
        'test_data_path': args.esnli_test_data_path
    }
    ewg = {
        'train_data_path': args.ewg_train_data_path,
        'dev_data_path': args.ewg_dev_data_path,
        'test_data_path': args.ewg_test_data_path
    }
    comve = {
        'train_data_path': args.comve_train_data_path,
        'dev_data_path': args.comve_dev_data_path,
        'test_data_path': args.comve_test_data_path
    }
    if args.training_task == 'esnli':
        tasks = ['esnli']
        test_tasks = ['esnli']
        train_data_paths = [esnli['train_data_path']]
        dev_data_paths = [esnli['dev_data_path']]
        test_data_paths = [esnli['test_data_path']]
        force_explains = [False]
    elif args.training_task == 'ewg':
        tasks = ['ewg']
        test_tasks = ['ewg']
        train_data_paths = [ewg['train_data_path']]
        dev_data_paths = [ewg['dev_data_path']]
        test_data_paths = [ewg['test_data_path']]
        force_explains = [True]
    elif args.training_task == 'esnli_ewg':
        tasks = ['esnli', 'ewg']
        test_tasks = ['esnli', 'ewg']
        train_data_paths = [esnli['train_data_path'], ewg['train_data_path']]
        dev_data_paths = [esnli['dev_data_path'], ewg['dev_data_path']]
        test_data_paths = [esnli['test_data_path'], ewg['test_data_path']]
        force_explains = [False, True]
    elif args.training_task == 'esnli+ewg':
        tasks = ['esnli+ewg']
        test_tasks = ['esnli', 'ewg']
        train_data_paths = [[esnli['train_data_path'], ewg['train_data_path']]]
        dev_data_paths = [esnli['dev_data_path'], ewg['dev_data_path']]
        test_data_paths = [esnli['test_data_path'], ewg['test_data_path']]
        force_explains = [False, True]
    elif args.training_task == 'comve':
        tasks = ['comve']
        test_tasks = ['comve']
        train_data_paths = [comve['train_data_path']]
        dev_data_paths = [comve['dev_data_path']]
        test_data_paths = [comve['test_data_path']]
        force_explains = [True]
    elif args.training_task == 'esnli_comve':
        tasks = ['esnli', 'comve']
        test_tasks = ['esnli', 'comve']
        train_data_paths = [esnli['train_data_path'], comve['train_data_path']]
        dev_data_paths = [esnli['dev_data_path'], comve['dev_data_path']]
        test_data_paths = [esnli['test_data_path'], comve['test_data_path']]
        force_explains = [False, True]
    elif args.training_task == 'esnli+comve':
        tasks = ['esnli+comve']
        test_tasks = ['esnli', 'comve']
        train_data_paths = [[esnli['train_data_path'], comve['train_data_path']]]
        dev_data_paths = [esnli['dev_data_path'], comve['dev_data_path']]
        test_data_paths = [esnli['test_data_path'], comve['test_data_path']]
        force_explains = [False, True]
    else:
        raise NotImplementedError
    train_loaders = [
        WT5Loader(
            data_paths=train_data_path, tokenizer=tokenizer,
            input_length=args.input_length, batch_size=args.train_b_size,
            task_name=task, do_train=True,
            input_format=args.wg_input_format
        ).data_loader
        for task, train_data_path in zip(tasks, train_data_paths)
    ]
    valid_loaders = [
        WT5Loader(
            data_paths=dev_data_path, tokenizer=tokenizer,
            input_length=args.input_length, batch_size=args.dev_b_size,
            task_name=task, do_train=False,
            force_explain=force_explain, input_format=args.wg_input_format
        ).data_loader
        for task, dev_data_path, force_explain in zip(
            test_tasks, dev_data_paths, force_explains)
    ]
    test_loaders = [
        WT5Loader(
            data_paths=test_data_path, tokenizer=tokenizer,
            input_length=args.input_length, batch_size=args.dev_b_size,
            task_name=task, do_train=False,
            force_explain=(not args.force_predict_test),
            input_format=args.wg_input_format
        ).data_loader
        for task, test_data_path in zip(
            test_tasks, test_data_paths)
    ]

    if 'ewg' in args.training_task:
        wg_dev_nles_path = 'Data/e-WG/new_dev_nles_only.jsonl'
        wg_dev_nle_loader = WT5Loader(
            data_paths=wg_dev_nles_path, tokenizer=tokenizer,
            input_length=args.input_length, batch_size=args.dev_b_size,
            task_name='wg_nles', do_train=False,
            force_explain=False, input_format=args.wg_input_format
        ).data_loader
        valid_loaders.append(wg_dev_nle_loader)

        # Add WG loader for accuracy evaluation
        wg_dev_loader = WT5Loader(
            data_paths=ewg['dev_data_path'], tokenizer=tokenizer,
            input_length=args.input_length, batch_size=args.dev_b_size,
            task_name='wg_acc', do_train=False,
            force_explain=False, input_format=args.wg_input_format
        ).data_loader
        valid_loaders.append(wg_dev_loader)

    if 'comve' in args.training_task:
        comve_dev_nles_path = 'Data/ComVE/dev.csv'
        comve_dev_nle_loader = WT5Loader(
            data_paths=comve_dev_nles_path, tokenizer=tokenizer,
            input_length=args.input_length, batch_size=args.dev_b_size,
            task_name='comve_nles', do_train=False,
            force_explain=False
        ).data_loader
        valid_loaders.append(comve_dev_nle_loader)

        # Add ComVE loader for accuracy evaluation
        comve_acc_dev_path = 'Data/ComVE/dev_no_nles.csv'
        comve_dev_loader = WT5Loader(
            data_paths=comve_acc_dev_path, tokenizer=tokenizer,
            input_length=args.input_length, batch_size=args.dev_b_size,
            task_name='comve_acc', do_train=False,
            force_explain=False
        ).data_loader
        valid_loaders.append(comve_dev_loader)

    if 'ewg' in args.training_task and args.quick_evaluation:
        valid_loaders = [loader for loader in valid_loaders
                         if loader.task_name == 'wg_nles']
    if 'comve' in args.training_task and args.quick_evaluation:
        valid_loaders = [loader for loader in valid_loaders
                         if loader.task_name == 'comve_nles']

    config = {
        'lm_name': args.lm_name,
        'lr': args.lr,
        'eval_beam_size': args.beam_size,
        'grad_accum_steps': args.grad_accum_steps,
        'fp16': args.fp16,
        'debug': args.debug,
        'silent': args.silent
    }

    model = WT5(config, tokenizer=tokenizer)
    model.to(device)
    model.device = device

    if args.optimizer == 'adamw':
        optimizer_class = AdamW
    elif args.optimizer == 'sgd':
        optimizer_class = torch.optim.SGD
    else:
        raise NotImplementedError
    optimizer = optimizer_class(lr=config['lr'], params=model.parameters())
    model.optimizer = optimizer
    # Note: this epoch size is so that the dataloaders don't overflow
    epoch_size = min([len(loader) for loader in train_loaders])
    num_training_steps = epoch_size * args.num_epochs // \
        args.grad_accum_steps
    if args.scheduler is None:
        model.scheduler = None
    elif args.scheduler == 'linear':
        model.scheduler = get_linear_schedule_with_warmup(
            optimizer, num_training_steps=num_training_steps,
            num_warmup_steps=math.floor(num_training_steps *
                                        args.warmup_proportion))
    else:
        raise NotImplementedError

    if args.evaluate_model is not None:
        model_folder = args.evaluate_model
        model_dir = os.path.join(model_folder, 'final_model.pth')
        state_dict = torch.load(model_dir, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        for test_loader in test_loaders:
            if args.wg_evaluate:
                if 'wg' not in test_loader.task_name:
                    continue
            if args.comve_evaluate:
                if 'comve' not in test_loader.task_name:
                    continue
            test_results_dict = model.evaluate(
                test_loader, print_predictions=args.print_predictions,
                report_bleu=True, force_explain=args.force_explain,
                save_predictions=args.save_predictions,
                prediction_save_path=os.path.join(
                    model_folder, args.predictions_file_name)
            )
            task_name = test_loader.task_name
            print('Results on {}:'.format(task_name), test_results_dict)
            test_results_file = os.path.join(
                args.evaluate_model, task_name+'_test_results.json')
            with open(test_results_file, 'w') as fp:
                json.dump(test_results_dict, fp)

    if args.dev_evaluate_model is not None:
        model_folder = args.dev_evaluate_model
        model_dir = os.path.join(model_folder, 'final_model.pth')
        state_dict = torch.load(model_dir, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        for dev_loader in valid_loaders:
            if args.wg_evaluate:
                if 'wg' not in dev_loader.task_name:
                    continue
            if args.comve_evaluate:
                if 'comve' not in dev_loader.task_name:
                    continue
            dev_results_dict = model.evaluate(
                dev_loader, print_predictions=args.print_predictions,
                report_bleu=True, force_explain=args.force_explain,
                save_predictions=args.save_predictions,
                prediction_save_path=os.path.join(
                    model_folder, args.predictions_file_name)
            )
            task_name = dev_loader.task_name
            print('Results on {}:'.format(task_name), dev_results_dict)
            dev_results_file = os.path.join(
                args.dev_evaluate_model, 'final_'+task_name+'_dev_results.json')
            with open(dev_results_file, 'w') as fp:
                json.dump(dev_results_dict, fp)

    if args.load_model is not None:
        model_folder = args.load_model
        print('Loading model {} ...'.format(model_folder))
        model_dir = os.path.join(model_folder, 'final_model.pth')
        state_dict = torch.load(model_dir, map_location=device)
        model.load_state_dict(state_dict, strict=True)

    epoch = 1
    train_start_batch = 1
    if args.resume_from_checkpoint is not None:
        model_folder = args.resume_from_checkpoint
        print('Resuming training of {} ...'.format(model_folder))
        checkpoint = os.path.join(model_folder, 'checkpoint.pth')
        save_dict = torch.load(checkpoint, map_location=device)
        epoch = save_dict['epoch']
        train_start_batch = save_dict['mini_batch'] + 1
        model.model.load_state_dict(save_dict['model_state_dict'])
        model.optimizer.load_state_dict(
            save_dict['optimizer_state_dict'])
        model.scaler.load_state_dict(
            save_dict['scaler_state_dict']
        )
        if model.scheduler is not None:
            model.scheduler.load_state_dict(
                save_dict['scheduler_state_dict'])
        print('Model resumed from epoch {} and batch {}.'
              .format(epoch, train_start_batch))

        if train_start_batch >= len(train_loaders[0]) + 1:
            epoch += 1

    if args.pre_evaluate:
        print('Pre-evaluation...')
        for valid_loader in valid_loaders:
            valid_results_dict = model.evaluate(
                valid_loader, print_predictions=args.print_predictions,
                force_explain=args.force_explain)
            print('Zero-shot dev {} results:'.format(valid_loader.task_name),
                  valid_results_dict)

    num_epochs = args.num_epochs

    print('Training for {} epochs...'.format(num_epochs))
    for epoch_id in range(num_epochs):
        print('Epoch {}'.format(epoch))
        model.run_epoch(
            epoch=epoch,
            epoch_size=epoch_size,
            train_loaders=train_loaders,
            valid_loaders=valid_loaders, tb_writer=tb_writer,
            save_dir=args.save_dir, log_interval=args.log_interval,
            save_interval=args.save_interval,
            start_batch=train_start_batch,
            print_predictions=args.print_predictions)
        epoch += 1
        train_start_batch = 1

    if num_epochs > 0:
        print('Saving final model...')
        if not args.discard_saved_model:
            torch.save(model.state_dict(),
                       '{}/final_model.pth'.format(args.save_dir))
        print('Evaluating final model...')
        model.eval()
        for dev_loader in valid_loaders:
            result_dict = model.evaluate(
                dev_loader, report_bleu=True,
                print_predictions=args.print_predictions)
            with open('{}/final_{}_dev_results.json'.format(
                    args.save_dir, dev_loader.task_name), 'w') as fp:
                json.dump(result_dict, fp)
            print(result_dict)

        print('Deleting checkpoint...')
        checkpoint_path = '{}/checkpoint.pth'.format(args.save_dir)
        if os.path.isfile(checkpoint_path):
            os.system('rm {}'.format(checkpoint_path))

    # Update the GPU synch file
    if args.use_devices is not None:
        if len(args.use_devices) == 1:
            update_gpu_synch_file(args.gpu_synch_file, int(args.use_devices),
                                  is_running=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformers model")
    parser.add_argument("--use_devices", '-gpu', type=str, default=None,
                        help="comma-separated list of devices to run on")
    parser.add_argument('--save_dir', type=str, default='saved_models/demo',
                        help="Directory to save the models (for resumption)")
    parser.add_argument('--discard_saved_model', action='store_true',
                        help="Discard the saved model for large-scale exper-s")
    parser.add_argument('--training_task', '-task', choices=[
        'esnli', 'ewg', 'esnli_ewg', 'esnli+ewg',
        'comve', 'esnli_comve', 'esnli+comve'],
                        default='esnli', help="Choose the training task")
    parser.add_argument('--esnli_train_data_path', type=str,
                        default='Data/e-SNLI/esnli_train.csv',
                        help="Path to the training data for e-WG")
    parser.add_argument('--esnli_dev_data_path', type=str,
                        default='Data/e-SNLI/esnli_dev.csv',
                        help="Path to the dev data for e-SNLI")
    parser.add_argument('--esnli_test_data_path', type=str,
                        default='Data/e-SNLI/esnli_test.csv',
                        help="Path to the test data for e-SNLI")
    parser.add_argument('--ewg_train_data_path', type=str,
                        default='Data/e-WG/new_train.jsonl',
                        help="Path to the training data for e-WG")
    parser.add_argument('--ewg_dev_data_path', type=str,
                        default='Data/e-WG/new_dev.jsonl',
                        help="Path to the dev data for e-WG")
    parser.add_argument('--ewg_test_data_path', type=str,
                        default='Data/e-WG/dev.jsonl',
                        help="Path to the test data for e-WG")
    parser.add_argument('--comve_train_data_path', type=str,
                        default='Data/ComVE/train_no_nles.csv',
                        help="Path to the dev data for ComVE")
    parser.add_argument('--comve_dev_data_path', type=str,
                        default='Data/ComVE/dev.csv',
                        help="Path to the dev data for ComVE")
    parser.add_argument('--comve_test_data_path', type=str,
                        default='Data/ComVE/test.csv',
                        help="Path to the dev data for ComVE")
    parser.add_argument('--exper_name', type=str, default=None,
                        help="The name of the experiment which "
                             "contains this run")
    parser.add_argument('--model_seed', '-seed', type=int, default=2809)
    parser.add_argument('--debug', '-debug', action='store_true',
                        help='Debug mode')
    parser.add_argument('--silent', '-silent', action='store_true',
                        help='Silent mode: no warnings printed!')
    parser.add_argument('--use_cpu', '-cpu', action='store_true',
                        help='Use CPU to run')
    parser.add_argument('--load_model', '-load', type=str, default=None,
                        help='Load a pre-trained model with part of its '
                             'config file')
    parser.add_argument('--resume_from_checkpoint', '-resume_checkpoint',
                        type=str, default=None,
                        help='Resume training from model checkpoint')
    parser.add_argument('--evaluate_model', '-eval', type=str, default=None,
                        help='Run evaluation on pre-trained model,'
                             'located in the corresponding folder,'
                             'final_model.pth')
    parser.add_argument('--dev_evaluate_model', '-dev_eval', type=str,
                        default=None,
                        help='Run final dev evaluation on pre-trained model,'
                             'located in the corresponding folder,'
                             'final_model.pth')
    parser.add_argument('--save_predictions', '-save_predictions',
                        action='store_true',
                        help='Save model predictions (w.r.t. test set)'
                             ' to file for hand evaluation.')
    parser.add_argument('--wg_evaluate', '-wg_eval',
                        action='store_true',
                        help='Evaluate on WG only, for --evaluate_model')
    parser.add_argument('--comve_evaluate', '-comve_eval',
                        action='store_true',
                        help='Evaluate on ComVE only, for --evaluate_model')
    parser.add_argument('--print_predictions', '-print',
                        action='store_true',
                        help='Print predictions of evaluated model '
                             'on test data')
    parser.add_argument('--predictions_file_name', '-pred_file',
                        type=str, default='predictions.jsonl',
                        help='Name of the predictions file')
    parser.add_argument('--force_explain', '-force_explain',
                        action='store_true',
                        help='Feed "explanation" at the model output'
                             'to force it to explain.')
    parser.add_argument('--force_predict_test', '-force_predict_test',
                        action='store_true',
                        help='Prediction mode for WG datasets. '
                             'For test loaders only')
    parser.add_argument('--pre_evaluate', '-pre_eval', action='store_true',
                        help='Zero-shot evaluate the pre-trained model')
    parser.add_argument('--quick_evaluation', '-quick_eval',
                        action='store_true',
                        help='Quick evaluation for WG on WG_dev_nles_only')
    parser.add_argument('--log_interval', '-log_int', type=int, default=200)
    parser.add_argument('--save_interval', '-save_int', type=int,
                        default=10e+10)
    parser.add_argument('--gpu_synch_file', '-gpu_synch', type=str,
                        default='gpu_synch.txt',
                        help="GPU synchronization script, for scheduling")
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help="Number of updates steps to accumulate "
                             "before performing a backward/update pass."
                             "This effectively allows for larger batch "
                             "sizes to be used.")
    parser.add_argument('--fp16', '-fp16', action='store_true',
                        help='Mixed precision training FP16/32.'
                             'Note: usually needs Pascal GPU or newer.')

    parser.add_argument('--lm_name', '-lm', choices=[
        't5-small', 't5-base', 't5-large', 't5-3b', 't5-11b'],
                        default='t5-base', help="Choose a generative LM")
    parser.add_argument('--lr', '-lr', type=float,
                        default=1e-4, help='Learning rate')
    parser.add_argument('--input_length', '-in_len', type=int, default=1024,
                        help="Maximum lenght of input. Not sure if I need it.")
    parser.add_argument('--wg_input_format', '-input', choices=[
        'default', 't5', 'nli', 'no_options', 'correct_option_prompt'],
                        default='default',
                        help='The WG input format: default, T5, NLI,'
                             'without given options, and with a prompt '
                             'for the correct option')
    parser.add_argument('--train_b_size', '-bs', type=int, default=16,
                        help="Train batch size")
    parser.add_argument('--dev_b_size', '-dbs', type=int, default=20,
                        help="Dev batch size")
    parser.add_argument('--beam_size', '-beam', type=int, default=1,
                        help="The beam size in beam search decoding "
                             "for NLE generation in evaluation.")
    parser.add_argument('--num_epochs', '-ep', type=int, default=1,
                        help="Number of epochs for Backprop")
    parser.add_argument('--optimizer', choices=['adamw', 'sgd'],
                        default='adamw', help='weight optimizer')
    parser.add_argument('--scheduler', choices=[None, 'linear'],
                        default=None, help='lr scheduler with warm-up')
    parser.add_argument('--warmup_proportion', '-warm_prop', type=float,
                        default=0.1, help='proportion of training data'
                                          ' to warm-up the scheduler')

    args = parser.parse_args()

    main(args=args)
