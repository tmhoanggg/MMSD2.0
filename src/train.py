import os
from data_set import MyDataset
from torch.utils.data import DataLoader
import torch
import logging
from tqdm import tqdm, trange
from sklearn import metrics
import wandb
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def train(args, model, device, train_data, dev_data, test_data, processor):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.train_batch_size,
                              collate_fn=MyDataset.collate_func,
                              shuffle=True)
    total_steps = int(len(train_loader) * args.num_train_epochs)
    model.to(device)

    if args.optimizer_name == 'adafactor':
        from transformers.optimization import Adafactor, AdafactorSchedule

        print('Use Adafactor Optimizer for Training.')
        optimizer = Adafactor(
            model.parameters(),
            # lr=1e-3,
            # eps=(1e-30, 1e-3),
            # clip_threshold=1.0,
            # decay_rate=-0.8,
            # beta1=None,
            lr=None,
            weight_decay=args.weight_decay,
            relative_step=True,
            scale_parameter=True,
            warmup_init=True
        )
        scheduler = AdafactorSchedule(optimizer)
    elif args.optimizer_name == 'adam':
        print('Use AdamW Optimizer for Training.')
        from transformers.optimization import AdamW, get_linear_schedule_with_warmup
        if args.model == 'MV_CLIP':
            clip_params = list(map(id, model.model.parameters()))
            base_params = filter(lambda p: id(p) not in clip_params, model.parameters())
            optimizer = AdamW([
                    {"params": base_params},
                    {"params": model.model.parameters(),"lr": args.clip_learning_rate}
                    ], lr=args.learning_rate, weight_decay=args.weight_decay)

            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                    num_training_steps=total_steps)
        else:
            optimizer = optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon, weight_decay=args.weight_decay)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * total_steps),
                                                num_training_steps=total_steps)
    else:
        raise Exception('Wrong Optimizer Name!!!')


    max_f1 = 0.
    for i_epoch in trange(0, int(args.num_train_epochs), desc="Epoch", disable=False):
        sum_loss = 0.
        sum_step = 0

        iter_bar = tqdm(train_loader, desc="Iter (loss=X.XXX)", disable=False)
        model.train()

        for step, batch in enumerate(iter_bar):
            text_list, image_list, label_list, id_list = batch
            if args.model == 'MV_CLIP':
                inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
                labels = torch.tensor(label_list).to(device)

            loss, score = model(inputs,labels=labels)
            sum_loss += loss.item()
            sum_step += 1

            iter_bar.set_description("Iter (loss=%5.3f)" % loss.item())
            loss.backward()
            optimizer.step()
            if args.optimizer_name == 'adam':
                scheduler.step() 
            optimizer.zero_grad()

        wandb.log({'train_loss': sum_loss/sum_step})
        dev_f1 ,dev_precision, dev_recall = evaluate_acc_f1(args, model, device, dev_data, processor, macro=True, mode='dev')
        wandb.log({'dev_f1': dev_f1, 'dev_precision': dev_precision, 'dev_recall': dev_recall})
        print(f"i_epoch is {i_epoch}, dev_f1 is {dev_f1}, dev_precision is {dev_precision}, dev_recall is {dev_recall}")

        if dev_f1 > max_f1:
            max_f1 = dev_f1

            path_to_save = os.path.join(args.output_dir, args.model)
            if not os.path.exists(path_to_save):
                os.mkdir(path_to_save)
            model_to_save = (model.module if hasattr(model, "module") else model)
            torch.save(model_to_save.state_dict(), os.path.join(path_to_save, 'model.pt'))

            test_f1, test_precision, test_recall = evaluate_acc_f1(args, model, device, test_data, processor, macro=True, mode='test')
            wandb.log({'test_f1': test_f1, 'test_precision': test_precision,'test_recall': test_recall})
            print(f"i_epoch is {i_epoch}, test_f1 is {test_f1}, test_precision is {test_precision}, test_recall is {test_recall}")

        torch.cuda.empty_cache()
    logger.info('Train done')


def evaluate_acc_f1(args, model, device, data, processor, macro=True, pre = None, mode='test'):
        data_loader = DataLoader(data, batch_size=args.dev_batch_size, collate_fn=MyDataset.collate_func,shuffle=False)
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None

        model.eval()
        sum_loss = 0.
        sum_step = 0
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                text_list, image_list, label_list, id_list = t_batch
                if args.model == 'MV_CLIP':
                    inputs = processor(text=text_list, images=image_list, padding='max_length', truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
                    labels = torch.tensor(label_list).to(device)
                
                t_targets = labels
                loss, t_outputs = model(inputs,labels=labels)
                sum_loss += loss.item()
                sum_step += 1
  
                outputs = torch.argmax(t_outputs, -1)

                n_correct += (outputs == t_targets).sum().item()
                n_total += len(outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, outputs), dim=0)
        if mode == 'test':
            wandb.log({'test_loss': sum_loss/sum_step})
        else:
            wandb.log({'dev_loss': sum_loss/sum_step})
        if pre != None:
            with open(pre,'w',encoding='utf-8') as fout:
                predict = t_outputs_all.cpu().numpy().tolist()
                label = t_targets_all.cpu().numpy().tolist()
                for x,y,z in zip(predict,label):
                    fout.write(str(x) + str(y) +z+ '\n')
        if not macro:
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), average='micro')
            precision =  metrics.precision_score(t_targets_all.cpu(),t_outputs_all.cpu(), average='micro')
            recall = metrics.recall_score(t_targets_all.cpu(),t_outputs_all.cpu(), average='micro')
        else:
            f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), average='macro')
            precision =  metrics.precision_score(t_targets_all.cpu(),t_outputs_all.cpu(), average='macro')
            recall = metrics.recall_score(t_targets_all.cpu(),t_outputs_all.cpu(), average='macro')


        # Tính F1 cho từng class
        class_f1 = metrics.f1_score(t_targets_all.cpu(), t_outputs_all.cpu(), average=None)
        class_names = [str(i) for i in range(len(class_f1))]  # hoặc tên nhãn nếu bạn có

        # In ra console
        print(f"Per-class F1 ({mode}):")
        for i, f1_score in enumerate(class_f1):
            print(f"  Class {class_names[i]}: {f1_score:.4f}")

        # Log vào wandb nếu là test
        if mode == 'test':
            for i, f1_score in enumerate(class_f1):
                wandb.log({f'test_f1_class_{class_names[i]}': f1_score})

        return f1, precision, recall
