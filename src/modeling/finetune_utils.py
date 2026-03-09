import os, sys
import time
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from .models.seresnet18 import resnet18
from ..dataloader.dataset import ECGDataset, get_transforms
from .metrics import cal_multilabel_metrics, roc_curves
import pickle
from utils import set_seeds
from torchinfo import summary

class Finetuner(object):
    def __init__(self, args):
        self.args = args

    def setup(self):
        """Initializing the device conditions, datasets, dataloaders,
        model, loss, criterion and optimizer
        """

        # Consider the GPU or CPU condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Device used: ", self.device)
            self.device_count = self.args.device_count
            self.args.logger.info('using {} gpu(s)'.format(self.device_count))
            assert self.args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            self.device = torch.device("cpu")
            print("Device used: ", self.device)
            self.device_count = 1
            self.args.logger.info('using {} cpu'.format(self.device_count))

        # Load the datasets. Use train config since finetuning is a similar routine.
        finetuning_set = ECGDataset(self.args.train_path, get_transforms('train'))
        channels = finetuning_set.channels

        self.finetuning_dl = DataLoader(finetuning_set,
                                   batch_size=self.args.batch_size,
                                   shuffle=True,
                                   num_workers=self.args.num_workers,
                                   pin_memory=(True if self.device == 'cuda' else False),
                                   drop_last=False)
        print("Done data loading")
        if self.args.val_path is not None:
            validation_set = ECGDataset(self.args.val_path, get_transforms('val'))
            self.validation_files = validation_set.data
            self.val_dl = DataLoader(validation_set,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=self.args.num_workers,
                                     pin_memory=(True if self.device == 'cuda' else False),
                                     drop_last=True)

        print("Done validation data loading")
        self.model = resnet18(in_channel=channels,
                              out_channel=len(self.args.labels))
        print("Args labels: ", self.args.labels)

        # Load the source model
        #exclude_keys = ['fc.weight', 'fc.bias']

        # Set seeds since we are creating a new layer with random weights
        set_seeds()

        if hasattr(self.args, 'load_model_path'):
            source_model = torch.load(self.args.load_model_path, map_location=self.device, weights_only=True)
            self.args.logger.info('Loaded the model from: {}'.format(self.args.load_model_path))

            if source_model['fc.weight'].size() != self.model.state_dict()['fc.weight'].size():
                print("Source and target model output class size mismatch. Discarding weight and bias of last layer.")
                for key in source_model.keys():
                    # if key in exclude_keys or key not in self.model.state_dict().keys():
                    #    continue
                    if source_model[key].size() != self.model.state_dict()[key].size():
                        continue
                        # print(key, " size: ", source_model[key].size(dim=0))
                        # self.model.state_dict()[key][:source_model[key].size(dim=0)] = source_model[key]

                    # Use copy_ to get the tensor values. Direct assignment don't work.
                    self.model.state_dict()[key].copy_(source_model[key])

            else:
                self.model.load_state_dict(source_model)
                self.args.logger.info('Loaded the model from: {}'.format(self.args.load_model_path))
            # print(summary(self.model)) # Uncomment to check model summary
        else:
            print("Model not loaded")
            self.args.logger.info('Training a new model from the beginning.')


        # Freeze model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last layers
        """for param in self.model.fc.parameters():
            param.requires_grad = True"""

        # Unfreeze the last two layers
        for name, child in self.model.named_children():
            if "layer4" in name or "fc" in name or "fc1" in name:
                for param in child.parameters():
                    param.requires_grad = True
                    
        # Unfreeze the fc (classifier head) and fc1 (age and gender)
        """for name, child in self.model.named_children():
            if "fc" in name or "fc1" in name:
                for param in child.parameters():
                    param.requires_grad = True"""
        
        # Unfreeze fc (classifier head) only
        """for name, child in self.model.named_children():
            if "fc" in name:
                for param in child.parameters():
                    param.requires_grad = True"""

        summary(self.model) # Uncomment to check model summary after freezing and unfreezing

        # Set seeds since we are creating a new layer with random weights
        set_seeds()

        # If more than 1 CUDA device used, use data parallelism
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Optimizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.args.lr,
                                    weight_decay=self.args.weight_decay)

        
        # Original BCEWithLogitsLoss has no parameter
        self.criterion = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid.to(self.device)
        self.model.to(self.device)
        print("Done setting up finetuning model")
        #for name, parameter in self.model.named_parameters():
        #    print(name)

    def finetune(self):
        """ PyTorch finetuning loop
        """

        self.args.logger.info('finetune() called: model={}, opt={}(lr={}), epochs={}, device={}'.format(
            type(self.model).__name__,
            type(self.optimizer).__name__,
            self.optimizer.param_groups[0]['lr'],
            self.args.epochs,
            self.device))

        # Add all wanted history information
        history = {}
        history['train_csv'] = self.args.train_path
        history['train_loss'] = []
        history['train_micro_auroc'] = []
        history['train_micro_avg_prec'] = []
        history['train_macro_auroc'] = []
        history['train_macro_avg_prec'] = []

        if self.args.val_path is not None:
            history['val_csv'] = self.args.val_path
            history['val_loss'] = []
            history['val_micro_auroc'] = []
            history['val_micro_avg_prec'] = []
            history['val_macro_auroc'] = []
            history['val_macro_avg_prec'] = []

        history['labels'] = self.args.labels
        history['epochs'] = self.args.epochs
        history['batch_size'] = self.args.batch_size
        history['lr'] = self.args.lr
        history['optimizer'] = self.optimizer
        history['criterion'] = self.criterion

        start_time_sec = time.time()
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time_sec))
        print("Training started:", start_time)
        
        ### For early stopping
        patience = 15
        best_val_macro_auc = 0.0
        epochs_without_improvement = 0

        for epoch in range(1, self.args.epochs + 1):
            print("Epoch {}/{}".format(epoch, self.args.epochs))
            # --- TRAIN ON TRAINING SET ------------------------------------------
            self.model.train()
            train_loss = 0.0
            labels_all = torch.tensor((), device=self.device)
            logits_prob_all = torch.tensor((), device=self.device)

            batch_loss = 0.0
            batch_count = 0
            step = 0

            for batch_idx, (ecgs, ag, labels) in enumerate(self.finetuning_dl):
                ecgs = ecgs.to(self.device)  # ECGs
                ag = ag.to(self.device)  # age and gender
                labels = labels.to(self.device)  # diagnoses in SNOMED CT codes

                with torch.set_grad_enabled(True):

                    logits = self.model(ecgs, ag)
                    loss = self.criterion(logits, labels)
                    logits_prob = self.sigmoid(logits)
                    loss_tmp = loss.item() * ecgs.size(0)
                    labels_all = torch.cat((labels_all, labels), 0)
                    logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

                    train_loss += loss_tmp

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # Printing training information
                    if step % 100 == 0:
                        batch_loss += loss_tmp
                        batch_count += ecgs.size(0)
                        batch_loss = batch_loss / batch_count
                        self.args.logger.info('epoch {:^3} [{}/{}] train loss: {:>5.4f}'.format(
                            epoch,
                            batch_idx * len(ecgs),
                            len(self.finetuning_dl.dataset),
                            batch_loss
                        ))

                        batch_loss = 0.0
                        batch_count = 0
                    step += 1

            train_loss = train_loss / len(self.finetuning_dl.dataset)
            train_macro_avg_prec, train_micro_avg_prec, train_macro_auroc, train_micro_auroc = cal_multilabel_metrics(
                labels_all, logits_prob_all, self.args.labels, self.args.threshold)

            self.args.logger.info('epoch {:^4}/{:^4} train loss: {:<6.2f}  train micro auroc: {:<6.2f}  train macro auroc: {:<6.2f}'.format(
                epoch,
                self.args.epochs,
                train_loss,
                train_micro_auroc,
                train_macro_auroc))

            # Add information for training history
            history['train_loss'].append(train_loss)
            history['train_micro_auroc'].append(train_micro_auroc)
            history['train_micro_avg_prec'].append(train_micro_avg_prec)
            history['train_macro_auroc'].append(train_macro_auroc)
            history['train_macro_avg_prec'].append(train_macro_avg_prec)

            # --- EVALUATE ON VALIDATION SET -------------------------------------
            if self.args.val_path is not None:
                self.model.eval()
                val_loss = 0.0
                labels_all = torch.tensor((), device=self.device)
                logits_prob_all = torch.tensor((), device=self.device)

                for ecgs, ag, labels in self.val_dl:
                    ecgs = ecgs.to(self.device)  # ECGs
                    ag = ag.to(self.device)  # age and gender
                    labels = labels.to(self.device)  # diagnoses in SNOMED CT codes

                    with torch.set_grad_enabled(False):
                        logits = self.model(ecgs, ag)
                        loss = self.criterion(logits, labels)
                        logits_prob = self.sigmoid(logits)
                        val_loss += loss.item() * ecgs.size(0)
                        labels_all = torch.cat((labels_all, labels), 0)
                        logits_prob_all = torch.cat((logits_prob_all, logits_prob), 0)

                val_loss = val_loss / len(self.val_dl.dataset)
                val_macro_avg_prec, val_micro_avg_prec, val_macro_auroc, val_micro_auroc = cal_multilabel_metrics(
                    labels_all, logits_prob_all, self.args.labels, self.args.threshold)

                self.args.logger.info('                val loss:  {:<6.2f}   val micro auroc: {:<6.2f}   val macro auroc: {:<6.2f}   '.format(
                    val_loss,
                    val_micro_auroc,
                    val_macro_auroc))

                history['val_loss'].append(val_loss)
                history['val_micro_auroc'].append(val_micro_auroc)
                history['val_micro_avg_prec'].append(val_micro_avg_prec)
                history['val_macro_auroc'].append(val_macro_auroc)
                history['val_macro_avg_prec'].append(val_macro_avg_prec)

                print("Training Loss: %.5f. Validation loss: %.5f. Validation Macro AUC: %.5f." % (train_loss, val_loss, val_macro_auroc))
                
                # Early stopping
                if val_macro_auroc > best_val_macro_auc:
                    best_val_macro_auc = val_macro_auroc
                    model_state_dict = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()

                    # -- Save model each time best validation macro auroc is found
                    model_savepath = os.path.join(self.args.model_save_dir, self.args.yaml_file_name + '_best.pth')
                    torch.save(model_state_dict, model_savepath)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    print("Best val macro AUC: %.5f." % (best_val_macro_auc))
                    print("No improvement: {}".format(epochs_without_improvement))


            # --------------------------------------------------------------------

            # Create ROC Curves for manual checking
            roc_curves(labels_all, logits_prob_all, self.args.labels, epoch, self.args.roc_save_dir)

            # Save a model at every 5th epoch (backup)
            if epoch in list(range(self.args.epochs)[0::5]):
                self.args.logger.info('Saved model at the epoch {}!'.format(epoch))
                # Whether or not you use data parallelism, save the state dictionary this way
                # to have the flexibility to load the model any way you want to any device you want
                model_state_dict = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()

                # -- Save model backup
                model_savepath = os.path.join(self.args.model_save_dir, self.args.yaml_file_name + '_backup.pth')
                torch.save(model_state_dict, model_savepath)
                print("Model backup saved at epoch: ", epoch)
                
            # Save trained model (.pth), history (.pickle) and validation logits (.csv) after the last epoch
            if epoch == self.args.epochs or epochs_without_improvement >= patience:

                self.args.logger.info('Saving the model, training history and validation logits...')

                # Whether or not you use data parallelism, save the state dictionary this way
                # to have the flexibility to load the model any way you want to any device you want
                model_state_dict = self.model.module.state_dict() if self.device_count > 1 else self.model.state_dict()

                # -- Save model
                model_savepath = os.path.join(self.args.model_save_dir,
                                              self.args.yaml_file_name + '.pth')
                torch.save(model_state_dict, model_savepath)

                # -- Save history
                history_savepath = os.path.join(self.args.model_save_dir,
                                                self.args.yaml_file_name + '_train_history.pickle')
                with open(history_savepath, mode='wb') as file:
                    pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)

                # -- Save the logits from validation if used, either save the logits from the training phase
                if self.args.val_path is not None:
                    self.args.logger.info('- Validation logits and labels saved')
                    logits_csv_path = os.path.join(self.args.model_save_dir,
                                                   self.args.yaml_file_name + '_val_logits.csv')
                    labels_all_csv_path = os.path.join(self.args.model_save_dir,
                                                       self.args.yaml_file_name + '_val_labels.csv')
                    # Use filenames as indeces
                    filenames = [os.path.basename(file) for file in self.validation_files]

                else:
                    self.args.logger.info('- Training logits and actual labels saved (no validation set available)')
                    logits_csv_path = os.path.join(self.args.model_save_dir,
                                                   self.args.yaml_file_name + '_train_logits.csv')
                    labels_all_csv_path = os.path.join(self.args.model_save_dir,
                                                       self.args.yaml_file_name + '_train_labels.csv')
                    filenames = None

                # Save logits and corresponding labels
                labels_numpy = labels_all.cpu().detach().numpy().astype(np.float32)
                labels_df = pd.DataFrame(labels_numpy, columns=self.args.labels, index=filenames)
                labels_df.to_csv(labels_all_csv_path, sep=',')

                logits_numpy = logits_prob_all.cpu().detach().numpy().astype(np.float32)
                logits_df = pd.DataFrame(logits_numpy, columns=self.args.labels, index=filenames)
                logits_df.to_csv(logits_csv_path, sep=',')
                
                if epochs_without_improvement >= patience:
                    self.args.logger.info('Patience reached. Training stopped at epoch {}'.format(epoch))
                    print("Best val macro AUC: %.5f." % (best_val_macro_auc))
                    print(f'Early stopping triggered after {epoch} epochs.')
                    break

            # Free up memory
            del logits_prob_all
            del labels_all
            torch.cuda.empty_cache()

        # END OF TRAINING LOOP

        end_time_sec = time.time()
        total_time_sec = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / self.args.epochs
        self.args.logger.info('Time total:     %5.2f sec' % (total_time_sec))
        self.args.logger.info('Time per epoch: %5.2f sec' % (time_per_epoch_sec))
        
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time_sec))
        print("Training finished:", end_time)
        print('Time total:     %5.2f sec' % (total_time_sec))
        print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))