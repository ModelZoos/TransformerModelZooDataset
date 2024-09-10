import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import enable_grad

from models.vit_s import ViTSmallPatch16
from models.bert_sst import SentimentClassifier
from models.bert_mlm import MaskedLanguageModel
from utils.vision_augmentations import get_data_augmentations
from timm.loss import SoftTargetCrossEntropy
from utils.info_nce import InfoNCELoss
from utils.print_progress_bar import printProgressBar



import numpy as np

import logging

class NNmodule(nn.Module):
    def __init__(self, config, cuda=False, seed=42, verbosity=0):
        super(NNmodule, self).__init__()
        # make sure mixup does not interfere with older model definitions
        self.use_mixup = False
        # set verbosity
        self.verbosity = verbosity
        if not cuda:
            cuda = True if config.get("device", "cpu") == "cuda" else False
        if cuda and torch.cuda.is_available():
            self.device = "cuda"
            logging.info("cuda availabe:: use cuda")
        else:
            self.device = "cpu"
            self.cuda = False
            logging.info("cuda unavailable:: fallback to cpu")

        # setting seeds for reproducibility
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if config["model::type"] == "vit_s_16":
                logging.info("=> create vit_s_16")
                if config.get("training::mixup", None) is not None:
                    mixup = config["training::mixup"]
                    cutmix = config.get("training::cutmix", 0)
                    rand_erase = config.get("training::random_erase", 0)
                    mixup, rand_erase = get_data_augmentations(config.get("model::o_dim", 100), mixup, cutmix, rand_erase)
                    self.use_mixup = True
                else:
                    mixup = None
                    rand_erase = None
                    self.use_mixup = False
                    
                model = ViTSmallPatch16(
                    num_classes=config.get("model::o_dim", 100),
                    init_type=config.get("model::init_type", "kaiming_normal"),
                    fc_mlp=config.get("model::head::mlp", False),
                    hidden_dim=config.get("model::head::hidden_dim", None),
                    dropout=config.get("model::dropout", 0.),
                    attn_dropout=config.get("model::attn_dropout", 0.),
                    mixup=mixup,
                    random_erase=rand_erase,
                    pretrained_model_path=config.get("pretraining::model::path", None),
                )
                self.language_task = None
        else:
            raise NotImplementedError("error: model type unkown")
        
        logging.info(f"send model to {self.device}")

        model.to(self.device)
        self.model = model

        self.task = config.get("training::task", "classification")

        self.criterion_val = None

        if config.get("training::loss", None) == "stce": # stce = soft target cross entropy
            self.criterion = SoftTargetCrossEntropy()
            # for validation we don't use soft targets
            if config.get("validation::loss", None) == "ce":
                self.criterion_val = nn.CrossEntropyLoss()
        elif config.get("training::loss", None) == "info_nce":
            self.criterion = InfoNCELoss(
                temperature=config.get("training::temperature", 0.07)
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        if self.device == "cuda":
            self.criterion.to(self.device)

        self.set_optimizer(config)
        self.set_scheduler(config)

    # module forward function
    def forward(self, x, target=None, attention_mask=None):
        # compute model prediction
        if self.training and self.use_mixup and target is not None: # mixup requires targets in forward pass
            output, target = self.model(x, target)
            return output, target
        else:
            output = self.model(x)
        return output

    def set_scheduler(self, config):
        if config.get("optim::scheduler", None) == None:
            self.scheduler = None
        elif config.get("optim::scheduler", None) == "OneCycleLR":
            logging.info("use onecycleLR scheduler")
            max_lr = config["optim::lr"]
            try:
                steps_per_epoch = config["scheduler::steps_per_epoch"]
            except KeyError:
                steps_per_epoch = 1
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=max_lr,
                epochs=config["training::epochs_train"],
                steps_per_epoch=steps_per_epoch,
            )

    def set_optimizer(self, config):
        if config["optim::optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config["optim::lr"],
                momentum=config["optim::momentum"],
                weight_decay=config["optim::wd"],
                nesterov=config.get("optim::nesterov", False),
            )
        elif config["optim::optimizer"] == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config["optim::lr"],
                weight_decay=config["optim::wd"],
            )
        else:
            raise NotImplementedError("optimizer not implemented")

    # one training step / batch
    def train_step(self, input, target, attention_mask=None):
        # zero grads before training steps
        self.optimizer.zero_grad()
        target_tmp = None # for mixup we also need the adjusted labels

        if self.use_mixup: # mixup requires targets in forward pass

            output, target_tmp = self.forward(input, target)
        else:
            output = self.forward(input)

        # compute loss
        if target_tmp is not None: # if we have mixup, we need to compute the loss with the adjusted labels
            loss = self.criterion(output, target_tmp)
        elif self.pretraining_mode == "cl": 
            loss, correct = self.criterion(output, target)
        else:
            loss = self.criterion(output, target)
        
        # prop loss backwards to
        loss.backward()
        # update parameters
        self.optimizer.step()
        # scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        # compute correct
        correct = 0
        if self.pretraining_mode == "cl":
            return loss.item(), correct
        elif self.task == "classification":
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
        return loss.item(), correct

    # one training epoch
    @enable_grad()
    def train_epoch(self, trainloader, epoch, idx_out=10):
        logging.info(f"train epoch {epoch}")
        # set model to training mode
        self.model.train()

        if self.verbosity > 2:
            printProgressBar(
                0,
                len(trainloader),
                prefix="Batch Progress:",
                suffix="Complete",
                length=50,
            )
        # init accumulated loss, accuracy
        loss_acc = 0
        correct_acc = 0
        n_data = 0

        # enter loop over batches
        for idx, data in enumerate(trainloader):
            # check if we have a language model
            if self.pretraining_mode == "cl":
                input, target = data

            else:
                input, target = data
                # send to cuda
                input, target = input.to(self.device), target.to(self.device)

            loss, correct = self.train_step(input, target)
            loss_acc += loss * len(target)
            correct_acc += correct
            n_data += len(target)
            
            # logging
            if idx > 0 and idx % idx_out == 0:
                loss_running = loss_acc / n_data
                if self.pretraining_mode == "cl":
                    accuracy = correct
                elif self.task == "classification":
                    accuracy = correct_acc / n_data

                logging.info(
                    f"epoch {epoch} -batch {idx}/{len(trainloader)} --- running ::: loss: {loss_running}; accuracy: {accuracy} "
                )

        self.model.eval()
        # compute epoch running losses
        loss_running = loss_acc / n_data
        if self.task == "classification":
            accuracy = correct_acc / n_data

        return loss_running, accuracy

    # test batch
    def test_step(self, input, target, attention_mask=None):
        correct = 0
        with torch.no_grad():
            output = self.forward(input)
            # compute loss
            if self.criterion_val is not None: # if we have a different loss for validation, e.g. when using mixup for training, we use a different loss for validation
                loss = self.criterion_val(output, target)
            else:
                loss = self.criterion(output, target)

            if self.pretraining_mode == "cl":
                loss, correct = self.criterion(output, target)
                return loss.item(), correct
            elif self.task == "classification":
                # compute correct
                _, predicted = torch.max(output.data, 1)
                correct = (predicted == target).sum().item()
            return loss.item(), correct

    # test epoch
    def test_epoch(self, testloader, epoch):
        logging.info(f"validate at epoch {epoch}")
        # set model to eval mode
        self.model.eval()
        # initilize counters
        loss_acc = 0
        correct_acc = 0
        n_data = 0
        for idx, data in enumerate(testloader):

            if self.pretraining_mode == "cl":
                input, target = data # for contrastive learning, input is a list of augmented images, concat is handled in the model
            else:
                input, target = data
            # send to cuda
            input, target = input.to(self.device), target.to(self.device)
            # perform test step on batch.
            
            loss, correct = self.test_step(input, target)
            # scale loss with batchsize
            loss_acc += loss * len(target)
            correct_acc += correct
            n_data += len(target)
        # logging
        # compute epoch running losses
        loss_running = loss_acc / n_data
        if self.pretraining_mode == "cl":
            accuracy = correct # for contrastive learning, accuracy is computed in the loss function
        elif self.task == "classification":
            accuracy = correct_acc / n_data
        logging.info(f"test ::: loss: {loss_running}; accuracy: {accuracy}")

        return loss_running, accuracy

