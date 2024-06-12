from ray.tune import Trainable

import torch
# print(f"sys path in experiment: {sys.path}")
from pathlib import Path

from experiment_runner.def_net import NNmodule

from transformers import DataCollatorForLanguageModeling, BertTokenizer

import ray
"""
define Tune Trainable
##############################################################################
"""


class NN_tune_trainable(Trainable):
    def setup(self, config):
        
        self.config = config
        self.seed = config.get("seed", 42)
        self.cuda = config["cuda"]
        dataset = ray.get(config["dataset::ref"])
        self.trainset = dataset["trainset"]
        self.testset = dataset["testset"]
        self.valset = dataset.get("valset", None)
        if self.config.get("dataset::name", None) == "openweb":
            tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            self.trainloader = torch.utils.data.DataLoader(
                dataset=self.trainset,
                batch_size=self.config["training::batchsize"],
                shuffle=True,
                num_workers=self.config.get("trainloader::workers", 6),
                collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15),
                drop_last=True
            )
            self.testloader = torch.utils.data.DataLoader(
                dataset=self.testset,
                batch_size=self.config["training::batchsize"],
                shuffle=False,
                num_workers=self.config.get("trainloader::workers", 4),
                collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15),
                drop_last=True
            )
            if self.valset is not None:
                self.valloader = torch.utils.data.DataLoader(
                    dataset=self.valset,
                    batch_size=self.config["training::batchsize"],
                    shuffle=False,
                    num_workers=self.config.get("trainloader::workers", 4),
                    collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15),
                    drop_last=True
                )
        else:
            self.trainloader = torch.utils.data.DataLoader(
                dataset=self.trainset,
                batch_size=self.config["training::batchsize"],
                shuffle=True,
                num_workers=self.config.get("trainloader::workers", 6),
                drop_last=True
            )
            self.testloader = torch.utils.data.DataLoader(
                dataset=self.testset,
                batch_size=self.config["training::batchsize"],
                shuffle=False,
                num_workers=self.config.get("trainloader::workers", 4),
                drop_last=True
            )
            if self.valset is not None:
                self.valloader = torch.utils.data.DataLoader(
                    dataset=self.valset,
                    batch_size=self.config["training::batchsize"],
                    shuffle=False,
                    num_workers=self.config.get("trainloader::workers", 4),
                    drop_last=True
                )

        config["scheduler::steps_per_epoch"] = len(self.trainloader)

        # init model
        self.NN = NNmodule(
            config=self.config, cuda=self.cuda, seed=self.seed, verbosity=0
        )

        # run first test epoch and log results
        self._iteration = -1

    def step(self):
        # here, all manual writers are disabled. tune takes care of that
        # run one training epoch
        if self._iteration < 0:
            print("test first validation mode")
            loss_train, acc_train = self.NN.test_epoch(self.testloader, 0)
            loss_test, acc_test = loss_train, acc_train

        else:
            loss_train, acc_train = self.NN.train_epoch(self.trainloader, 0, idx_out=10)
            # run one test epoch
            loss_test, acc_test = self.NN.test_epoch(self.testloader, 0)

        result_dict = {
            "train_loss": loss_train,
            "train_acc": acc_train,
            "test_loss": loss_test,
            "test_acc": acc_test,
        }
        if self.valset is not None:
            loss_val, acc_val = self.NN.test_epoch(self.valloader, 0)
            result_dict["val_loss"] = loss_val
            result_dict["val_acc"] = acc_val

        return result_dict

    def save_checkpoint(self, tmp_checkpoint_dir):
        # define checkpoint path
        path = Path(tmp_checkpoint_dir).joinpath("checkpoints")
        # save model state dict
        torch.save(self.NN.model.state_dict(), path)
        # save optimizer
        path = Path(tmp_checkpoint_dir).joinpath("optimizer")
        torch.save(self.NN.optimizer.state_dict(), path)

        # tune apparently expects to return the directory
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        # define checkpoint path
        path = Path(tmp_checkpoint_dir).joinpath("checkpoints")
        # save model state dict
        checkpoint = torch.load(path)
        self.NN.model.load_state_dict(checkpoint)
        # load optimizer
        try:
            path = Path(tmp_checkpoint_dir).joinpath("optimizer")
            opt_dict = torch.load(path)
            self.NN.optimizer.load_state_dict(opt_dict)
        except:
            print(f"Could not load optimizer state_dict. (not found at path {path})")
