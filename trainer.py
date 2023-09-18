import os
import sys
import time
import torch
import torchmetrics
from datetime import timedelta

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler=None, amp=False, device=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device is not None:
            self.device = device
        self.amp = amp
        self.model = model
        self.model = self.model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.cache = {
            "train_loss": [],
            "train_acc": [],
            "valid_loss": [],
            "valid_acc": [],
            "lr": []
        }
    #

    def load_checkpoint(self, path, load_optimizer=False, load_scheduler=False):
        params = torch.load(path)
        self.model.load_state_dict(params["model"])
        if load_optimizer:
            self.optimizer.load_state_dict(params["optimizer"])
        if self.scheduler is not None and load_scheduler:
            self.scheduler.load_state_dict(params["scheduler"])
        self.cache = params["cache"]
        print("[+] Model load successful")
    #

    def save_checkpoint(self, path):
        params = {
            "model": self.model.state_dict(),
            "cache": self.cache
        }
        torch.save(params, path)
        print("[+] Save checkpoint successfully")
    #

    def _compute_acc(self, inputs, labels, num_classes, ignore_index=-100):
        inputs = inputs.cpu()
        labels = labels.cpu()
        acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.model.n_token_classes)
        return acc(inputs, labels)
    #

    def _match_size_label(self, token_logits, labels):
        labels[0] = labels[0][:, :token_logits.size(1)]
        d1 = token_logits.size(1) - labels[0].size(1)
        if d1:
            labels[0] = torch.cat([labels[0], torch.ones((labels[0].size(0), d1)).to(self.device) * -100], dim=1)
        labels = [i.long() for i in labels]
        return labels

    def compute_acc(self, token_logits, intent_logits, labels):
        labels = self._match_size_label(token_logits, labels)
        token_logits = token_logits.argmax(-1)
        intent_logits = intent_logits.argmax(-1)
        acc = []
        for i in range(token_logits.size(0)):
            tmp_tk_logit = token_logits[i, :]
            # print(tmp_tk_logit, "\n", labels[0][i, :])
            # import sys
            tmp_tk_label = labels[0][i, :].long()
            filter_index = torch.where(tmp_tk_label > 0)[0].tolist()
            tmp_tk_label = tmp_tk_label[filter_index]
            tmp_tk_logit = tmp_tk_logit[filter_index]
            cp_tk = (tmp_tk_logit.long() == tmp_tk_label.long()).float().mean()
            tmp_it_logit = intent_logits[i]
            cp_it = (tmp_it_logit == labels[1][i]).float()
            acc.append(1.0 if (cp_tk + cp_it) == 2.0 else 0.0)
        return torch.tensor(acc, dtype=torch.float).mean()

    def compute_loss(self, token_logits, intent_logits, labels):
        labels = self._match_size_label(token_logits, labels)
        token_loss = self.criterion(
            token_logits.reshape(-1, self.model.n_token_classes), labels[0].reshape(-1)
        )
        intent_loss = self.criterion(
            intent_logits, labels[1].reshape(-1)
        )
        all_loss = (token_loss + intent_loss) / 2
        return token_loss, intent_loss, all_loss
    #

    def forward(self, dataloader, fw_mode="train"):
        if fw_mode == "train" :
            self.model.train()
        else:
            self.model.eval()

        loss_his = []
        acc_his = []
        N = len(dataloader)
        for idx, (batch) in enumerate(dataloader, 1):
            if fw_mode == "train":
                self.optimizer.zero_grad()
            X_batch = {k : v.to(self.device) for k, v in batch[0].items()}
            token_labels = batch[1].to(self.device)
            intent_labels = batch[2].to(self.device)
            with torch.set_grad_enabled(fw_mode=="train"):
                with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                    token_logits, intent_logits = self.model(X_batch)
                    token_loss, intent_loss, all_loss = self.compute_loss(token_logits, intent_logits, [token_labels, intent_labels])
                    acc = self.compute_acc(token_logits, intent_logits, [token_labels, intent_labels])
                    # mean_acc = (acc["intent_acc"][-1] + acc["token_acc"][-1]) / 2.0
            if fw_mode == "train":
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(all_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
            loss_his.append(all_loss.item())
            acc_his.append(acc.item())
            print("\r", end="")
            print(f"{fw_mode.capitalize()} step: {idx} / {N} - loss: {all_loss.item():.5f} - acc: {acc.item():.4f}", end="" if idx != N else "\n")

        loss = sum(loss_his) / N
        acc = sum(acc_his) / N
        self.cache[f"{fw_mode}_loss"].append(loss)
        self.cache[f"{fw_mode}_acc"].append(acc)
    #

    def fit(self, train_loader, valid_loader=None, epochs=10, checkpoint="./checkpoint.pt"):
        print(f"Running on: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Total update step: {len(train_loader) * epochs}")

        for epoch in range(1, epochs+1):
            start_time = time.time()
            print(f"Epoch: {epoch}")
            logs = []
            current_lr = f": {self.optimizer.param_groups[0]['lr']:e}"
            try:
                self.forward(train_loader, "train")
                train_loss = self.cache["train_loss"][-1]
                train_acc = self.cache["train_acc"][-1]
                logs.append(f"\t => Train loss: {train_loss:.5f} - Train acc: {train_acc:.2f}")
            except KeyboardInterrupt:
                sys.exit()
            if valid_loader is not None:
                try:
                    self.forward(valid_loader, "valid")
                    valid_loss = self.cache["valid_loss"][-1]
                    valid_acc = self.cache["valid_acc"][-1]
                    logs.append(f"\t => Valid loss: {valid_loss:.5f} - Valid acc: {valid_acc:.2f}")
                except KeyboardInterrupt:
                    sys.exit()
            total_time = round(time.time() - start_time, 1)
            logs.append(f"\t => Time: {timedelta(seconds=int(total_time))}/step - lr: {current_lr}")
            # if len(logs) == 3:
                # logs = [f"\t=> {logs[0]} - {logs[1]}", logs[2]]
            print("\n".join(logs))
            self.cache["lr"].append(current_lr)
            self.save_checkpoint(checkpoint)
    #

    def test(self, test_loader):
        self.model.eval()
        all_tokens = []
        all_intents = []
        for idx, batch in enumerate(test_loader, 1):
            X_batch = {k: v.to(self.device) for k, v in batch[0].items()}
            token_logits, intent_logits = self.model(X_batch)
            token_logits = token_logits.argmax(-1).cpu().tolist()
            intent_logits = intent_logits.argmax(-1).view(-1).cpu().tolist()
            all_tokens += token_logits
            all_intents += intent_logits
            print("\r", end="")
            print(f"\r {idx} / {len(test_loader)}", end = "" if idx != len(test_loader) else "\n")
        return all_tokens, all_intents
    
class TrainerV2:
    def __init__(self, model, optimizer, criterion, scheduler=None, amp=False, device=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.amp = amp
        self.model = model
        self.model = self.model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.cache = {
            "train_loss": [],
            "train_acc": [],
            "valid_loss": [],
            "valid_acc": [],
            "lr": []
        }
    #

    def load_checkpoint(self, path, load_optimizer=False, load_scheduler=False):
        params = torch.load(path)
        self.model.load_state_dict(params["model"])
        if load_optimizer:
            self.optimizer.load_state_dict(params["optimizer"])
        if self.scheduler is not None and load_scheduler:
            self.scheduler.load_state_dict(params["scheduler"])
        self.cache = params["cache"]
        print("[+] Model load successful")
    #

    def save_checkpoint(self, path):
        params = {
            "model": self.model.state_dict(),
            "cache": self.cache
        }
        torch.save(params, path)
        print("[+] Save checkpoint successfully")
    #

    def compute_acc(self, token_logits, intent_logits, labels):
        token_logits = token_logits.argmax(-1) if token_logits is not None else None
        intent_logits = intent_logits.argmax(-1) if intent_logits is not None else None
        acc = []
        N = token_logits.size(0) if token_logits is not None else intent_logits.size(0)
        for i in range(N):
            cp_tk = 0.0
            cp_it = 0.0
            if token_logits is not None:
                tmp_tk_logit = token_logits[i, :]
                tmp_tk_label = labels[0][i, :].long()
                filter_index = torch.where(tmp_tk_label > 0)[0].tolist()
                tmp_tk_label = tmp_tk_label[filter_index]
                tmp_tk_logit = tmp_tk_logit[filter_index]
                cp_tk = (tmp_tk_logit.long() == tmp_tk_label.long()).float().mean()
            if intent_logits is not None:
                tmp_it_logit = intent_logits[i]
                cp_it = (tmp_it_logit == labels[1][i]).float().item()
            acc.append([1.0 if cp_tk == 1.0 else 0.0, cp_it])
        return torch.tensor(acc, dtype=torch.float).mean(dim=0).sum()

    def compute_loss(self, token_logits, intent_logits, labels):
        token_loss = self.criterion(
            token_logits.reshape(-1, self.model.n_token_classes), labels[0].reshape(-1)
        ) if token_logits is not None else None
        intent_loss = self.criterion(
            intent_logits, labels[1].reshape(-1)
        ) if intent_logits is not None else None
        return token_loss if token_loss is not None else intent_loss
    #

    def forward(self, dataloader, fw_mode="train"):
        if fw_mode == "train" :
            self.model.train()
        else:
            self.model.eval()

        loss_his = []
        acc_his = []
        N = len(dataloader)
        for idx, (batch) in enumerate(dataloader, 1):
            if fw_mode == "train":
                self.optimizer.zero_grad()
            X_batch = {k : v.to(self.device) for k, v in batch[0].items()}
            token_labels = batch[1].to(self.device)
            intent_labels = batch[2].to(self.device)
            with torch.set_grad_enabled(fw_mode=="train"):
                with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                    token_logits, intent_logits = self.model(X_batch)
                    loss = self.compute_loss(token_logits, intent_logits, [token_labels, intent_labels])
                    acc = self.compute_acc(token_logits, intent_logits, [token_labels, intent_labels])
            if fw_mode == "train":
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
            loss_his.append(loss.item())
            acc_his.append(acc.item())
            print("\r", end="")
            print(f"{fw_mode.capitalize()} step: {idx} / {N} - loss: {loss.item():.5f} - acc: {acc.item():.4f}", end="" if idx != N else "\n")

        loss = sum(loss_his) / N
        acc = sum(acc_his) / N
        self.cache[f"{fw_mode}_loss"].append(loss)
        self.cache[f"{fw_mode}_acc"].append(acc)
    #

    def fit(self, train_loader, valid_loader=None, epochs=10, checkpoint="./checkpoint.pt"):
        print(f"Running on: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Total update step: {len(train_loader) * epochs}")

        for epoch in range(1, epochs+1):
            start_time = time.time()
            print(f"Epoch: {epoch}")
            logs = []
            current_lr = f": {self.optimizer.param_groups[0]['lr']:e}"
            try:
                self.forward(train_loader, "train")
                train_loss = self.cache["train_loss"][-1]
                train_acc = self.cache["train_acc"][-1]
                logs.append(f"\t => Train loss: {train_loss:.5f} - Train acc: {train_acc:.2f}")
            except KeyboardInterrupt:
                sys.exit()
            if valid_loader is not None:
                try:
                    self.forward(valid_loader, "valid")
                    valid_loss = self.cache["valid_loss"][-1]
                    valid_acc = self.cache["valid_acc"][-1]
                    logs.append(f"\t => Valid loss: {valid_loss:.5f} - Valid acc: {valid_acc:.2f}")
                except KeyboardInterrupt:
                    sys.exit()
            total_time = round(time.time() - start_time, 1)
            logs.append(f"\t => Time: {timedelta(seconds=int(total_time))}/step - lr: {current_lr}")
            # if len(logs) == 3:
                # logs = [f"\t=> {logs[0]} - {logs[1]}", logs[2]]
            print("\n".join(logs))
            self.cache["lr"].append(current_lr)
            self.save_checkpoint(checkpoint)
    #

    def test(self, test_loader):
        self.model.eval()
        all_tokens = []
        all_intents = []
        for idx, batch in enumerate(test_loader, 1):
            X_batch = {k: v.to(self.device) for k, v in batch[0].items()}
            token_logits, intent_logits = self.model(X_batch)
            token_logits = token_logits.argmax(-1).cpu().tolist() if token_logits is not None else [None]
            intent_logits = intent_logits.argmax(-1).view(-1).cpu().tolist() if intent_logits is not None else [None]
            all_tokens += token_logits
            all_intents += intent_logits
            print("\r", end="")
            print(f"\r {idx} / {len(test_loader)}", end = "" if idx != len(test_loader) else "\n")
        return all_tokens, all_intents