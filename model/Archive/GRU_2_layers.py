import numpy as np
import os

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch_geometric.nn import conv
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy import sparse

# Get the paths of the repository
file_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(file_path))


class GRU(pl.LightningModule):
    def __init__(self, args):
        super(GRU, self).__init__()
        self.args = args

        self.save_hyperparameters()

        self.encoder_gru = EncoderGRU(self.args)

        self.decoder_residual = DecoderResidual(self.args)

        self.reg_loss = nn.SmoothL1Loss(reduction="none")

        self.is_frozen = False

    @staticmethod
    def init_args(parent_parser):
        parser_dataset = parent_parser.add_argument_group("dataset")
        parser_dataset.add_argument(
            "--train_split", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "train", "data"))
        parser_dataset.add_argument(
            "--val_split", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "val", "data"))
        parser_dataset.add_argument(
            "--test_split", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "test_obs", "data"))
        parser_dataset.add_argument(
            "--train_split_pre", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "train_pre.pkl"))
        parser_dataset.add_argument(
            "--val_split_pre", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "val_pre.pkl"))
        parser_dataset.add_argument(
            "--test_split_pre", type=str, default=os.path.join(
                root_path, "dataset", "argoverse", "test_pre.pkl"))
        parser_dataset.add_argument(
            "--reduce_dataset_size", type=int, default=0)
        parser_dataset.add_argument(
            "--use_preprocessed", type=bool, default=True)
        parser_dataset.add_argument(
            "--align_image_with_target_x", type=bool, default=True)

        parser_training = parent_parser.add_argument_group("training")
        parser_training.add_argument("--num_epochs", type=int, default=72)
        parser_training.add_argument(
            "--lr_values", type=list, default=[1e-3, 1e-4, 1e-3, 1e-4])
        parser_training.add_argument(
            "--lr_step_epochs", type=list, default=[32, 36, 68])
        parser_training.add_argument("--wd", type=float, default=0.01)
        parser_training.add_argument("--batch_size", type=int, default=32)
        parser_training.add_argument("--val_batch_size", type=int, default=32)
        parser_training.add_argument("--workers", type=int, default=0)
        parser_training.add_argument("--val_workers", type=int, default=0)
        parser_training.add_argument("--gpus", type=int, default=1)

        parser_model = parent_parser.add_argument_group("model")
        parser_model.add_argument("--latent_size", type=int, default=128)  # hidden size
        parser_model.add_argument("--num_preds", type=int, default=30)
        parser_model.add_argument("--mod_steps", type=list, default=[1, 5])
        parser_model.add_argument("--mod_freeze_epoch", type=int, default=36)

        return parent_parser

    def forward(self, batch):
        # Set batch norm to eval mode in order to prevent updates on the running means,
        # if the weights are frozen
        if self.is_frozen:
            for module in self.modules():
                if isinstance(module, torch.nn.modules.BatchNorm1d):
                    module.eval()

        displ, centers = batch["displ"], batch["centers"]
        rotation, origin = batch["rotation"], batch["origin"]

        # Configure for single vehicle
        ###################################################################################
        # Number of agents is 1 only, as only agent is extracted
        agents_per_sample = [1 for x in displ]

        # Extract only the agent car displacements
        # Convert the list of tensors to tensors
        displ_cat = torch.stack([vehicles[0] for vehicles in displ])

        # Encoder Layer
        ##################################################################################
        out_encoder_gru = self.encoder_gru(displ_cat, agents_per_sample)

        # Decoder Layer
        ###################################################################################
        out_linear = self.decoder_residual(out_encoder_gru, self.is_frozen)

        # Postprocessing
        ###################################################################################
        out = out_linear.view(len(displ), 1, -1, self.args.num_preds, 2)  # Rearrange into (B,1,Heads,30,2)

        # Iterate over each batch and transform predictions into the global coordinate frame
        for i in range(len(out)):
            out[i] = torch.matmul(out[i], rotation[i]) + origin[i].view(
                1, 1, 1, -1
            )
        return out

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        self.decoder_residual.unfreeze_layers()

        self.is_frozen = True

    def prediction_loss(self, preds, gts):
        # Stack all the predicted trajectories of the target agent
        num_mods = preds.shape[2]
        # [0] is required to remove the unneeded dimensions
        preds = torch.cat([x[0] for x in preds], 0)

        # Stack all the true trajectories of the target agent
        # Keep in mind, that there are multiple trajectories in each sample, but only the first one ([0]) corresponds
        # to the target agent
        gt_target = torch.cat([torch.unsqueeze(x[0], 0) for x in gts], 0)
        # Creating ground truth for every prediction in order to calculate loss
        gt_target = torch.repeat_interleave(gt_target, num_mods, dim=0)

        loss_single = self.reg_loss(preds, gt_target)
        # Sum the loss across x and y, then sum across the whole path
        loss_single = torch.sum(torch.sum(loss_single, dim=2), dim=1)

        loss_single = torch.split(loss_single, num_mods)

        # Tuple to tensor
        loss_single = torch.stack(list(loss_single), dim=0)

        min_loss_index = torch.argmin(loss_single, dim=1)

        # Only calculate loss for the nearest path, let the path have more possibilities (one may go straight,
        # one may turn)
        min_loss_combined = [x[min_loss_index[i]]
                             for i, x in enumerate(loss_single)]

        loss_out = torch.sum(torch.stack(min_loss_combined))

        return loss_out

    def configure_optimizers(self):
        if self.current_epoch == self.args.mod_freeze_epoch:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), weight_decay=self.args.wd)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), weight_decay=self.args.wd)
        return optimizer

    def on_train_epoch_start(self):
        # Trigger weight freeze and optimizer reinit on mod_freeze_epoch
        if self.current_epoch == self.args.mod_freeze_epoch:
            self.freeze()
            self.trainer.accelerator.setup_optimizers(self.trainer)

        # Set learning rate according to current epoch
        for single_param in self.optimizers().param_groups:
            single_param["lr"] = self.get_lr(self.current_epoch)

    def training_step(self, train_batch, batch_idx):
        out = self.forward(train_batch)
        loss = self.prediction_loss(out, train_batch["gt"])
        self.log("loss_train", loss / len(out))
        return loss

    def get_lr(self, epoch):
        lr_index = 0
        for lr_epoch in self.args.lr_step_epochs:
            if epoch < lr_epoch:
                break
            lr_index += 1
        return self.args.lr_values[lr_index]

    def validation_step(self, val_batch, batch_idx):
        out = self.forward(val_batch)
        loss = self.prediction_loss(out, val_batch["gt"])
        self.log("loss_val", loss / len(out))

        # Extract target agent only
        pred = [x[0].detach().cpu().numpy() for x in out]
        gt = [x[0].detach().cpu().numpy() for x in val_batch["gt"]]
        return pred, gt

    def validation_epoch_end(self, validation_outputs):
        # Extract predictions
        pred = [out[0] for out in validation_outputs]
        pred = np.concatenate(pred, 0)
        gt = [out[1] for out in validation_outputs]
        gt = np.concatenate(gt, 0)
        ade1, fde1, ade, fde = self.calc_prediction_metrics(pred, gt)
        self.log("ade1_val", ade1, prog_bar=True)
        self.log("fde1_val", fde1, prog_bar=True)
        self.log("ade_val", ade, prog_bar=True)
        self.log("fde_val", fde, prog_bar=True)

    def calc_prediction_metrics(self, preds, gts):
        # Calculate prediction error for each mode
        # Output has shape (batch_size, n_modes, n_timesteps)
        error_per_t = np.linalg.norm(preds - np.expand_dims(gts, axis=1), axis=-1)

        # Calculate the error for the first mode (at index 0)
        fde_1 = np.average(error_per_t[:, 0, -1])
        ade_1 = np.average(error_per_t[:, 0, :])

        # Calculate the error for all modes
        # Best mode is always the one with the lowest final displacement
        lowest_final_error_indices = np.argmin(error_per_t[:, :, -1], axis=1)
        error_per_t = error_per_t[np.arange(
            preds.shape[0]), lowest_final_error_indices]
        fde = np.average(error_per_t[:, -1])
        ade = np.average(error_per_t[:, :])
        return ade_1, fde_1, ade, fde


class EncoderGRU(nn.Module):
    def __init__(self, args):
        super(EncoderGRU, self).__init__()
        self.args = args

        self.input_size = 3
        self.hidden_size = args.latent_size
        self.num_layers = 2

        self.GRU = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,  # Batch comes first shape = (B, Data)
        )

    def forward(self, rnn_in, agents_per_sample):
        # lstm_in are all agents over all samples in the current batch
        # Format for LSTM has to be has to be (batch_size, timeseries_length, latent_size), because batch_first=True

        # Initialize the hidden state.
        # lstm_in.shape[0] corresponds to the number of all agents in the current batch
        rnn_hidden_state = torch.randn(
            self.num_layers, rnn_in.shape[0], self.hidden_size, device=rnn_in.device)

        rnn_out, rnn_hidden = self.GRU(rnn_in, rnn_hidden_state)

        # lstm_out is the hidden state over all time steps from the last LSTM layer
        # In this case, only the features of the last time step are used
        return rnn_out[:, -1, :]


class DecoderResidual(nn.Module):
    def __init__(self, args):
        super(DecoderResidual, self).__init__()

        self.args = args

        output = []
        for i in range(sum(args.mod_steps)):
            output.append(PredictionNet(args))

        self.output = nn.ModuleList(output)

    def forward(self, decoder_in, is_frozen):
        sample_wise_out = []

        if self.training is False:
            for out_subnet in self.output:
                sample_wise_out.append(out_subnet(decoder_in))
        elif is_frozen:
            for i in range(self.args.mod_steps[0], sum(self.args.mod_steps)):
                sample_wise_out.append(self.output[i](decoder_in))
        else:
            sample_wise_out.append(self.output[0](decoder_in))

        decoder_out = torch.stack(sample_wise_out)  # convert list to tensor only
        decoder_out = torch.swapaxes(decoder_out, 0, 1)

        return decoder_out

    def unfreeze_layers(self):
        for layer in range(self.args.mod_steps[0], sum(self.args.mod_steps)):
            for param in self.output[layer].parameters():
                param.requires_grad = True


class PredictionNet(nn.Module):
    def __init__(self, args):
        super(PredictionNet, self).__init__()

        self.args = args

        self.latent_size = args.latent_size

        self.weight1 = nn.Linear(self.latent_size, self.latent_size)
        self.norm1 = nn.GroupNorm(1, self.latent_size)

        self.weight2 = nn.Linear(self.latent_size, self.latent_size)
        self.norm2 = nn.GroupNorm(1, self.latent_size)

        self.output_fc = nn.Linear(self.latent_size, args.num_preds * 2)

    def forward(self, prednet_in):
        # Residual layer
        x = self.weight1(prednet_in)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.weight2(x)
        x = self.norm2(x)

        x += prednet_in

        x = F.relu(x)

        # Last layer has no activation function
        prednet_out = self.output_fc(x)

        return prednet_out
