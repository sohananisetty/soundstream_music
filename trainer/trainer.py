from math import sqrt
import copy
from random import choice
from pathlib import Path
from shutil import rmtree
import os
from beartype.typing import Union, List, Optional, Tuple
from typing_extensions import Annotated

from beartype import beartype
from beartype.door import is_bearable
from beartype.vale import Is

import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

from einops import rearrange

from core.optimizer import get_optimizer

from ema_pytorch import EMA

from core.soundstream import SoundStream

from datasets.data import SoundDataset, get_dataloader
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate import DistributedType
import wandb

import transformers

# constants

DEFAULT_SAMPLE_RATE = 24000

# for automatically routing data emitted from a dataset to keywords of the transformer wrappers

DATASET_FIELD_TYPE_CONFIG = dict(
	raw_wave = Annotated[
		torch.Tensor,
		Is[lambda t: t.dtype == torch.float and t.ndim in {2, 3}]
	],
	text = List[str],
	text_embeds = Annotated[
		torch.Tensor,
		Is[lambda t: t.dtype == torch.float and t.ndim == 3]
	],
)

# helpers

def exists(val):
	return val is not None

def noop(*args, **kwargs):
	pass

def cycle(dl):
	while True:
		for data in dl:
			yield data

def cast_tuple(t):
	return t if isinstance(t, (tuple, list)) else (t,)

def yes_or_no(question):
	answer = input(f'{question} (y/n) ')
	return answer.lower() in ('yes', 'y')

def accum_log(log, new_logs):
	for key, new_value in new_logs.items():
		old_value = log.get(key, 0.)
		log[key] = old_value + new_value
	return log

# auto data to module keyword argument routing functions

def has_duplicates(tup):
	counts = dict()
	for el in tup:
		if el not in counts:
			counts[el] = 0
		counts[el] += 1
	return any(filter(lambda count: count > 1, counts.values()))

def determine_types(data, config):
	output = []
	for el in data:
		for name, data_type in config.items():
			if is_bearable(el, data_type):
				output.append(name)
				break
		else:
			raise TypeError(f'unable to determine type of {data}')

	return tuple(output)

# main trainer class

class SoundStreamTrainer(nn.Module):
	def __init__(
		self,
		soundstream: SoundStream,
		*,
		num_train_steps,
		batch_size,
		data_max_length = None,
		data_max_length_seconds = None,
		folder,
		lr = 2e-4,
		grad_accum_every = 4,
		wd = 0.,
		max_grad_norm = 0.5,
		discr_max_grad_norm = None,
		save_results_every = 100,
		save_model_every = 1000,
		wandb_every = 50,
		log_losses_every = 1,
		results_folder = './results',
		valid_frac = 0.01,
		random_split_seed = 42,
		use_ema = False,
		ema_beta = 0.995,
		ema_update_after_step = 500,
		ema_update_every = 10,
		apply_grad_penalty_every = 4,
		dl_num_workers = 0,
		accelerate_kwargs: dict = dict(),
		force_clear_prev_results = False  # set to True | False to skip the prompt
	):
		super().__init__()

		kwargs = DistributedDataParallelKwargs(find_unused_parameters = True)
		self.accelerator = Accelerator(kwargs_handlers = [kwargs], **accelerate_kwargs)

		transformers.set_seed(42)

		if self.is_main:
			wandb.login()
			wandb.init(project="soundstream_music_no_mhse")
			
		self.soundstream = soundstream

		self.use_ema = use_ema
		if self.use_ema:
			self.ema_soundstream = EMA(soundstream, beta = ema_beta, update_after_step = ema_update_after_step, update_every = ema_update_every)

		self.register_buffer('steps', torch.Tensor([0]))

		self.num_train_steps = num_train_steps
		self.batch_size = batch_size
		self.grad_accum_every = grad_accum_every

		# optimizers

		self.optim = get_optimizer(soundstream.non_discr_parameters(), lr = lr, wd = wd)

		for discr_optimizer_key, discr in self.multiscale_discriminator_iter():
			one_multiscale_discr_optimizer = get_optimizer(discr.parameters(), lr = lr, wd = wd)
			setattr(self, discr_optimizer_key, one_multiscale_discr_optimizer)

		self.discr_optim = get_optimizer(soundstream.stft_discriminator.parameters(), lr = lr, wd = wd)

		# max grad norm

		self.max_grad_norm = max_grad_norm
		self.discr_max_grad_norm = discr_max_grad_norm

		# create dataset

		assert not (exists(data_max_length) and exists(data_max_length_seconds))

		if exists(data_max_length_seconds):
			data_max_length = data_max_length_seconds * soundstream.target_sample_hz

		self.ds = SoundDataset(
			folder,
			max_length = data_max_length,
			target_sample_hz = soundstream.target_sample_hz,
			seq_len_multiple_of = soundstream.seq_len_multiple_of
		)

		# split for validation

		if valid_frac > 0:
			train_size = int((1 - valid_frac) * len(self.ds))
			valid_size = len(self.ds) - train_size
			self.ds, self.valid_ds = random_split(self.ds, [train_size, valid_size], generator = torch.Generator().manual_seed(random_split_seed))
			self.print(f'training with dataset of {len(self.ds)} samples and validating with randomly splitted {len(self.valid_ds)} samples')
		else:
			self.valid_ds = self.ds
			self.print(f'training with shared training and valid dataset of {len(self.ds)} samples')

		# dataloader

		self.dl = get_dataloader(self.ds, batch_size = batch_size, num_workers = dl_num_workers, shuffle = True)

		self.valid_dl = get_dataloader(self.valid_ds, batch_size = batch_size, num_workers = dl_num_workers, shuffle = True)

		# prepare with accelerator

		(
			self.soundstream,
			self.optim,
			self.discr_optim,
			self.dl,
			self.valid_dl
		) = self.accelerator.prepare(
			self.soundstream,
			self.optim,
			self.discr_optim,
			self.dl,
			self.valid_dl
		)

		# prepare the multiscale discriminators with accelerator


		for name, _ in self.multiscale_discriminator_iter():
			optimizer = getattr(self, name)
			optimizer = self.accelerator.prepare(optimizer)
			setattr(self, name, optimizer)

		# dataloader iterators

		self.dl_iter = cycle(self.dl)
		self.valid_dl_iter = cycle(self.valid_dl)

		self.save_model_every = save_model_every
		self.save_results_every = save_results_every
		self.log_losses_every = log_losses_every
		self.wandb_every = wandb_every

		self.apply_grad_penalty_every = apply_grad_penalty_every

		self.results_folder = Path(results_folder)

		if self.is_main and force_clear_prev_results is True or (not exists(force_clear_prev_results) and len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?')):
			rmtree(str(self.results_folder))

		self.results_folder.mkdir(parents = True, exist_ok = True)
		self.best_loss = float("inf")

		hps = {"num_train_steps": num_train_steps, "data_max_length": data_max_length, "learning_rate": lr}
		self.accelerator.init_trackers("soundstream", config=hps)        

	def set_model_as_ema_model_(self):
		""" this will force the main 'online' model to have same parameters as the exponentially moving averaged model """
		assert self.use_ema
		self.ema_soundstream.ema_model.load_state_dict(self.soundstream.state_dict())




	def save(self, path):
		pkg = dict(
			model = self.accelerator.get_state_dict(self.soundstream),
			optim = self.optim.state_dict(),
			discr_optim = self.discr_optim.state_dict(),
			steps = self.steps
		)

		if self.use_ema:
			pkg['ema_model'] = self.ema_soundstream.state_dict()

		for key, _ in self.multiscale_discriminator_iter():
			discr_optim = getattr(self, key)
			pkg[key] = discr_optim.state_dict()

		torch.save(pkg, path)

	@property
	def unwrapped_soundstream(self):
		return self.accelerator.unwrap_model(self.soundstream)

	def load(self, path):
		path = Path(path)
		assert path.exists()
		pkg = torch.load(str(path), map_location = 'cpu')

		# if loading from old version, make a hacky guess

		if len(pkg.keys()) > 20:
			self.unwrapped_soundstream.load_state_dict(pkg)

			if self.use_ema:
				self.ema_soundstream.ema_model.load_state_dict(pkg)
			return

		# otherwise load things normally

		self.unwrapped_soundstream.load_state_dict(pkg['model'] , strict = False)

		if self.use_ema:
			assert 'ema_model' in pkg
			self.ema_soundstream.load_state_dict(pkg['ema_model'])

		self.optim.load_state_dict(pkg['optim'])
		self.discr_optim.load_state_dict(pkg['discr_optim'])

		for key, _ in self.multiscale_discriminator_iter():
			discr_optim = getattr(self, key)
			discr_optim.load_state_dict(pkg[key])

		
		try:
			self.steps = pkg["steps"]
			print("starting at step: ", self.steps)
		except:
			pass

	def multiscale_discriminator_iter(self):
		for ind, discr in enumerate(self.unwrapped_soundstream.discriminators):
			yield f'multiscale_discr_optimizer_{ind}', discr

	def multiscale_discriminator_optim_iter(self):
		for name, _ in self.multiscale_discriminator_iter():
			yield name, getattr(self, name)

	def print(self, msg):
		self.accelerator.print(msg)

	@property
	def device(self):
		return self.accelerator.device

	@property
	def is_distributed(self):
		return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

	@property
	def is_main(self):
		return self.accelerator.is_main_process

	@property
	def is_local_main(self):
		return self.accelerator.is_local_main_process

	def train_step(self):
		# device = self.device

		steps = int(self.steps.item())
		apply_grad_penalty = self.apply_grad_penalty_every > 0 and not (steps % self.apply_grad_penalty_every)
		log_losses = self.log_losses_every > 0 and not (steps % self.log_losses_every)

		self.soundstream.train()

		# logs

		logs = {}

		# update vae (generator)

		for _ in range(self.grad_accum_every):
			wave, = next(self.dl_iter)
			# wave = wave.to(device)

			loss, (recon_loss, multi_spectral_recon_loss, adversarial_loss, feature_loss, all_commitment_loss) = self.soundstream(wave, return_loss_breakdown = True)

			self.accelerator.backward(loss / self.grad_accum_every)

			accum_log(logs, dict(
				loss = loss.item() / self.grad_accum_every,
				recon_loss = recon_loss.item() / self.grad_accum_every,
			))

			if log_losses:
				accum_log(logs, dict(
					multi_spectral_recon_loss = multi_spectral_recon_loss.item() / self.grad_accum_every,
					adversarial_loss = adversarial_loss.item() / self.grad_accum_every,
					feature_loss = feature_loss.item() / self.grad_accum_every,
					all_commitment_loss = all_commitment_loss.item() / self.grad_accum_every,
				))

		if exists(self.max_grad_norm):
			self.accelerator.clip_grad_norm_(self.soundstream.parameters(), self.max_grad_norm)

		self.optim.step()
		self.optim.zero_grad()

		# update discriminator

		self.discr_optim.zero_grad()

		for name, multiscale_discr_optim in self.multiscale_discriminator_optim_iter():
			multiscale_discr_optim.zero_grad()

		for _ in range(self.grad_accum_every):
			wave, = next(self.dl_iter)

			# print(wave.shape)
			# wave = wave.to(device)

			discr_losses = self.soundstream(
				wave,
				apply_grad_penalty = apply_grad_penalty,
				return_discr_loss = True,
				return_discr_losses_separately = True
			)

			for name, discr_loss in discr_losses:
				self.accelerator.backward(discr_loss / self.grad_accum_every, retain_graph = True)
				accum_log(logs, {name: discr_loss.item() / self.grad_accum_every})

		if exists(self.discr_max_grad_norm):
			self.accelerator.clip_grad_norm_(self.soundstream.stft_discriminator.parameters(), self.discr_max_grad_norm)

		# gradient step for all discriminators

		self.discr_optim.step()

		for name, multiscale_discr_optim in self.multiscale_discriminator_optim_iter():
			multiscale_discr_optim.step()

		# build pretty printed losses

		losses_str = f"{steps}: soundstream total loss: {logs['loss']:.3f}, soundstream recon loss: {logs['recon_loss']:.3f}"
		if log_losses:
			self.accelerator.log({
				"total_loss": logs['loss'],
				"recon_loss": logs['recon_loss'],
				"multi_spectral_recon_loss": logs['multi_spectral_recon_loss'],
				"adversarial_loss": logs['adversarial_loss'],
				"feature_loss": logs['feature_loss'],
				"all_commitment_loss": logs['all_commitment_loss'],
				"stft_discr_loss": logs['stft']
			}, step=steps)

		for key, loss in logs.items():
			if not key.startswith('scale:'):
				continue
			_, scale_factor = key.split(':')

			losses_str += f" | discr (scale {scale_factor}) loss: {loss:.3f}"
			if log_losses:
				self.accelerator.log({f"discr_loss (scale {scale_factor})": loss}, step=steps)

		# log
		if self.is_main and (steps%self.wandb_every == 0):
			for key , value in logs.items():
				wandb.log({f'train_loss/{key}': value})           

		self.print(losses_str)

		# print("ema")

		# update exponential moving averaged generator

		# self.accelerator.wait_for_everyone()
		with self.accelerator.main_process_first():
			if self.use_ema:
				self.ema_soundstream.update()

		# sample results every so often

		# print("evaluation")

		# self.accelerator.wait_for_everyone()

		if self.is_main and (steps % self.save_results_every == 0):
			models = [(self.unwrapped_soundstream, str(steps))]
			if self.use_ema:
				models.append((self.ema_soundstream.ema_model if self.use_ema else self.unwrapped_soundstream, f'{steps}.ema'))

			wave, = next(self.valid_dl_iter)
			wave = wave

			for model, label in models:
				model.eval()

				with torch.no_grad():
					recons = model(wave, return_recons_only = True)

				for ind, recon in enumerate(recons.unbind(dim = 0)):
					os.makedirs(os.path.join(self.results_folder , "samples" ) , exist_ok=True)
					filename = (os.path.join(self.results_folder , "samples" , f'sample_{label}.flac'))
					torchaudio.save(filename, recon.cpu().detach(), self.unwrapped_soundstream.target_sample_hz)

			self.print(f'{steps}: saving sample to {str(os.path.join(self.results_folder , "samples" ))}')

		# save model every so often

		# print("saving model")

		# self.accelerator.wait_for_everyone()
		
		if self.is_main and not (steps % self.save_model_every):
			os.makedirs(os.path.join(self.results_folder , "results" ) , exist_ok=True)
			model_path = os.path.join(self.results_folder , "results" ,  f'soundstream.{steps}.pt')
			self.save(model_path)

			self.print(f'{steps}: saving model to {str(os.path.join(self.results_folder , "results" ) )}')

		# print("saved model")
		# self.accelerator.wait_for_everyone()

		self.steps += 1
		return logs

	def train(self, resume = False, log_fn = noop):


		if resume:
			save_path = os.path.join(self.results_folder , "results")
			chk = sorted(os.listdir(save_path) , key = lambda x: int(x.split('.')[1]))[-1]
			print("resuming from ", os.path.join(save_path , chk))
			self.load(os.path.join(save_path , chk))

		while self.steps < self.num_train_steps:
			logs = self.train_step()
			log_fn(logs)

		self.print('training complete')
