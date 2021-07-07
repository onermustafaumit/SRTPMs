import numpy as np
import argparse
from datetime import datetime
import os
import sys
import time

from model import Model
from dataset import Dataset, custom_collate_fn, worker_init_fn

import torch
import torch.utils.data

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train a deep MIL model to predict sample-level tumor purity from WSIs')

parser.add_argument('--init_model_file', default='', help='the path of initial model file', dest='init_model_file')
parser.add_argument('--image_dir', default='../Images/all_cropped_patches_primary_solid_tumor__level1__stride512__size512', help='Image directory for tumor patches', dest='image_dir')
parser.add_argument('--normal_image_dir', default='../Images/all_cropped_patches_solid_tissue_normal__level1__stride512__size512', help='Image directory for normal patches', dest='normal_image_dir')
parser.add_argument('--dataset_dir', default='../dataset/all_patches__level1__stride512__size512', help='dataset info folder', dest='dataset_dir')
parser.add_argument('--patch_size', default='299', type=int, help='patch size', dest='patch_size')
parser.add_argument('--num_instances', default='200', type=int, help='number of instances (patches) in a bag', dest='num_instances')
parser.add_argument('--num_features', default='128', type=int, help='number of features', dest='num_features')
parser.add_argument('--num_bins', default='21', type=int, help='number of bins in distribution pooling filter', dest='num_bins')
parser.add_argument('--sigma', default='0.05', type=float, help='sigma in distribution pooling filter', dest='sigma')
parser.add_argument('--num_classes', default='1', type=int, help='number of classes', dest='num_classes')
parser.add_argument('--batch_size', default='2', type=int, help='batch size', dest='batch_size')
parser.add_argument('--learning_rate', default='1e-4', type=float, help='number of patches each patient has', dest='learning_rate')
parser.add_argument('--num_epochs', default=10000, type=int, help='number of steps of execution (default: 1000000)', dest='num_epochs')
parser.add_argument('--save_interval', default=10, type=int, help='model save interval (default: 1000)', dest='save_interval')
parser.add_argument('--metrics_dir', default='loss_data', help='file to log training metrics (e.g. loss)', dest='metrics_dir')
parser.add_argument('--models_dir', default='saved_models', help='directory to save models', dest='models_dir')
parser.add_argument('--valid_fold', default=3, type=int, help='id of fold to be used as validation set', dest='valid_fold')
parser.add_argument('--test_fold', default=4, type=int, help='id of fold to be used as test set', dest='test_fold')

FLAGS = parser.parse_args()
FLAGS_dict = vars(FLAGS)

# create metrics_dir
if not os.path.exists(FLAGS.metrics_dir):
	os.makedirs(FLAGS.metrics_dir)

# create models_dir
if not os.path.exists(FLAGS.models_dir):
	os.makedirs(FLAGS.models_dir)

# get current time and use as model id
current_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
metrics_file = '{}/loss_metrics__{}.txt'.format(FLAGS.metrics_dir,current_time)

train_fold_list = np.arange(5)
train_fold_list = np.delete(train_fold_list, [FLAGS.valid_fold,FLAGS.test_fold])
print('train_fold_list:{}'.format(train_fold_list))

print('Preparing training dataset ...')
train_dataset = Dataset(image_dir=FLAGS.image_dir, normal_image_dir=FLAGS.normal_image_dir, dataset_dir=FLAGS.dataset_dir, dataset_type='train', patch_size=FLAGS.patch_size, fold_list=train_fold_list, num_instances=FLAGS.num_instances)
num_patients_train = train_dataset.num_patients
print("Training Data - num_patients: {}".format(num_patients_train))

print('Preparing validation dataset ...')
val_dataset = Dataset(image_dir=FLAGS.image_dir, normal_image_dir=FLAGS.normal_image_dir, dataset_dir=FLAGS.dataset_dir, dataset_type='valid', patch_size=FLAGS.patch_size, fold_list=[FLAGS.valid_fold], num_instances=FLAGS.num_instances)
num_patients_val = val_dataset.num_patients
print("Validation Data - num_patients: {}".format(num_patients_val))

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn, worker_init_fn=worker_init_fn)

# construct model
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(num_classes=FLAGS.num_classes, num_instances=FLAGS.num_instances, num_features=FLAGS.num_features, num_bins=FLAGS.num_bins, sigma=FLAGS.sigma)
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=FLAGS.learning_rate, weight_decay=0.0005)

# initialize weights from a file
if FLAGS.init_model_file:
	if os.path.isfile(FLAGS.init_model_file):
		state_dict = torch.load(FLAGS.init_model_file, map_location=device)
		model.load_state_dict(state_dict['model_state_dict'])
		optimizer.load_state_dict(state_dict['optimizer_state_dict'])
		print('weights loaded successfully!!!\n{}'.format(FLAGS.init_model_file))


# print model parameters
print('# Model parameters:')
for key in FLAGS_dict.keys():
	print('# {} = {}'.format(key, FLAGS_dict[key]))

print("# Training Data - num_samples: {}".format(num_patients_train))
print("# Validation Data - num_samples: {}".format(num_patients_val))


# write model parameters into metrics file
with open(metrics_file,'w') as f_metrics_file:
	f_metrics_file.write('# Model parameters:\n')

	for key in FLAGS_dict.keys():
		f_metrics_file.write('# {} = {}\n'.format(key, FLAGS_dict[key]))

	f_metrics_file.write("# Training Data - num_samples: {}\n".format(num_patients_train))
	f_metrics_file.write("# Validation Data - num_samples: {}\n".format(num_patients_val))
	
	f_metrics_file.write('# epoch\ttraining_loss\tvalidation_loss\n')


# define loss criterion
criterion = torch.nn.L1Loss()

for epoch in range(FLAGS.num_epochs):
	print('############## EPOCH - {} ##############'.format(epoch+1))
	training_loss = 0
	validation_loss = 0

	# train for one epoch
	print('******** training ********')
		
	num_predictions = 0

	pbar = tqdm(total=len(train_data_loader))
	
	model.train()
	for images, targets in train_data_loader:
		images = images.to(device)
		targets = targets.to(device)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		y_logits = model(images)
		loss = criterion(y_logits, targets)
		loss.backward()
		optimizer.step()

		training_loss += loss.item()*targets.size(0)

		num_predictions += targets.size(0)

		pbar.update(1)

	training_loss /= num_predictions

	pbar.close()


	# evaluate on the validation dataset
	print('******** validation ********')

	num_predictions = 0

	pbar = tqdm(total=len(val_data_loader))

	model.eval()
	with torch.no_grad():
		for images, targets in val_data_loader:
			images = images.to(device)
			targets = targets.to(device)

			# forward
			y_logits = model(images)
			loss = criterion(y_logits, targets)

			validation_loss += loss.item()*targets.size(0)

			num_predictions += targets.size(0)

			pbar.update(1)

	validation_loss /= num_predictions

	pbar.close()

	print('Epoch=%d ### training_loss=%5.3f ### validation_loss=%5.3f' % (epoch+1, training_loss, validation_loss))

	# logging loss values into metrics file
	with open(metrics_file,'a') as f_metrics_file:
		f_metrics_file.write('%d\t%5.3f\t%5.3f\n' % (epoch+1, training_loss, validation_loss))

	# save model
	if (epoch+1) % FLAGS.save_interval == 0:
		model_weights_filename = '{}/model_weights__{}__{}.pth'.format(FLAGS.models_dir,current_time,epoch+1)
		state_dict = {	'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict()}
		torch.save(state_dict, model_weights_filename)
		print("Model weights saved in file: ", model_weights_filename)


model_weights_filename = '{}/model_weights__{}__{}.pth'.format(FLAGS.models_dir,current_time,epoch+1)
state_dict = {	'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}
torch.save(state_dict, model_weights_filename)
print("Model weights saved in file: ", model_weights_filename)

print('Training finished!!!')

