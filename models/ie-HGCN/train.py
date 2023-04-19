import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from model import HGCN
from dgl.data.utils import load_graphs
import warnings
import sklearn.exceptions
from sklearn.metrics import *

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def row_normalize(mat):
	rowsum = mat.sum(1)
	rowsum[rowsum == 0.] = 0.01
	return (mat.t()/rowsum).t()


def load_data():
	graph_path = "../../data/others/company_graph.bin"
	graph = load_graphs(graph_path)
	g = graph[0][0]

	labels = g.nodes['company'].data['label']
  
	total_len = len(labels)
	idx_all = np.arange(total_len)	
	total_len = len(labels)

	np.random.shuffle(idx_all)

	train_idx = torch.tensor(idx_all[:int(total_len/10*7)]).long()
	val_idx = torch.tensor(idx_all[int(total_len/10*4):int(total_len/10*7)]).long()
	test_idx = torch.tensor(idx_all[int(total_len/10*7):]).long()

	label_dict = {'company':[labels,train_idx,val_idx,test_idx]}
	f_c = g.nodes['company'].data['feat']
	f_p = g.nodes['person'].data['feat']
	feature_dict = {'company':f_c,'person':f_p}

	adj_name = [t for t in g.etypes if 're' not in t]
	adj_list = {name:row_normalize(g.adj(etype=name).to_dense()) for name in adj_name}
	adj_dict = {'company':{'company':adj_list['branch'],
						   'company':adj_list['invest_c'],
						   'company':adj_list['own_c']},
				'person':{'company':adj_list['serve'],
					      'company':adj_list['invest_h'],
						  'company':adj_list['own']}}

	return label_dict,feature_dict,adj_dict


def train(epoch):

	model.train()
	optimizer.zero_grad()
	logits, _ = model(ft_dict, adj_dict)

	p_logits = F.log_softmax(logits['company'], dim=1)
	idx_train_p = label['company'][1]
	x_train_p = p_logits[idx_train_p]
	y_train_p = label['company'][0][idx_train_p]
	loss_train = F.nll_loss(x_train_p, y_train_p)
	f1_micro_train_p = f1_score(y_train_p.data.cpu(), x_train_p.data.cpu().argmax(1), average='micro')
	f1_macro_train_p = f1_score(y_train_p.data.cpu(), x_train_p.data.cpu().argmax(1), average='macro')
	loss_train.backward()
	optimizer.step()

	'''///////////////// Validation ///////////////////'''
	model.eval()
	logits, _ = model(ft_dict, adj_dict)
	p_logits = F.log_softmax(logits['company'], dim=1)
	idx_val_p = label['company'][2]
	x_val_p = p_logits[idx_val_p]
	y_val_p = label['company'][0][idx_val_p]
	f1_micro_val_p = f1_score(y_val_p.data.cpu(), x_val_p.data.cpu().argmax(1), average='micro')
	f1_macro_val_p = f1_score(y_val_p.data.cpu(), x_val_p.data.cpu().argmax(1), average='macro')
	
	if epoch % 1 == 0:
		print(
			  'epoch: {:3d}'.format(epoch),
			  'train loss: {:.4f}'.format(loss_train.item()),
			  'train micro f1 p: {:.4f}'.format(f1_micro_train_p.item()),
			  'train macro f1 p: {:.4f}'.format(f1_macro_train_p.item()),
			  'val micro f1 p: {:.4f}'.format(f1_micro_val_p.item()),
			  'val macro f1 p: {:.4f}'.format(f1_macro_val_p.item()),
			 )


def test():
	model.eval()
	logits, _ = model(ft_dict, adj_dict)

	p_logits = F.log_softmax(logits['company'], dim=1)
	idx_test_p = label['company'][3]
	x_test_p = p_logits[idx_test_p]
	y_test_p = label['company'][0][idx_test_p]

	if cuda:
		l = y_test_p.data.cpu()
		p = x_test_p.data.cpu().argmax(1)
	else:
		l = y_test_p.data
		p = x_test_p.data.argmax(1)
	acc = accuracy_score(l,p)
	recall = recall_score(l,p)
	f1 = f1_score(l,p)
	pre = f1 * recall /(2 * recall - f1)
	print()
	print("Test set results:")
	print("      Accuracy = {:.4f}".format(acc))
	print("     Precision = {:.4f}".format(pre))
	print("        Recall = {:.4f}".format(recall))
	print("      F1-score = {:.4f}".format(f1))
	print()


if __name__ == '__main__':

	cuda = True # Enables CUDA training.
	lr = 0.01 # Initial learning rate.c
	weight_decay = 5e-4 # Weight decay (L2 loss on parameters).
	type_att_size = 64 # type attention parameter dimension
	type_fusion = 'att' # mean
	seed = 42
	np.random.seed(seed)
	torch.manual_seed(seed)
	run_num = 1
	train_percent = 0.3

	for run in range(run_num):
		t_start = time.time()

		if cuda and torch.cuda.is_available():
			torch.cuda.manual_seed(seed)
		print('\nHGCN run: ', run)
		print('train percent: ', train_percent)
		print('seed: ', seed)
		print('type fusion: ', type_fusion)
		print('type att size: ', type_att_size)

		hid_layer_dim = [32,32,32] # company
		epochs = 200
		label, ft_dict, adj_dict = load_data()
		output_layer_shape = dict.fromkeys(ft_dict.keys(), 2)

		layer_shape = []
		input_layer_shape = dict([(k, ft_dict[k].shape[1]) for k in ft_dict.keys()])
		layer_shape.append(input_layer_shape)
		hidden_layer_shape = [dict.fromkeys(ft_dict.keys(), l_hid) for l_hid in hid_layer_dim]
		layer_shape.extend(hidden_layer_shape)
		layer_shape.append(output_layer_shape)
 
		# Model and optimizer
		net_schema = dict([(k, list(adj_dict[k].keys())) for k in adj_dict.keys()])
		model = HGCN(
					net_schema=net_schema,
					layer_shape=layer_shape,
					label_keys=list(label.keys()), 
					type_fusion=type_fusion,
					type_att_size=type_att_size,
					)
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

		if cuda and torch.cuda.is_available():
			model.cuda()
			for k in ft_dict:
				ft_dict[k] = ft_dict[k].cuda()
			for k in adj_dict:
				for kk in adj_dict[k]:
					adj_dict[k][kk] = adj_dict[k][kk].cuda()
			for k in label:
				for i in range(len(label[k])):
					label[k][i] = label[k][i].cuda()

		for epoch in range(epochs):
			train(epoch)
		t_end = time.time()
		print('Total time: ', t_end - t_start)

		test()

		
		