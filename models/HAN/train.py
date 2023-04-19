import torch
from models import HAN
from sklearn.metrics import *
from utils import load_data, EarlyStopping
import warnings
warnings.filterwarnings('ignore')

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1,logits


def main(args):
    g, features, labels, num_classes,  train_mask, val_mask, test_mask = load_data()

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])
    meta_paths = [['cc','cc_re'],['pc_re','pc']]
    
    model = HAN(meta_paths=meta_paths,
                in_size=features.shape[1],
                hidden_size=args['hidden_units'],   
                out_size=num_classes,
                num_heads=args['num_heads'],  
                dropout=args['dropout']).to(args['device'])
    g = g.to(args['device'])

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1 , _= evaluate(model, g, features, labels, val_mask, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
              'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f} | train acc {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1,train_acc))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1 ,logits = evaluate(model, g, features, labels, test_mask, loss_fcn)
    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | test acc {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1,test_acc))

    preds = logits.max(1)[1]
    if args['device']:
        l = labels[test_mask].cpu().detach().numpy()
        p = preds[test_mask].cpu().detach().numpy()
    else:
        l = labels[test_mask].detach().numpy()
        p = preds[test_mask].detach().numpy()
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
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    args = parser.parse_args().__dict__

    args = setup(args)

    main(args)
