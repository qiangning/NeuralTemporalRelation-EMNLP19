import matplotlib
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import click
from sklearn.model_selection import train_test_split
from utils import *
import os

from ELMo_Cache import *
from TemporalDataSet import *
from myLSTM import *
from pairwise_ffnn_pytorch import VerbNet

matplotlib.use('Agg')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
seed_everything(13234)

class experiment:
    def __init__(self,model,trainset,testset,testsetname,output_labels,params,exp_name,modelPath,skiptuning,gen_output=False):
        self.model = model
        self.params = params
        self.max_epoch = self.params.get('max_epoch',20)
        self.trainset, self.devset = self.split_train_dev(trainset)
        self.testset = testset
        self.testsetname = testsetname
        self.output_labels = output_labels
        self.exp_name = exp_name
        self.modelPath = "%s_%s" %(modelPath,self.exp_name)
        self.skiptuning = skiptuning
        self.gen_output = gen_output

    def split_train_dev(self,trainset):
        train,dev = train_test_split(trainset,test_size=0.2,random_state=self.params.get('seed',2093))
        return train,dev
    def train(self):
        self.best_epoch = self.max_epoch-1
        if not self.skiptuning:
            print("------------Training and Development------------")
            # all_test_accuracies refer to performance on devset
            all_train_losses, all_train_accuracies, all_test_accuracies =\
                self.trainHelper(self.trainset,self.devset,self.max_epoch,"tuning")
            # smooth within a window of +-2
            all_test_accuracies_smooth = [all_test_accuracies[0]]*2+all_test_accuracies+[all_test_accuracies[-1]]*2
            all_test_accuracies_smooth = [1.0/5*(all_test_accuracies_smooth[i-2]+all_test_accuracies_smooth[i-1]+all_test_accuracies_smooth[i]+all_test_accuracies_smooth[i+1]+all_test_accuracies_smooth[i+2]) for i in range(2,2+len(all_test_accuracies))]
            # all_test_accuracies_smooth = savgol_filter(all_test_accuracies, 3, 1)
            self.best_epoch, best_dev_acc = 0,0
            print("Select epoch based on smoothed dev accuracy curve")
            for i,acc in enumerate(all_test_accuracies_smooth):
                print("Epoch %d,\tAcc %.4f" %(i,acc))
                if acc > best_dev_acc:
                    best_dev_acc = acc
                    self.best_epoch = i
                # cool down
                # if i>=2 and acc < all_test_accuracies_smooth[i-1] and acc < all_test_accuracies_smooth[i-2]:
                #    break

            print("Best epoch=%d, best_dev_acc=%.4f/%.4f (before/after smoothing)" \
                  % (self.best_epoch, all_test_accuracies[self.best_epoch], all_test_accuracies_smooth[self.best_epoch]))

            print("------------Training with the best epoch number------------")
        else:
            print("------------Training with the max epoch number (skipped tuning)------------")
        trainset_aug = self.trainset+self.devset
        # self.trainHelper(trainset_aug,self.testset,self.max_epoch,"retrain")
        self.trainHelper(trainset_aug,self.testset,self.best_epoch+1,"retrain")


        print("\n\n#####Summary#####")
        print("---Max Epoch (%d) Acc=%.4f" %(self.max_epoch-1,all_test_accuracies[self.max_epoch-1]))
        if not self.skiptuning:
            print("---Tuned Epoch (%d) Acc=%.4f" %(self.best_epoch,all_test_accuracies[self.best_epoch]))

    def trainHelper(self,trainset,testset,max_epoch,tag):
        self.model.train()
        lr = self.params.get('lr',0.1)
        weight_decay = self.params.get('weight_decay',1e-4)
        step_size = self.params.get('step_size',10)
        gamma = self.params.get('gamma',0.3)
        optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        criterion = nn.CrossEntropyLoss()
        all_train_losses = []
        all_train_accuracies = []
        all_test_accuracies = []
        start = time.time()
        self.model.reset_parameters()
        for epoch in range(max_epoch):
            print("epoch: %d/%d" % (epoch, max_epoch-1), flush=True)
            current_train_loss = 0
            random.shuffle(trainset)
            scheduler.step()
            for i,temprel in enumerate(trainset):
                self.model.zero_grad()
                target = torch.cuda.LongTensor([self.output_labels[temprel.label]])
                output = self.model(temprel)
                loss = criterion(output, target)
                current_train_loss += loss
                if i % 1000 == 0:
                    print("%d/%d: %s %.4f %.4f" % (i, len(trainset), timeSince(start), loss, current_train_loss), flush=True)
                loss.backward()
                optimizer.step()
            all_train_losses.append(current_train_loss)
            current_train_acc, _, _ = self.eval(trainset)
            current_test_acc, confusion, curr_output = self.eval(testset,True)
            all_train_accuracies.append(float(current_train_acc))
            all_test_accuracies.append(float(current_test_acc))
            print("Loss at epoch %d: %.4f" % (epoch, current_train_loss), flush=True)
            print("Train acc at epoch %d: %.4f" % (epoch, current_train_acc), flush=True)
            print("Dev/Test acc at epoch %d: %.4f" % (epoch, current_test_acc), flush=True)
            print(confusion, flush=True)
            prec,rec,f1 = confusion2prf(confusion)
            print("Prec=%.4f, Rec=%.4f, F1=%.4f" %(prec,rec,f1))
            if tag=='retrain':
                if epoch==self.best_epoch and not self.skiptuning:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss
                    }, self.modelPath+"_selected")
                    self.writeoutput(os.path.join("./output",self.exp_name+"."+self.testsetname+".selected.output"),curr_output)
                
            # plot figures
            plt.figure(figsize=(6,6))
            plt.subplot(211)
            plt.plot(all_train_losses,'k')
            plt.grid()
            plt.ylabel('Training loss')
            plt.xlabel('Epoch')
            plt.rcParams.update({'font.size': 12})
            plt.subplot(212)
            plt.plot(all_train_accuracies,'k--')
            plt.plot(all_test_accuracies,'k-*')
            if tag=='retrain':
                plt.legend(["Train","Test"])
            else:
                plt.legend(["Train","Dev"])
            plt.grid()
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.rcParams.update({'font.size': 12})

            plt.tight_layout()
            plt.savefig("figs/%s_%s.pdf" % (self.exp_name,tag))
            plt.savefig("figs/%s_%s.pdf" % (self.exp_name,tag))
            plt.close('all')
        return all_train_losses,all_train_accuracies,all_test_accuracies

    def writeoutput(self,write2path,output):
        f = open(write2path, 'w')
        for docid in output:
            for pairkey in output[docid]:
                f.write("%s,%s,%s\n" \
                        % (docid, pairkey, output[docid][pairkey]))
        f.close()
    def test(self):
        self.model.eval()
        test_acc, test_confusion, test_output = self.eval(self.testset,self.gen_output)
        test_prec = (test_confusion[0][0]+test_confusion[1][1]+test_confusion[2][2])/(np.sum(test_confusion)-np.sum(test_confusion,axis=0)[3])
        test_rec = (test_confusion[0][0]+test_confusion[1][1]+test_confusion[2][2])/(np.sum(test_confusion)-np.sum(test_confusion[3][:]))
        test_f1 = 2*test_prec*test_rec / (test_rec+test_prec)
        print("DATASET=%s" % self.testsetname)
        print("TEST ACCURACY=%.4f" % test_acc)
        print("TEST PRECISION=%.4f" % test_prec)
        print("TEST RECALL=%.4f" % test_rec)
        print("TEST F1=%.4f" % test_f1)
        print("CONFUSION MAT:")
        print(test_confusion)
        if self.gen_output:
            self.writeoutput(os.path.join("./output",self.exp_name+"."+self.testsetname+".output"),test_output)

    def eval(self,eval_on_set, gen_output=False):
        was_training = self.model.training
        self.model.eval()
        confusion = np.zeros((len(self.output_labels), len(self.output_labels)), dtype=int)
        output = {}
        softmax = nn.Softmax()
        for ex in eval_on_set:
            prediction = self.model(ex)
            prediction_label = categoryFromOutput(prediction)
            if gen_output:
                prediction_scores = softmax(prediction)
                if ex.docid not in output:
                    output[ex.docid] = {}
                output[ex.docid]["%s,%s" %(ex.source,ex.target)]\
                = "%d,%f,%f,%f,%f" %(prediction_label,prediction_scores[0][0],prediction_scores[0][1],prediction_scores[0][2],prediction_scores[0][3])
            confusion[self.output_labels[ex.label]][prediction_label] += 1
        if was_training:
            self.model.train()
        return 1.0 * np.sum([confusion[i][i] for i in range(4)]) / np.sum(confusion), confusion, output

class bigramGetter_fromNN:
    def __init__(self,emb_path,mdl_path,ratio=0.3,layer=1,emb_size=200,splitter=','):
        self.verb_i_map = {}
        f = open(emb_path)
        lines = f.readlines()
        for i,line in enumerate(lines):
            self.verb_i_map[line.split(splitter)[0]] = i
        f.close()
        self.model = VerbNet(len(self.verb_i_map),hidden_ratio=ratio,emb_size=emb_size,num_layers=layer)
        checkpoint = torch.load(mdl_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    def eval(self,v1,v2):
        return self.model(torch.from_numpy(np.array([[self.verb_i_map[v1],self.verb_i_map[v2]]])).cuda())
    def getBigramStatsFromTemprel(self,temprel):
        v1,v2='',''
        for i,position in enumerate(temprel.position):
            if position == 'E1':
                v1 = temprel.lemma[i]
            elif position == 'E2':
                v2 = temprel.lemma[i]
                break
        if v1 not in self.verb_i_map or v2 not in self.verb_i_map:
            return torch.cuda.FloatTensor([0,0]).view(1,-1)
        return torch.cat((self.eval(v1,v2),self.eval(v2,v1)),1).view(1,-1)
    def retrieveEmbeddings(self,temprel):
        v1, v2 = '', ''
        for i, position in enumerate(temprel.position):
            if position == 'E1':
                v1 = temprel.lemma[i]
            elif position == 'E2':
                v2 = temprel.lemma[i]
                break
        if v1 not in self.verb_i_map or v2 not in self.verb_i_map:
            return torch.zeros_like(self.model.retrieveEmbeddings(torch.from_numpy(np.array([[0,0]])).cuda()).view(1,-1))
        return self.model.retrieveEmbeddings(torch.from_numpy(np.array([[self.verb_i_map[v1],self.verb_i_map[v2]]])).cuda()).view(1,-1)

@click.command()
@click.option("--lstm_hid_dim",default=128)
@click.option("--nn_hid_dim",default=64)
@click.option("--pos_emb_dim",default=32)
@click.option("--common_sense_emb_dim",default=64)
@click.option("--bigramstats_dim",default=1)
@click.option("--granularity",default=0.1)
@click.option("--lr",default=0.1)
@click.option("--weight_decay",default=1e-4)
@click.option("--step_size",default=10)
@click.option("--gamma",default=0.2)
@click.option("--max_epoch",default=50)
@click.option("--expname",default="test")
@click.option("--skiptuning", is_flag=True)
@click.option("--skiptraining", is_flag=True)
@click.option("--gen_output", is_flag=True)
@click.option("--bilstm",is_flag=True)
@click.option("--debug",is_flag=True)
@click.option("--sd",default=13234)
@click.option("--testsetname",default="matres")
def run(lstm_hid_dim, nn_hid_dim, pos_emb_dim, common_sense_emb_dim, bigramstats_dim, granularity, lr, weight_decay, step_size, gamma, max_epoch, expname, skiptuning, skiptraining, gen_output, bilstm, debug, sd, testsetname):
    seed_everything(sd)
    trainset = temprel_set("data/trainset-temprel.xml")
    if testsetname == "matres":
        testset = temprel_set("data/testset-temprel.xml","matres")
        w2v_ser_dir = "ser/"
    else:
        testset = temprel_set("data/tcr-temprel.xml","tcr")
        w2v_ser_dir = "ser/TCR/"
    
    embedding_dim = 1024
    print("Using ELMo (original)")
    emb_cache = elmo_cache(None, w2v_ser_dir+"elmo_cache_original.pkl", verbose=False)
    position2ix = {"B":0,"M":1,"A":2,"E1":3,"E2":4}

    output_labels = {"BEFORE":0,"AFTER":1,"EQUAL":2,"VAGUE":3}

    params = {'embedding_dim':embedding_dim,\
                  'lstm_hidden_dim':lstm_hid_dim,\
                  'nn_hidden_dim':nn_hid_dim,\
                  'position_emb_dim':pos_emb_dim,\
                  'bigramStats_dim':bigramstats_dim,\
                  'lemma_emb_dim':200,\
                  'dropout':False,\
                  'batch_size':1}
    params_optim = {'lr':lr,'weight_decay':weight_decay,'step_size':step_size,'gamma':gamma,'max_epoch':max_epoch}
    print("___________________HYPER-PARAMETERS:LSTM___________________")
    print(params)
    print("___________________HYPER-PARAMETERS:OPTIMIZER___________________")
    print(params_optim)

    ratio = 0.3
    emb_size = 200
    layer = 1
    splitter = " "
    print("---------")
    print("ratio=%s,emb_size=%d,layer=%d" % (str(ratio), emb_size, layer))
    emb_path = './ser/embeddings_%.1f_%d_%d_timelines.txt' % (ratio, emb_size, layer)
    mdl_path = './ser/pairwise_model_%.1f_%d_%d.pt' % (ratio, emb_size, layer)
    
    bigramGetter = bigramGetter_fromNN(emb_path, mdl_path, ratio, layer, emb_size, splitter=splitter)
    model = lstm_siam(params, emb_cache, bigramGetter, granularity=granularity, common_sense_emb_dim=common_sense_emb_dim,bidirectional=bilstm,lowerCase=False)
    if debug:
        expname += "_debug"
        exp = experiment(model=model, trainset=trainset.temprel_ee[:100], testset=testset.temprel_ee[:100], testsetname=testsetname, \
                         params=params_optim, exp_name=expname, modelPath="models/ckpt", \
                         output_labels=output_labels, skiptuning=skiptuning,gen_output=gen_output)
    else:
        exp = experiment(model=model,trainset=trainset.temprel_ee,testset=testset.temprel_ee, testsetname=testsetname,\
                         params=params_optim,exp_name=expname,modelPath="models/ckpt", \
                         output_labels=output_labels,skiptuning=skiptuning,gen_output=gen_output)
    if not skiptraining:
        exp.train()
    else:
        exp.model.load_state_dict(torch.load(exp.modelPath+"_selected")['model_state_dict'])
    exp.test()

if __name__ == '__main__':
    run()
