import torch
import sys,time
import pickle

def save_checkpoint(state_dict, model_name):
    t = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    filename = 'checkpoints/' + model_name + "/checkpoint_"+ '.pth.tar' #+ str(state_dict["epoch"]) + "_" + t + '.pth.tar'
    torch.save(state_dict,filename)
    # if is_best:
    #     shutil.copyfile(filename,  "checkpoints/best_model/" + model_name+ '/best_model.pth.tar')

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))#100.0
        return res

class ShowProcess():

    i = 0
    max_steps = 0 #
    max_arrow = 88 #
    infoDone = 'done'


    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0

    # [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, loss, prec1,prec5, min, sec):

        self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps) #
        num_line = self.max_arrow - num_arrow #
        percent = self.i * 100.0 / self.max_steps
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%'\
                      + '  loss:%.4f'% loss + '  prec1:%.4f'% prec1+ '  prec5:%.4f'% prec5\
                      + '  time:%.0fm'% min + ' %.0fs'% sec+ '     \r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        self.i = 0

def save_model(model,name):
    f = open('./'+name+'.pickle', 'wb')
    pickle.dump(model, f)
    f.close()

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()