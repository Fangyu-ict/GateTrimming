# Network Slimming
This is our implementations of Network Slimming. The requirements are the same as Gate Trimming.

## Train with Sparsity

```shell
python main.py -sr --s 0.0001 --dataset cifar10 --arch vgg --depth 16
python main.py -sr --s 0.0001 --dataset cifar100 --arch vgg --depth 16
python main.py -sr --s 0.0001 --dataset cifar100 --arch vgg --depth 19
```

## Prune

```shell
python vggprune.py --dataset cifar10 --depth 16 --percent 0.45 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
python vggprune.py --dataset cifar100 --depth 16 --percent 0.35 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
python vggprune.py --dataset cifar100 --depth 19 --percent 0.35 --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
```
The pruned model will be named `pruned.pth.tar`.
