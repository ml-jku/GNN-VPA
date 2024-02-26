# VPA 
## A Variance-Preserving Aggregation Strategy for Graph Neural Networks



## Getting started

Install dependencies:
> pip install -r requirements.txt

Execute train.py with arguments, e.g.:
> python.exe train.py accelerator=cpu devices=3

Further configurations can be found in the conf-folder.

Results can be reproduced with:
>python.exe train.py --multirun model=gin model_name=gin agg=sum,mean,max,vpa tag_to_index=0 dataset_name=MUTAG,NCI1,PROTEINS,PTC_FM,COLLAB,REDDIT-BINARY,REDDIT-MULTI-5K fold_idx=0,1,2,3,4,5,6,7,8,9 devices=[0]
>python.exe train.py --multirun model=gcn model_name=gcn agg=sum,mean,max,vpa tag_to_index=0 dataset_name=MUTAG,NCI1,PROTEINS,PTC_FM,COLLAB,REDDIT-BINARY,REDDIT-MULTI-5K fold_idx=0,1,2,3,4,5,6,7,8,9 devices=[0]
>python.exe train.py --multirun model=sgc model_name=sgc agg=default,vpa tag_to_index=0 dataset_name=MUTAG,NCI1,PROTEINS,PTC_FM,COLLAB,REDDIT-BINARY,REDDIT-MULTI-5K fold_idx=0,1,2,3,4,5,6,7,8,9 devices=[0]
>python.exe train.py --multirun model=gat model_name=gat agg=sum,vpa tag_to_index=0 dataset_name=MUTAG,NCI1,PROTEINS,PTC_FM,COLLAB,REDDIT-BINARY,REDDIT-MULTI-5K fold_idx=0,1,2,3,4,5,6,7,8,9 devices=[0]

## Sources

https://pytorch-geometric.readthedocs.io/en/latest/
https://github.com/weihua916/powerful-gnns/tree/master
