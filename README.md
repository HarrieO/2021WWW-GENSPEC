# Robust Generalization and Safe Query-Specialization in Counterfactual Learning to Rank
This repository contains the code used for the experiments in "Robust Generalization and Safe Query-Specialization in Counterfactual Learning to Rank" published at WWW 2021 ([preprint available](https://arxiv.org/abs/2102.05990)).

Citation
--------

If you use this code to produce results for your scientific publication, or if you share a copy or fork, please refer to our WWW 2021 paper:

```
@inproceedings{oosterhuis2021genspec,
  Author = {Oosterhuis, Harrie and de Rijke, Maarten},
  Booktitle = {Proceedings of The Web Conference 2021},
  Organization = {ACM},
  Title = {Robust Generalization and Safe Query-Specialization in Counterfactual Learning to Rank},
  Year = {2021}
}
```

License
-------

The contents of this repository are licensed under the [MIT license](LICENSE). If you modify its contents in any way, please link back to this repository.

Usage
-------

This code makes use of [Python 3](https://www.python.org/) and the [NumPy](https://numpy.org/) package, make sure they are installed.

A file is required that explains the location and details of the LTR datasets available on the system, for the Yahoo! Webscope, MSLR-Web30k, and Istella datasets an example file is available. Copy the file:
```
cp example_datasets_info.txt local_dataset_info.txt
```
Open this copy and edit the paths to the folders where the train/test/vali files are placed.

Here are some command-line examples that illustrate how the results in the paper can be replicated.
First create a folder to store the results:
```
mkdir local_output
```
We reused the pretrained from our previous WSDM'21 publication, the models are included in the repository.
If you wish to pretrain your own models, you can use ([the following repository](https://github.com/HarrieO/2021wsdm-unifying-LTR)).
First we create a folder to store the clicks we will generate for GENSPEC:
```
mkdir -p local_output/yahoo/clicks/genspec
```
Then we will generate clicks using the pretrained model for the Yahoo! Webscope dataset:
```
python3 genspec_generate_clicks.py pretrained/Webscope_C14_Set1/pretrained_model.txt local_output/yahoo/clicks/genspec/ --dataset_info_path local_dataset_info.txt
```
This command will fill the *local_output/yahoo/clicks/genspec/* folder with pickle files containing varying numbers of clicks.
The *generate_clicks.py* implementation supports multiprocessing to speed things up, enable this by setting the *--num_proc* flag to the number of parallel processes it may use.
Now that we have gathered large numbers of clicks, we can apply GENSPEC, for example, to 10000000 generated clicks.
Again, we create a folder to store the results:
```
mkdir -p local_output/yahoo/results/genspec
```
The following will compute results for GENSPEC and SEA with different confidence values and stores them in a results file:
```
python3 genspec_eval.py pretrained/Webscope_C14_Set1/pretrained_model.txt local_output/yahoo/clicks/genspec/10000000clicks.pkl local_output/yahoo/results/genspec/10000000_clicks_results.txt --dataset_info_path local_dataset_info.txt 
```
The results file *local_output/yahoo/results/genspec/10000000_clicks_results.txt* contains json with all the information relevant to reproduce the results in our paper.

To reproduce the PBM bandit baseline, we again generate clicks first and then perform evaluation afterwards.
First we make the folders:
```
mkdir -p local_output/yahoo/clicks/pbm
mkdir -p local_output/yahoo/results/pbm
```
The following will first generate clicks and then perform evaluation on them:
```
python3 pbm.py local_output/yahoo/clicks/pbm --dataset_info_path local_dataset_info.txt
python3 bandit_eval.py local_output/yahoo/clicks/pbm/10000000clicks.pkl local_output/yahoo/results/pbm/10000000_clicks_results.txt
```
For the hotfix baseline:
```
mkdir -p local_output/yahoo/clicks/hotfix
mkdir -p local_output/yahoo/results/hotfix
python3 hotfix.py pretrained/Webscope_C14_Set1/pretrained_model.txt local_output/yahoo/clicks/hotfix --dataset_info_path local_dataset_info.txt
python3 bandit_eval.py local_output/yahoo/clicks/hotfix/10000000clicks.pkl local_output/yahoo/results/hotfix/10000000_clicks_results.txt --dataset_info_path local_dataset_info.txt
```
For the top-k hotfix baseline, we simply add the *--k* flag following the paper will use the top-10:
```
mkdir -p local_output/yahoo/clicks/top10_hotfix
mkdir -p local_output/yahoo/results/top10_hotfix
python3 hotfix.py pretrained/Webscope_C14_Set1/pretrained_model.txt local_output/yahoo/clicks/hotfix --k 10 --dataset_info_path local_dataset_info.txt
python3 bandit_eval.py local_output/yahoo/clicks/top10_hotfix/10000000clicks.pkl local_output/yahoo/results/top10_hotfix/10000000_clicks_results.txt --dataset_info_path local_dataset_info.txt
```

