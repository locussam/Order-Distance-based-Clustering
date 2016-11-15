# Order-Distance-based-Clustering
mainly based on 2011_CVPR_A Rank-Order Distance based Clustering Algorithm for Face Tagging

diff:
at first just compute k  nearest neighborhood 
when the cluster num become less than k ,set k = cluster_num in function cal_merge_m

How to use the cluster:
1: in the data file, every .txt represents a 1024 dimension face features.  
2: It constains a MakeList.bat of windows, use it to creat a LIST.TXT which contains all files name. 
3: open the LIST.TXT, in the end of the txt, delete the redandent name.
4: in init_data.h, the function:  get_data_and_name has default path = "./data/featureFolder/" which constains the feautres txt file.
