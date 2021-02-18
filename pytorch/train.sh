#python main.py ../dataset --dist-url 'tcp://chanti00.utep.edu:12356' --world-size 2 --multiprocessing-distributed -j 8 --batch-size 128 --epochs 2
python main.py ../dataset --multiprocessing-distributed -j 8 --batch-size 128 --epochs 100
