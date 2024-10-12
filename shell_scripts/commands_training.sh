python3 train.py -classes 6 -custom_preprocess 0 -network efficientnet -epochs 352

echo "First done (samplewisenorm off), val acc based lr scheduling" >> logger.txt


python3 train.py -classes 6 -custom_preprocess 1 -network efficientnet -epochs 353

echo "Second done (samplewisenorm off)" >> logger.txt
