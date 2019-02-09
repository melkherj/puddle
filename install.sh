#!/bin/bash

# Download the amazon rewiews dataset
curl http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz > data/reviews_Toys_and_Games_5.json.gz
cd data
gunzip reviews_Toys_and_Games_5.json.gz
cd ..
