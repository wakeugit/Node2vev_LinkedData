first take look of the requirement.txt file

# Start example
python src/class_prediction.py --graph graph/bgs.nt --dataset graph/bgs_completeDataset_lith.tsv

# Simulate Homophilie
python src/class_prediction.py --graph graph/bgs.nt --dataset graph/bgs_completeDataset_lith.tsv --p 1 --q 0.5

# Simulate Structural equivalence
python src/class_prediction.py --graph graph/bgs.nt --dataset graph/bgs_completeDataset_lith.tsv --p 1 --q 2

# Set the vector dimensions
python src/class_prediction.py --graph graph/bgs.nt --dataset graph/bgs_completeDataset_lith.tsv --dimensions 200

 


