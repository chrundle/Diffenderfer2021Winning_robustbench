# Full-precision weight CARDS (LRR pruned)
bsub -nnodes 1 -G safeml -W 60 python Diffenderfer_benchmarks/LRR_CARD_benchmark.py 
bsub -nnodes 1 -G safeml -W 60 python Diffenderfer_benchmarks/LRR_CARD_Deck_benchmark.py 

bsub -nnodes 1 -G safeml -W 60 python Diffenderfer_benchmarks/C100_LRR_CARD_benchmark.py 
bsub -nnodes 1 -G safeml -W 60 python Diffenderfer_benchmarks/C100_LRR_CARD_Deck_benchmark.py 

# Binary weight CARDS (EP pruned)
bsub -nnodes 1 -G safeml -W 60 python Diffenderfer_benchmarks/Binary_CARD_benchmark.py 
bsub -nnodes 1 -G safeml -W 60 python Diffenderfer_benchmarks/Binary_CARD_Deck_benchmark.py 

bsub -nnodes 1 -G safeml -W 60 python Diffenderfer_benchmarks/C100_Binary_CARD_benchmark.py 
bsub -nnodes 1 -G safeml -W 60 python Diffenderfer_benchmarks/C100_Binary_CARD_Deck_benchmark.py 
