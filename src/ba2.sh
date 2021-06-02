bsub -R "rusage[mem=1500,ngpus_excl_p=1]" python train.py --verbose --layers 1 --nodes 32 --pde $1 --update_rule lrannealing
bsub -R "rusage[mem=1500,ngpus_excl_p=1]" python train.py --verbose --layers 1 --nodes 32 --pde $1 --update_rule lrannealing --resample
bsub -R "rusage[mem=1500,ngpus_excl_p=1]" python train.py --verbose --layers 1 --nodes 32 --pde $1 --update_rule lrannealing --aggregate_boundaries
bsub -R "rusage[mem=1500,ngpus_excl_p=1]" python train.py --verbose --layers 1 --nodes 32 --pde $1 --update_rule lrannealing --resample --aggregate_boundaries

bsub -R "rusage[mem=2500,ngpus_excl_p=1]" python train.py --verbose --layers 1 --nodes 128 --pde $1 --update_rule lrannealing
bsub -R "rusage[mem=2500,ngpus_excl_p=1]" python train.py --verbose --layers 1 --nodes 128 --pde $1 --update_rule lrannealing --resample
bsub -R "rusage[mem=2500,ngpus_excl_p=1]" python train.py --verbose --layers 1 --nodes 128 --pde $1 --update_rule lrannealing --aggregate_boundaries
bsub -R "rusage[mem=2500,ngpus_excl_p=1]" python train.py --verbose --layers 1 --nodes 128 --pde $1 --update_rule lrannealing --resample --aggregate_boundaries

bsub -R "rusage[mem=3500,ngpus_excl_p=1]" python train.py --verbose --layers 1 --nodes 256 --pde $1 --update_rule lrannealing
bsub -R "rusage[mem=3500,ngpus_excl_p=1]" python train.py --verbose --layers 1 --nodes 256 --pde $1 --update_rule lrannealing --resample
bsub -R "rusage[mem=3500,ngpus_excl_p=1]" python train.py --verbose --layers 1 --nodes 256 --pde $1 --update_rule lrannealing --aggregate_boundaries
bsub -R "rusage[mem=3500,ngpus_excl_p=1]" python train.py --verbose --layers 1 --nodes 256 --pde $1 --update_rule lrannealing --resample --aggregate_boundaries

bsub -R "rusage[mem=2500,ngpus_excl_p=1]" python train.py --verbose --layers 2 --nodes 32 --pde $1 --update_rule lrannealing
bsub -R "rusage[mem=2500,ngpus_excl_p=1]" python train.py --verbose --layers 2 --nodes 32 --pde $1 --update_rule lrannealing --resample
bsub -R "rusage[mem=2500,ngpus_excl_p=1]" python train.py --verbose --layers 2 --nodes 32 --pde $1 --update_rule lrannealing --aggregate_boundaries
bsub -R "rusage[mem=2500,ngpus_excl_p=1]" python train.py --verbose --layers 2 --nodes 32 --pde $1 --update_rule lrannealing --resample --aggregate_boundaries

bsub -R "rusage[mem=3500,ngpus_excl_p=1]" python train.py --verbose --layers 2 --nodes 128 --pde $1 --update_rule lrannealing
bsub -R "rusage[mem=3500,ngpus_excl_p=1]" python train.py --verbose --layers 2 --nodes 128 --pde $1 --update_rule lrannealing --resample
bsub -R "rusage[mem=3500,ngpus_excl_p=1]" python train.py --verbose --layers 2 --nodes 128 --pde $1 --update_rule lrannealing --aggregate_boundaries
bsub -R "rusage[mem=3500,ngpus_excl_p=1]" python train.py --verbose --layers 2 --nodes 128 --pde $1 --update_rule lrannealing --resample --aggregate_boundaries

bsub -R "rusage[mem=4500,ngpus_excl_p=1]" python train.py --verbose --layers 2 --nodes 256 --pde $1 --update_rule lrannealing
bsub -R "rusage[mem=4500,ngpus_excl_p=1]" python train.py --verbose --layers 2 --nodes 256 --pde $1 --update_rule lrannealing --resample
bsub -R "rusage[mem=4500,ngpus_excl_p=1]" python train.py --verbose --layers 2 --nodes 256 --pde $1 --update_rule lrannealing --aggregate_boundaries
bsub -R "rusage[mem=4500,ngpus_excl_p=1]" python train.py --verbose --layers 2 --nodes 256 --pde $1 --update_rule lrannealing --resample --aggregate_boundaries

bsub -R "rusage[mem=3500,ngpus_excl_p=1]" python train.py --verbose --layers 3 --nodes 32 --pde $1 --update_rule lrannealing
bsub -R "rusage[mem=3500,ngpus_excl_p=1]" python train.py --verbose --layers 3 --nodes 32 --pde $1 --update_rule lrannealing --resample
bsub -R "rusage[mem=3500,ngpus_excl_p=1]" python train.py --verbose --layers 3 --nodes 32 --pde $1 --update_rule lrannealing --aggregate_boundaries
bsub -R "rusage[mem=3500,ngpus_excl_p=1]" python train.py --verbose --layers 3 --nodes 32 --pde $1 --update_rule lrannealing --resample --aggregate_boundaries

bsub -R "rusage[mem=4500,ngpus_excl_p=1]" python train.py --verbose --layers 3 --nodes 128 --pde $1 --update_rule lrannealing
bsub -R "rusage[mem=4500,ngpus_excl_p=1]" python train.py --verbose --layers 3 --nodes 128 --pde $1 --update_rule lrannealing --resample
bsub -R "rusage[mem=4500,ngpus_excl_p=1]" python train.py --verbose --layers 3 --nodes 128 --pde $1 --update_rule lrannealing --aggregate_boundaries
bsub -R "rusage[mem=4500,ngpus_excl_p=1]" python train.py --verbose --layers 3 --nodes 128 --pde $1 --update_rule lrannealing --resample --aggregate_boundaries

bsub -R "rusage[mem=5500,ngpus_excl_p=1]" python train.py --verbose --layers 3 --nodes 256 --pde $1 --update_rule lrannealing
bsub -R "rusage[mem=5500,ngpus_excl_p=1]" python train.py --verbose --layers 3 --nodes 256 --pde $1 --update_rule lrannealing --resample
bsub -R "rusage[mem=5500,ngpus_excl_p=1]" python train.py --verbose --layers 3 --nodes 256 --pde $1 --update_rule lrannealing --aggregate_boundaries
bsub -R "rusage[mem=5500,ngpus_excl_p=1]" python train.py --verbose --layers 3 --nodes 256 --pde $1 --update_rule lrannealing --resample --aggregate_boundaries
