for seed in 12345 23451 34512 45123 51234 67890 78906 89067 90678 6789; do
    python train_PrefPPO.py --env metaworld_button-press-v2 --seed $seed  --lr 0.0003 --batch-size 128 --n-envs 32 --ent-coef 0.0 --n-steps 250 --total-timesteps 3000000 --num-layer 3 --hidden-dim 256 --clip-init 0.4 --gae-lambda 0.92 --re-feed-type 1 --re-num-interaction 5000 --teacher-beta -1 --teacher-gamma 1 --teacher-eps-mistake 0 --teacher-eps-skip 0 --teacher-eps-equal 0.1 --re-segment 25 --unsuper-step 32000 --unsuper-n-epochs 50 --re-max-feed 20000 --re-batch 128 --re-update 50 --num_ratings $1
done
