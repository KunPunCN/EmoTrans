for dataset in 'dialogues'
do
  for dataset2 in 'emory'
  do
    for unseen in 15
    do
      for seed in 7
      do
          for k in 2
          do
          python -u main_crf.py \
          --gpu_available 7 \
          --unseen ${unseen} \
          --k ${k} \
          --dataset ${dataset} \
          --dataset2 ${dataset2} \
          --seed ${seed} \
          --train_batch_size 4 \
          --evaluate_batch_size 4 \
          --epochs 10 \
          --lr 1e-5 \
          --warm_up 100 \
          --pretrained_model_name_or_path yourpath/bert-base-uncased \
          --add_auto_match False
          done
      done
    done
  done
done
