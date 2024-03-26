pip install -r requirements.txt --no-deps

To reproduce results run the following commands organized by the results in Table 1:

Pytorch MAML ConvSNN+hard reset Double NMNIST:
python train.py --output-folder='logs/decolle_comp' --benchmark='doublenmnistsequence' --meta-lr=.2e-2 --step-size=1 --num-steps=1 --num-workers=10 --num-epochs=10 --num-batches=200 --num-shots=1 --batch-size=1 --num-batches-test=20 --params_file='logs/decolle_comp/e500_cnn_reset/params.yml' --device=0 --do-test --load-model='logs/decolle_comp/e500_cnn_reset/model.th'


Pytorch MAML ConvSNN+quantized Double NMNIST:
python train.py --output-folder='logs/decolle_comp' --benchmark='doublenmnistsequence' --meta-lr=.2e-2 --step-size=1 --num-steps=1 --num-workers=10 --num-epochs=10 --num-batches=200 --num-shots=1 --batch-size=1 --num-batches-test=20 --params_file='logs/decolle_comp/e250_cnn_qonly/params.yml' --device=0 --do-test --quantize=8 --quantize_in=8 --load-model='logs/decolle_comp/e250_cnn_qonly/model.th'

Pytorch MAML ConvSNN+hard reset+quantized Double NMNIST:
python train.py --output-folder='logs/decolle_comp' --benchmark='doublenmnistsequence' --meta-lr=.2e-2 --step-size=1 --num-steps=1 --num-workers=10 --num-epochs=10 --num-batches=200 --num-shots=1 --batch-size=1 --num-batches-test=20 --params_file='logs/decolle_comp/e250_cnn_reset_qin/params.yml' --device=0 --do-test --quantize=8 --quantize_in=8 --load-model='logs/decolle_comp/e250_cnn_reset_qin/model.th'

Pytorch MAML Lava-dl Double NMNIST:
python train_lava.py --output-folder='logs/dnmnistlava_sweep' --benchmark='doublenmnistlava' --meta-lr=.002e-2 --step-size=.01 --num-steps=1 --num-workers=10 --num-epochs=10 --num-batches=200 --num-shots=1 --batch-size=1 --num-batches-test=20 --params_file='logs/dnmnistlava_sweep/e_cnn_best/params.yml' --device=0 --do-test --load-model='logs/dnmnistlava_sweep/e_cnn_best/model.th'

Pytorch MAML Lava-dl SOEL Double NMNIST:
python train_lava.py --output-folder='logs/dnmnist_soel' --benchmark='doublenmnistlava' --meta-lr=.002e-2 --step-size=1 --num-steps=5 --num-workers=0 --num-epochs=10 --num-batches=200 --num-shots=1 --batch-size=1 --num-batches-test=20 --params_file='logs/dnmnistlava_sweep/best_e500/params.yml' --device=0 --do-test --load-model='logs/dnmnistlava_sweep/best_e500/model.th' --use-soel

Pytorch MAML ConvSNN+hard reset Double ASL:
python train.py --output-folder='logs/decolle_comp_asl' --benchmark='doubledvssignsequence' --meta-lr=.2e-2 --step-size=1 --num-steps=1 --num-workers=10 --num-epochs=10 --num-batches=200 --num-shots=1 --batch-size=1 --num-batches-test=20 --params_file='logs/decolle_comp_asl/e144_reset/params.yml' --device=0 --do-test --load-model='logs/decolle_comp_asl/e144_reset/model.th'

Pytorch MAML ConvSNN+quantize Double ASL:
python train.py --output-folder='logs/decolle_comp_asl' --benchmark='doubledvssignsequence' --meta-lr=.2e-2 --step-size=1 --num-steps=1 --num-workers=10 --num-epochs=10 --num-batches=200 --num-shots=1 --batch-size=1 --num-batches-test=20 --params_file='logs/decolle_comp_asl/e200_qonly/params.yml' --device=0 --do-test --quantize=8 --quantize_in=8 --load-model='logs/decolle_comp_asl/e200_qonly/model.th'

Pytorch MAML ConvSNN+hard reset+quantized Double ASL:
python train.py --output-folder='logs/decolle_comp_asl' --benchmark='doubledvssignsequence' --meta-lr=.2e-2 --step-size=1 --num-steps=1 --num-workers=10 --num-epochs=10 --num-batches=200 --num-shots=1 --batch-size=1 --num-batches-test=20 --params_file='logs/decolle_comp_asl/e125_reset_quant/params.yml' --device=0 --do-test --load-model='logs/decolle_comp_asl/e125_reset_quant/model.th'

Pytorch MAML MAML Lava-dl Double ASL:
python train_lava.py --output-folder='logs/asl_soel' --benchmark='doubledvssignlava' --meta-lr=.002e-2 --step-size=1 --num-steps=5 --num-workers=0 --num-epochs=10 --num-batches=200 --num-shots=1 --batch-size=1 --num-batches-test=20 --params_file='logs/dvssignlava_sweep/e200_mlp/params.yml' --device=0 --do-test --load-model='logs/dvssignlava_sweep/e200_mlp/model.th'

Pytorch MAML MAML Lava-dl SOEL Double ASL:
python train_lava.py --output-folder='logs/asl_soel' --benchmark='doubledvssignlava' --meta-lr=.002e-2 --step-size=1 --num-steps=5 --num-workers=0 --num-epochs=10 --num-batches=200 --num-shots=1 --batch-size=1 --num-batches-test=20 --params_file='logs/dvssignlava_sweep/e200_mlp/params.yml' --device=0 --do-test --load-model='logs/dvssignlava_sweep/e200_mlp/model.th' --use-soel


Pytorch MAML ConvSNN+hard reset DvsGesture actor+class:
python train.py --output-folder='logs/decolle_gesture' --benchmark='dvsgesturemeta' --meta-lr=.002e-2 --step-size=1 --num-steps=1 --num-workers=0 --num-epochs=10 --num-batches=200 --num-shots=1 --batch-size=1 --num-batches-test=20 --params_file='logs/decolle_gesture/e100_mlp_reset/params.yml' --device=0 --do-test --load-model='logs/decolle_gesture/e100_mlp_reset/model.th'

Pytorch MAML ConvSNN+quantize DvsGesture actor+class:
python train.py --output-folder='logs/decolle_gesture' --benchmark='dvsgesturemeta' --meta-lr=.002e-2 --step-size=1 --num-steps=1 --num-workers=0 --num-epochs=10 --num-batches=200 --num-shots=1 --batch-size=1 --num-batches-test=20 --params_file='logs/decolle_gesture/e200_qonly/params.yml' --device=0 --do-test --quantize=8 --quantize_in=8 --load-model='logs/decolle_gesture/e200_qonly/model.th'

Pytorch MAML ConvSNN+hard reset+quantized DvsGesture actor+class:
python train.py --output-folder='logs/decolle_gesture' --benchmark='dvsgesturemeta' --meta-lr=.2e-2 --step-size=1 --num-steps=1 --num-workers=10 --num-epochs=10 --num-batches=200 --num-shots=1 --batch-size=1 --num-batches-test=20 --params_file='logs/decolle_gesture/e250_qs/params.yml' --device=0 --do-test --quantize_in=8s --load-model='logs/decolle_gesture/e250_qs/model.th'

Pytorch MAML MAML Lava-dl DvsGesture actor+class:
python train_lava.py --output-folder='logs/gesture_lava' --benchmark='dvsgesturemetalava' --meta-lr=.002e-2 --step-size=1 --num-steps=5 --num-workers=0 --num-epochs=10 --num-batches=200 --num-shots=1 --batch-size=1 --num-batches-test=20 --params_file='logs/gesture_lava/e200_mlp/params.yml' --device=0 --do-test --load-model='logs/gesture_lava/e200_mlp/model.th'


Pytorch MAML MAML Lava-dl SOEL DvsGesture actor+class:
python train_lava.py --output-folder='logs/gesture_lava' --benchmark='dvsgesturemetalava' --meta-lr=.002e-2 --step-size=1 --num-steps=5 --num-workers=0 --num-epochs=10 --num-batches=200 --num-shots=1 --batch-size=1 --num-batches-test=20 --params_file='logs/gesture_lava/e200_mlp/params.yml' --device=0 --do-test --load-model='logs/gesture_lava/e200_mlp/model.th' --use-soel
