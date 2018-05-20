Run in root directory:
```
$ CUDA_VISIBLE_DEVICES=0 py predict_fcn.py
$ py predict_fcn.py -m 01_demo -i demo/test.tif -o results/demo_test.png
```

You can watch model training with
```
$ tensorboard --logdir logs --port <any free port>
```
