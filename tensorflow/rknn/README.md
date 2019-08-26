```                
 _______     ___  ____   ____  _____  ____  _____  
|_   __ \   |_  ||_  _| |_   \|_   _||_   \|_   _| 
  | |__) |    | |_/ /     |   \ | |    |   \ | |   
  |  __ /     |  __'.     | |\ \| |    | |\ \| |   
 _| |  \ \_  _| |  \ \_  _| |_\   |_  _| |_\   |_  
|____| |___||____||____||_____|\____||_____|\____|     
```
## Usage
1. Please download pre-trained model
```
cd NYU_FCRN-checkpoint
./get_model.sh
```
2. Freeze model to pb file using 
```
./save_pb.py
```

3. Transform pb to rknn model 
```
./pb_to_rknn.py -i NYU_FCRN-checkpoint/freeze_NYU_FCRN.pb -o NYU_FCRN-checkpoint/freeze_NYU_FCRN.rknn
```

4. Evaluate performance for the model in NPU 
```
./rknn_perf.py -i original.jpg -r NYU_FCRN-checkpoint/freeze_NYU_FCRN.rknn
```

5. Run test in 1808
```
./rknn_test.py -i original.jpg -r NYU_FCRN-checkpoint/freeze_NYU_FCRN.rknn
```

### Announcements
1. rknn-toolkit require version 1.2.0 and above
