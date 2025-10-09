# Download & materialize ARTeFACT locally
Since ARTeFACT is [hosted on Hugging Face](https://huggingface.co/datasets/danielaivanova/damaged-media), we can use the Hugging Face `datasets` API to pull the data once and save images/masks to disk. 
For this we can run `src/save_as_mmseg.py` to create our raw dataset consisting of base images accompanied with their respective single-channel, indexed label maps (not RGB). Here, each pixel stores a class ID:

- 0 = Clean
- 1..15 = 15 damage classes
- 255 = Background / ignore

Make sure the dataset was saved to src/data/artefact_raw. Here is a preview of the resulting tree structure:

```
.
├── src
│   ├── data
│   │   └── artefact_raw
│   │       ├── images
│   │       │   ├── cljmrkz5n341f07clcujw105j.png
│   │       │   ├── cljmrkz5n341j07cl3vqi38b4.png
│   │       │   ├── cljmrkz5n341n07clf1u410ed.png
│   │       ├── masks
│   │       │   ├── cljmrkz5n341f07clcujw105j.png
│   │       │   ├── cljmrkz5n341j07cl3vqi38b4.png
│   │       │   ├── cljmrkz5n341n07clf1u410ed.png
│   │       └── metadata.csv
```

# MMSeg style folder with splits
MMSeg expects this shape:
``` 
data/artefact/
  images/{train,val,test}/*.png
  masks/{train,val,test}/*.png
``` 
We can run `make_mmseg_layout.py` to make this layout: