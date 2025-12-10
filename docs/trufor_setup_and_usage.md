```markdown
# TruFor Model Setup and Testing

## 1️⃣ Clone the Repository
Clone the TruFor repo into your project directory:

```bash
git clone <repository_url> AIDetective/src/models/pretrained/TruFor
```

> Replace `<repository_url>` with the URL of the TruFor repository in your GitHub.

---

## 2️⃣ Download Pretrained Weights
Download the TruFor weights from the official source:

[TruFor Weights Download](https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip)

---

## 3️⃣ Extract the Weights
Unzip the downloaded file into the `test_docker` folder so that the path to the weights is:

```
AIDetective/src/models/pretrained/TruFor/test_docker/weights/trufor.pth.tar
```

---

## 4️⃣ Run TruFor on an Image
Use the following command to test TruFor on your image:

```bash
python3 AIDetective/scripts/test_trufor.py \
    -i <image_path> \
    -o <output_directory_path>
```

- `-i <image_path>` → Path to the input image (single image, folder, or glob pattern).  
- `-o <output_directory_path>` → Directory where results will be saved.  

---

## 5️⃣ Examples of Using TruFor

#### Test a Single Image
```bash
python test_trufor.py -i data/test_image.jpg -o results/
```

#### Test All Images in a Directory
```bash
python test_trufor.py -i data/images/ -o results/
```

#### Test Using a Glob Pattern
```bash
python test_trufor.py -i "data/**/*.jpg" -o results/
```

#### Visualize Results
```bash
python test_trufor.py -i data/test_image.jpg -o results/ --visualize
```

- Adds heatmaps or overlays showing detected forgery areas
```
