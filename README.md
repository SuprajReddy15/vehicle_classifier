# 🚗 VehicleVerse - Professional Vehicle Classification System

VehicleVerse is an advanced vehicle image classification system built using deep learning and full-stack technologies. It classifies different types of vehicles using a trained PyTorch model and provides an intuitive web interface for users to upload and classify images instantly.

---

## 🧠 Trained Model Files (Download Required)

Due to GitHub’s 100MB file size limit, the trained `.pth` model files are hosted externally.

📦 **Download the models from Google Drive**:  
➡️ [Click here to download model weights](https://drive.google.com/drive/folders/11rI4foboJqEg37u8KfRNIF7SXhvEBE9D?usp=sharing)

### 🔧 How to Use Model Files:

1. Download all `.pth` files from the Drive folder
2. Place them inside your project structure like this:

```
vehicle_classifier/
├── app.py
├── model/
│   ├── best_vehicle_classifier.pth
│   ├── checkpoint_epoch_5.pth
│   ├── ...
```

---

## 💻 Tech Stack

| Layer        | Technologies                         |
|-------------|--------------------------------------|
| 🔙 Backend   | Python, Flask, PyTorch               |
| 🧠 ML        | Custom CNNs, Transfer Learning (optional) |
| 🧮 Utilities | NumPy, Pandas, Matplotlib            |
| 🌐 Frontend  | HTML, CSS, JavaScript (Jinja2)       |
| 🗃️ Storage   | Git LFS for large files (excluded from GitHub) |

---

## 🚀 Features

- Upload vehicle images and classify them instantly
- Real-time predictions using trained PyTorch models
- Organized logs, results, and datasets
- Clean, modular project structure
- Easily extendable with new models or frontend improvements

---

## 📂 Project Structure

```
vehicle_classifier/
├── app.py
├── config.py
├── dataset_manager.py
├── model_architecture.py
├── trainer.py
├── run_pipeline.py
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── uploads/
├── logs/
│   ├── pipeline.log
│   └── training.log
├── results/
│   ├── dataset_analysis.png
│   ├── dataset_statistics.json
│   └── dataset_report.html
├── model/  ← (place `.pth` files here)
└── requirements.txt
```

---

## ⚙️ How to Run

1. **Install requirements**  
   *(Use a virtual environment if possible)*

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask app**

   ```bash
   python app.py
   ```

3. Open your browser and go to:  
   `http://127.0.0.1:5000`

---

## 📈 Training Pipeline (Optional)

You can re-train or fine-tune the model using your own dataset:

```bash
python run_pipeline.py
```

Logs and results will be saved in the `logs/` and `results/` folders.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📃 License

This project is open source and free to use under the MIT License.

---

## 🙌 Credits

Developed by Supraj Reddy  
Guided by real-world deployment and ML engineering best practices

## 🌐 Live Demo

Try the app here:  
👉 [Vehicle Classifier Web App (Hosted on Render)](https://vehicle-classifier-h9yn.onrender.com)

