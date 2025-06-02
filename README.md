# 🚗 Vehicle Image Classifier using Deep Learning

A deep learning-based web application that classifies vehicle images into one of four categories: **Car**, **Truck**, **Motorbike**, or **Bicycle**. Built using **PyTorch**, **Flask**, and **HTML/CSS**, this project demonstrates the full flow from training a model to deploying it on a web interface.

---

## 📸 Demo

> Upload an image of a vehicle and get its predicted class instantly.

![App Screenshot](screenshots/sample_output.png) <!-- (Replace with your screenshot path) -->

---

## 🔧 Tech Stack

- **Frontend**: HTML, CSS, TailwindCSS (optional), JavaScript
- **Backend**: Python, Flask
- **Deep Learning**: PyTorch, torchvision
- **Model**: Pretrained CNN (ResNet18 fine-tuned)
- **Tools**: Git, VS Code / PyCharm

---

## 🚀 Features

- Upload a vehicle image via UI
- Predict the vehicle type: `Car`, `Truck`, `Motorbike`, or `Bicycle`
- Display result with predicted label
- Modular folder structure
- Contact section and footer info personalized

---

## 📁 Project Structure

vehicle_jod/
├── static/ # CSS, JS, images
├── templates/ # HTML files
│ └── index.html # Main UI
├── model/ # Saved PyTorch model
│ └── vehicle_model.pth
├── app.py # Flask backend
├── utils.py # Prediction and helper functions
└── README.md # This file


---

## ⚙️ How to Run Locally

1. **Clone the Repository**  
```bash
git clone https://github.com/SuprajReddy15/vehicle_classifier.git
cd vehicle_classifier
Create a Virtual Environment

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # On Windows
# or
source venv/bin/activate  # On Mac/Linux
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Flask App

bash
Copy
Edit
python app.py
Visit in Browser
Open http://127.0.0.1:5000 and upload a vehicle image.

🔮 Future Improvements
 Drag-and-drop image upload

 Confidence score display

 Prediction history log

 React frontend (optional)

 Mobile app (React Native / Flutter)

🧠 Model Info
Architecture: ResNet18 (transfer learning)

Trained on: Custom Kaggle dataset (Car, Truck, Motorbike, Bicycle)

Accuracy: ~94% on validation set

📬 Contact
Thamadapally Supraj Reddy
Email: supu1513reddy@gmail.com
GitHub: github.com/SuprajReddy15
LinkedIn: View Profile

📜 License
This project is for academic and demonstration purposes.

🙏 Acknowledgements
PyTorch

Flask

Kaggle Dataset

vbnet
Copy
Edit

